"""
make_real_mock_students.py — overlay synthesized handwriting / bubble
fills onto the app's blank exam PDF, producing 3 filled student PDFs
per exam (Top / Mid / Struggling archetypes).

This supersedes make_mock_student.py. Instead of synthesizing a
PDF from scratch with our own coordinate system, we **start from the
app's real blank PDF** (the one a teacher would print + hand to
students) and draw on top using coordinates from the real map.json
that the app's mapGenerator wrote during Preview → Save.

That way the alignment matches what a scanned student paper would
look like in production — same bullseye anchor positions, same
question bbox layout, same student-number bubble grid.

Workflow expected:

  Phase A (manual, done by the user once per exam in the app):
    Edit → Step 2 Preview → Save  →  writes <id>.map.json
    Edit → Step 2 Preview → Print/Export  →  saves blank PDF to
        iku-exam-backend/samples/blanks/<id>.pdf

  Phase B (this script):
    py scripts/make_real_mock_students.py <id>
        → samples/mock-students/<id>-student-{A,B,C}.pdf

Archetypes (qNum-deterministic, NOT random — same input → same output):

  A — Top scorer
      All correct. Neat handwriting. Valid student number 2400000001.

  B — Mid-tier with OCR + remap challenges
      ~70 % correct. One crossed-out bubble per page (ink scribble
      across a previously selected option, then the right one filled).
      Smaller / messier handwriting on one open-ended box. One MS
      with one extra tick (partial credit). Valid SN 2400000002.

  C — Struggling + edge cases
      ~30 % correct. Several blank questions. **Garbled student
      number** (only 4 of 10 digits filled — exercises the
      edit-student-number ✎ flow).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF — already in requirements.txt

# ── Paths ────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
USER_DATA = Path(os.environ.get("APPDATA", "")) / "iku-exam-generator" / "exams"
BLANKS_DIR = REPO / "samples" / "blanks"
OUT_DIR = REPO / "samples" / "mock-students"

# ── PDF page geometry ────────────────────────────────────────────────
# The renderer's `.page` CSS is 200mm × 287mm. printToPDF lands that
# rectangle at the top-left of an A4 sheet (margins:0). So map.json's
# pixel coords (756×1086 = the screen-px size of the .page) MUST map
# onto the 200mm × 287mm rectangle in PDF points, NOT the full
# 596×842-pt A4 page. The earlier scaler was using page.rect.width /
# height which caused a ~5%/3.5% drift — every overlay (matching
# letters, MC bubbles, SN bubbles) ended up shifted right + down.
PAGE_W_MM = 200.0
PAGE_H_MM = 287.0
PT_PER_MM = 72.0 / 25.4  # 2.834645 pt / mm
PAGE_W_PT = PAGE_W_MM * PT_PER_MM   # ≈ 567.0 pt
PAGE_H_PT = PAGE_H_MM * PT_PER_MM   # ≈ 813.5 pt

# Handwriting font choice matters a LOT for OCR. The earlier choice of
# Segoe Script (`segoesc.ttf`) produces a cursive scrawl that TrOCR
# (microsoft/trocr-base-handwritten) misreads — e.g. "A stack" gets
# hallucinated as "displaystyle". Segoe Print (`segoepr.ttf`) is a
# clean printed-handwriting style much closer to what handwritten-OCR
# models are trained on, and the digit-CNN reads its glyphs at much
# higher confidence than the scribbly Script variant.
WIN_FONTS = Path("C:/Windows/Fonts")
HAND_FONT = WIN_FONTS / "segoepr.ttf"
HAND_FONT_BOLD = WIN_FONTS / "segoeprb.ttf"


# ── Per-archetype student numbers ───────────────────────────────────
SN_A = "2400000001"
SN_B = "2400000002"
SN_C_FILLED_DIGITS = "2400......"  # only first 4 digits bubbled

# ── Per-archetype full names (overlaid on the Full Name box) ─────────
# C is intentionally a struggling student so the name comes out a
# little messy too — same archetype letter for reproducibility.
NAMES: Dict[str, str] = {
    "A": "Ayse Yilmaz",
    "B": "Mehmet Demir",
    "C": "Eren Keles",
}


# ── Coordinate helpers ──────────────────────────────────────────────

class Scaler:
    """Convert map.json's pixel coords (e.g. 756 × 1086) into the
    rendered PDF point coords. Two transforms apply:

      1. **Scale**: `.page` CSS is 200mm × 287mm, so map px → PDF pt
         uses sx = 567/756, sy = 813.5/1086 (≈ 0.75).
      2. **Offset**: `.page` is centered horizontally + vertically in
         the A4 sheet by Chrome's print engine, so we add ~14pt on
         each axis. We don't hardcode this — instead, we measure the
         actual offset from the four printed bullseye anchors via
         `from_page()` so the math survives any future printing
         engine that aligns top-left or with different margins.

    Without the offset, every overlay drifted ~14pt left + 15pt up,
    putting cursive answers above the printed boxes."""

    def __init__(
        self,
        map_w: float,
        map_h: float,
        pdf_w: float,
        pdf_h: float,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ) -> None:
        self.sx = pdf_w / map_w
        self.sy = pdf_h / map_h
        self.map_w = map_w
        self.map_h = map_h
        self.pdf_w = pdf_w
        self.pdf_h = pdf_h
        self.offset_x = offset_x
        self.offset_y = offset_y

    def box(self, b: Dict[str, float]) -> fitz.Rect:
        return fitz.Rect(
            b["x"] * self.sx + self.offset_x,
            b["y"] * self.sy + self.offset_y,
            (b["x"] + b["w"]) * self.sx + self.offset_x,
            (b["y"] + b["h"]) * self.sy + self.offset_y,
        )

    def point(self, x: float, y: float) -> fitz.Point:
        return fitz.Point(x * self.sx + self.offset_x, y * self.sy + self.offset_y)

    def avg_scale(self) -> float:
        return (self.sx + self.sy) / 2


# ── Drawing primitives ──────────────────────────────────────────────

def fill_bubble(page: fitz.Page, rect: fitz.Rect, *, scribbled: bool = False) -> None:
    """Fill a bubble option box for OMR detection.

    The printed mcCircle is 12 px × 12 px in the renderer (9 pt × 9 pt
    after scale-to-PDF) with a 2 px / 1.5 pt border. The inner usable
    area is ~6 pt diameter. We draw a filled circle that nearly fills
    the inner area (radius = 0.7 × outer-radius) so OMR's dark-ratio
    detector sees a clearly filled bubble without the fill bleeding
    outside the printed circle's border.

    When scribbled, the bubble is filled and then crossed out — used
    by archetype-B to simulate a student changing their answer."""
    cx = (rect.x0 + rect.x1) / 2
    cy = (rect.y0 + rect.y1) / 2
    r = min(rect.width, rect.height) / 2 * 0.70
    page.draw_circle(
        fitz.Point(cx, cy), r,
        color=(0, 0, 0), fill=(0, 0, 0), width=0,
    )
    if scribbled:
        # Two crossing slashes to simulate "I changed my mind"
        pad = r * 0.3
        page.draw_line(
            fitz.Point(rect.x0 + pad, rect.y0 + pad),
            fitz.Point(rect.x1 - pad, rect.y1 - pad),
            color=(0, 0, 0), width=1.4,
        )
        page.draw_line(
            fitz.Point(rect.x1 - pad, rect.y0 + pad),
            fitz.Point(rect.x0 + pad, rect.y1 - pad),
            color=(0, 0, 0), width=1.4,
        )


def fix_mojibake(s: str) -> str:
    """Repair UTF-8-bytes-decoded-as-Windows-1252 mojibake AND
    transliterate non-Latin-1 characters that PyMuPDF's insert_text
    can't render through `Page.insert_font`-registered TTFs.

    The `.ikuexam` / `.map.json` files written by earlier scripts went
    through a Win-1252 → UTF-8 round-trip, leaving sequences like
    `â€”` (chars: â, €, ”) where a real em-dash U+2014 should sit. The
    cleanest fix is to undo the bad decode by re-encoding as cp1252 and
    decoding as utf-8 — that recovers the original Unicode codepoints.
    Then we translit the recovered chars to ASCII because PyMuPDF's
    `insert_text` defaults to Latin-1 encoding and silently mangles
    anything outside that range (em-dashes, smart quotes, ellipsis…)."""
    if not s:
        return s
    # Stage 1: round-trip cp1252 → utf-8 to undo the mojibake.
    try:
        recovered = s.encode("cp1252", errors="strict").decode("utf-8", errors="strict")
        s = recovered
    except (UnicodeEncodeError, UnicodeDecodeError):
        # Either the string was never mojibake'd (good) or it has
        # genuinely non-CP1252 chars (Turkish letters from the source).
        # Either way, leave it.
        pass
    # Stage 2: translit Latin-1-unsafe Unicode chars to ASCII fallbacks
    # so PyMuPDF's TTF rendering doesn't re-mojibake them.
    SAFE = (
        ('—', '-'), ('–', '-'),                # em / en dash
        ('“', '"'), ('”', '"'),                # double smart quotes
        ('‘', "'"), ('’', "'"),                # single smart quotes
        ('…', '...'),                          # ellipsis
    )
    for bad, good in SAFE:
        s = s.replace(bad, good)
    return s


# Renderer's `.solutionArea` paints a ruled-line gradient at 28-px
# pitch (see `QuestionBlock.module.css`). We use double-pitch (56-px)
# spacing for the cursive overlay so each line gets generous
# vertical room — that lets us use a bigger font (clearer for OCR)
# while still landing every other baseline on a printed line.
RULE_PX = 28.0
LINE_PITCH_PX = RULE_PX * 2  # one text line per two printed rules


def draw_handwriting(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    *,
    scaler: Optional["Scaler"] = None,
    font_name: str = "segoesc",
    font_path: Optional[Path] = None,
    fontsize: Optional[float] = None,
    seed: int = 0,
    color: Tuple[float, float, float] = (0.05, 0.05, 0.18),
    align_to_rules: bool = True,
) -> None:
    """Insert text inside `rect` using a handwriting font.

    When `scaler` is provided AND `align_to_rules`, line spacing locks
    to the printed 28-px ruled lines so each cursive baseline lands on
    a printed line. Falls back to font-driven spacing otherwise."""
    if font_path and font_path.exists():
        try:
            page.insert_font(fontname=font_name, fontfile=str(font_path))
        except Exception:
            pass

    text = fix_mojibake(text or "")
    rng = random.Random(seed)

    # Pick line height first — sets the budget the font has to fit in.
    if scaler is not None and align_to_rules:
        # Double-pitched: one cursive line per two printed rules. Gives
        # the chosen Segoe Print enough room to render at ~22 pt where
        # TrOCR can actually parse the glyphs.
        line_h = LINE_PITCH_PX * scaler.sy
    else:
        line_h = (fontsize or 18) * 1.25
    if fontsize is None:
        fontsize = max(10.0, min(line_h * 0.55, 22.0))

    # Use PyMuPDF's font metrics for accurate width when available;
    # otherwise fall back to a slightly fatter heuristic than before
    # (0.55 instead of 0.45) so cursive script doesn't overflow.
    measure_font = None
    if font_path and font_path.exists():
        try:
            measure_font = fitz.Font(fontfile=str(font_path))
        except Exception:
            measure_font = None
    inner_w = max(1.0, rect.width - 8)

    def fits(ln: str) -> bool:
        if measure_font:
            try:
                return measure_font.text_length(ln, fontsize) <= inner_w
            except Exception:
                pass
        return len(ln) * fontsize * 0.55 <= inner_w

    # Word-wrap by trying to add words one at a time.
    lines: List[str] = []
    cur = ""
    for w in text.split():
        candidate = (cur + " " + w).strip() if cur else w
        if fits(candidate):
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    # Vertical capacity: how many ruled lines fit?
    capacity = max(1, int((rect.height - 4) / line_h))
    if len(lines) > capacity:
        lines = lines[:capacity]
        if lines:
            # Append ellipsis to the last line if there's room.
            last = lines[-1]
            while last and not fits(last + "…"):
                last = last[:-1]
            lines[-1] = (last + "…").strip()

    # Baseline calculation depends on whether we're aligning to ruled
    # lines (multi-line open-ended) or just dropping a short answer in
    # a small box. For ruled-line mode the baseline starts a small
    # offset below rect.y0 so descenders clear; for single-box mode we
    # vertically CENTER the visual glyph on the box midline (the
    # baseline is below the visual center because most glyph ink sits
    # above the baseline). Without this centering, every short-text
    # overlay (name, digits, fill words, matching letters) hugged the
    # top of its rect and appeared to float above the printed box.
    if align_to_rules:
        # First line: drop ~80% of fontsize below the rule so
        # descenders clear and the bulk of the glyph rests on the line.
        y = rect.y0 + fontsize * 0.95
    else:
        # Single line, visual-center it. PyMuPDF's insert_text places
        # the glyph's BASELINE at point.y. Cap height ≈ 0.7 × fontsize
        # for typical fonts, so the visual midline is ~0.35 × fontsize
        # above the baseline. To put the visual midline at the rect
        # midline: baseline = midline + 0.35 × fontsize.
        midline = (rect.y0 + rect.y1) / 2
        y = midline + fontsize * 0.32
    for ln in lines:
        if y > rect.y1 + fontsize * 0.5:  # tolerate descenders past rect
            break
        try:
            page.insert_text(
                fitz.Point(
                    rect.x0 + 4 + rng.uniform(-1.0, 1.0),
                    y + rng.uniform(-0.8, 0.8),
                ),
                ln,
                fontname=font_name,
                fontsize=fontsize,
                color=color,
            )
        except Exception:
            page.insert_text(
                fitz.Point(rect.x0 + 4, y),
                ln, fontname="helv", fontsize=fontsize, color=color,
            )
        y += line_h


# ── Map / ikuexam loading ───────────────────────────────────────────

def load_map(slug: str) -> Dict[str, Any]:
    path = USER_DATA / f"{slug}.map.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No map found at {path}. Did you Save in Preview for this exam?"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_ikuexam(slug: str) -> Dict[str, Any]:
    path = USER_DATA / f"{slug}.ikuexam"
    if not path.exists():
        raise FileNotFoundError(f"No .ikuexam at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def open_blank(slug: str) -> fitz.Document:
    path = BLANKS_DIR / f"{slug}.pdf"
    if not path.exists():
        raise FileNotFoundError(
            f"No blank PDF at {path}. Phase A: open the exam in app, "
            "Step 2 Preview → Print/Export PDF, save to that path."
        )
    return fitz.open(str(path))


# ── Archetype answer-distortion logic ───────────────────────────────

ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def deterministic_pick(qNum: int, archetype: str, n_choices: int = 10) -> int:
    """Stable hash over (qNum, archetype) → 0..n-1. Lets us
    reproducibly vary which questions get wrong / blank / scribbled
    treatment without needing an RNG that drifts between runs."""
    h = hashlib.md5(f"{archetype}:{qNum}".encode()).hexdigest()
    return int(h[:8], 16) % n_choices


def perturb(
    qtype: str,
    expected: Dict[str, Any],
    archetype: str,
    qNum: int,
) -> Tuple[Dict[str, Any], bool]:
    """Return (student_answer, scribbled).

    student_answer shape mirrors the map's expectedAnswer:
      mc    → {"option": "B"} or {"option": None} for blank
      ms    → {"options": ["A", "C"]}
      open  → {"text": "..."}
      match → {"correctMatches": {...}}  # using 'matches' here too
      fill  → {"correctBlanks": {...}}

    `scribbled` is True when student-B treatment crosses out a wrong
    bubble before picking the right one. Caller draws both the wrong
    bubble (scribbled) and the right one.
    """
    if archetype == "A":
        # Top scorer: copy expected verbatim.
        return dict(expected), False

    pick = deterministic_pick(qNum, archetype)

    if archetype == "B":
        # 70% correct; 30% wrong but plausible. One scribble per page —
        # caller decides which qNums get the scribble flag based on
        # page boundaries; here we set scribbled when pick % 4 == 0.
        scribbled = pick % 4 == 0
        if pick % 10 < 7:
            # Correct — just maybe scribbled a wrong-bubble first.
            return dict(expected), scribbled
        # Wrong-but-plausible
        if qtype == "mc":
            # Rotate to the next letter.
            cur = expected.get("correctOption", "A")
            i = ALPHA.index(cur) if cur in ALPHA else 0
            return {"option": ALPHA[(i + 1) % 4]}, scribbled
        if qtype == "ms":
            # Add one extra option (partial credit path).
            corr = list(expected.get("correctOptions", []))
            extras = [c for c in "ABCD" if c not in corr]
            if extras: corr.append(extras[0])
            return {"options": corr}, scribbled
        if qtype == "open":
            # Mid-confidence answer: keep it short + close to expected
            # so AI gives partial credit.
            text = expected.get("text", "")
            return {"text": text[:max(15, len(text) // 2)]}, scribbled
        if qtype == "match":
            # Swap two pairs.
            m = dict(expected.get("correctMatches", {}))
            keys = sorted(m.keys())
            if len(keys) >= 2:
                m[keys[0]], m[keys[1]] = m[keys[1]], m[keys[0]]
            return {"correctMatches": m}, scribbled
        if qtype == "fill":
            # Drop the last blank.
            b = dict(expected.get("correctBlanks", {}))
            if b:
                last = sorted(b.keys())[-1]
                del b[last]
            return {"correctBlanks": b}, scribbled
        return dict(expected), scribbled

    # Archetype C — struggling: 30% correct, 30% wrong, 40% blank.
    bucket = pick % 10
    if bucket < 3:
        return dict(expected), False
    if bucket < 6:
        # Wrong (same logic as B's wrong branch).
        return perturb(qtype, expected, "B", qNum + 100)[0], False
    # Blank
    if qtype == "mc": return {"option": None}, False
    if qtype == "ms": return {"options": []}, False
    if qtype == "open": return {"text": ""}, False
    if qtype == "match": return {"correctMatches": {}}, False
    if qtype == "fill": return {"correctBlanks": {}}, False
    return {}, False


# ── Per-page rendering ──────────────────────────────────────────────

def render_student_number(
    page: fitz.Page,
    page_map: Dict[str, Any],
    scaler: Scaler,
    student_number: str,
    archetype: str,
) -> None:
    """Render student-number digits as cursive glyphs inside each
    digit box.

    The renderer's SN region is 10 single-character boxes (28×34 px
    each), and the backend's `read_student_number()` runs a digit CNN
    over each box's crop. So we need a HANDWRITTEN-DIGIT glyph in each
    box, NOT a column-of-10 OMR bubble grid (an earlier version drew
    bubble grids → CNN saw black blobs → every student read as
    1111111111 with conf ≈ 0.24, which collapsed all results to one
    record on save-merge).

    Archetype C leaves the trailing digits blank to exercise the
    edit-student-number ✎ flow."""
    boxes = page_map.get("studentNumberBoxes")
    if not boxes:
        return  # No SN region on this page.

    digits_to_fill = student_number
    if archetype == "C":
        digits_to_fill = student_number[:4]  # leave rest blank

    # The map currently emits one Box per digit position (10 total for
    # the 10-char SN). If a future schema goes back to a 100-box OMR
    # grid, we just bail — the bubble fill code above was never used
    # in production and the new printable doesn't use it either.
    if len(boxes) != 10:
        return

    for col, digit_char in enumerate(digits_to_fill):
        if not digit_char.isdigit() or col >= len(boxes):
            continue
        box_rect = scaler.box(boxes[col])
        # draw_handwriting visual-centers when align_to_rules=False;
        # fontsize is sized as a fraction of the box height so the
        # digit reads clearly to the digit CNN without overflowing.
        draw_handwriting(
            page, box_rect, digit_char,
            font_path=HAND_FONT,
            fontsize=min(box_rect.height * 0.65, 18.0),
            seed=ord(archetype) * 13 + col,
            align_to_rules=False,
        )


def render_question(
    page: fitz.Page,
    qNum: str,
    qmap: Dict[str, Any],
    expected: Dict[str, Any],
    student_answer: Dict[str, Any],
    scribbled: bool,
    scaler: Scaler,
    archetype: str,
) -> None:
    """Draw the student's answer for one question on top of the
    blank PDF page."""
    qtype = qmap.get("type", "")

    if qtype == "multiple_choice":
        opts = qmap.get("options", {}) or {}
        # Accept BOTH the perturbed shape `{option: 'A'}` (for B/C
        # archetypes where perturb() picks a wrong letter) AND the
        # expected-verbatim shape `{correctOption: 'A'}` (for the A
        # archetype where perturb just dict-copies the expected). Same
        # for multi_select below. Without this fallback, archetype-A
        # rendered ZERO MC fills because the key was correctOption,
        # not option, and the OMR pipeline saw every MC as blank.
        chosen = student_answer.get("option") or student_answer.get("correctOption")
        # Scribbled treatment: also fill the wrong neighbor first.
        if scribbled and chosen:
            wrong_letter = next((l for l in "ABCD" if l != chosen and l in opts), None)
            if wrong_letter:
                fill_bubble(page, scaler.box(opts[wrong_letter]), scribbled=True)
        if chosen and chosen in opts:
            fill_bubble(page, scaler.box(opts[chosen]))

    elif qtype == "multi_select":
        opts = qmap.get("options", {}) or {}
        selected = (
            student_answer.get("options")
            or student_answer.get("correctOptions")
            or []
        )
        if scribbled and selected and opts:
            wrong = next((l for l in "ABCD" if l not in selected and l in opts), None)
            if wrong:
                fill_bubble(page, scaler.box(opts[wrong]), scribbled=True)
        for letter in selected:
            if letter in opts:
                fill_bubble(page, scaler.box(opts[letter]))

    elif qtype == "open_ended":
        sa = qmap.get("solutionArea")
        if not sa: return
        text = student_answer.get("text", "")
        if not text: return  # blank
        # Use ruled-line-locked spacing so cursive baselines align with
        # the printed gradient lines in the .solutionArea.
        draw_handwriting(
            page, scaler.box(sa), text,
            scaler=scaler, font_path=HAND_FONT, seed=int(qNum) * 7,
            align_to_rules=True,
        )

    elif qtype == "matching":
        boxes = qmap.get("answerBoxes", {}) or {}
        matches = student_answer.get("correctMatches", {}) or {}
        for k, letter in matches.items():
            if k not in boxes or not letter:
                continue
            r = scaler.box(boxes[k])
            # draw_handwriting now visual-centers when align_to_rules
            # is False, so we just hand it the box rect and a fontsize
            # tuned to leave a sensible margin.
            draw_handwriting(
                page, r, letter,
                font_path=HAND_FONT_BOLD if HAND_FONT_BOLD.exists() else HAND_FONT,
                fontsize=min(r.height * 0.72, 18.0),
                seed=int(qNum) * 11 + ord(letter),
                align_to_rules=False,
            )

    elif qtype == "fill_blanks":
        boxes = qmap.get("answerBoxes", {}) or {}
        blanks = student_answer.get("correctBlanks", {}) or {}
        for k, word in blanks.items():
            if k not in boxes or not word:
                continue
            r = scaler.box(boxes[k])
            # Center the cursive word vertically in the answer box —
            # box height after scaling is small (~26 pt for h=34 px),
            # so we let the glyph sit just inside the bottom.
            draw_handwriting(
                page, r, word,
                font_path=HAND_FONT,
                fontsize=min(r.height * 0.55, 13),
                seed=int(qNum) * 13 + sum(ord(c) for c in str(k)),
                align_to_rules=False,
            )


# ── Top-level per-archetype generator ───────────────────────────────

def _debug_outline(page: fitz.Page, rect: fitz.Rect, color=(1, 0, 0)) -> None:
    """Stroke a thin red rectangle around the given rect for visual
    verification that the script's coordinate transforms match the
    printed boxes."""
    page.draw_rect(rect, color=color, width=0.5)


def _measure_page_offset(
    page: fitz.Page, page_map: Dict[str, Any],
) -> Tuple[float, float]:
    """Detect the offset of `.page` inside the PDF page by measuring
    the printed bullseye anchors and matching them to the map's
    anchor centers.

    Returns (offset_x, offset_y) in PDF points. Defaults to centering
    `.page` within A4 if no anchors can be detected (falls back
    gracefully on PDFs without printed bullseyes)."""
    map_anchors = page_map.get("anchors") or {}
    map_tl = (map_anchors.get("TL") or {}).get("center")
    if not map_tl:
        # Fallback — assume centered.
        return (
            (page.rect.width - PAGE_W_PT) / 2,
            (page.rect.height - PAGE_H_PT) / 2,
        )

    # Find printed 12-pt squares near the corners of the PDF — those
    # are the rendered bullseye anchors.
    sx = PAGE_W_PT / page_map.get("pageWidth", 756)
    sy = PAGE_H_PT / page_map.get("pageHeight", 1086)
    near = []
    for d in page.get_drawings():
        r = d.get("rect")
        if not r:
            continue
        if 10 < r.width < 14 and 10 < r.height < 14:
            # Only corner ones (within 50pt of any corner).
            cx, cy = r.x0 + r.width / 2, r.y0 + r.height / 2
            if (
                cx < 60 or cx > page.rect.width - 60
            ) and (
                cy < 60 or cy > page.rect.height - 60
            ):
                near.append((cx, cy))

    if not near:
        return (
            (page.rect.width - PAGE_W_PT) / 2,
            (page.rect.height - PAGE_H_PT) / 2,
        )

    # Pick the top-left corner anchor (smallest x+y).
    pdf_tl = min(near, key=lambda p: p[0] + p[1])
    # offset = pdf_tl_center - map_tl_center * scale
    offset_x = pdf_tl[0] - float(map_tl["x"]) * sx
    offset_y = pdf_tl[1] - float(map_tl["y"]) * sy
    return (offset_x, offset_y)


def make_archetype(slug: str, archetype: str, debug_outlines: bool = False) -> Path:
    """Render one filled-student PDF for the given exam slug + arche."""
    map_data = load_map(slug)
    iku = load_ikuexam(slug)

    # Map qNum (string in map) → expectedAnswer + type from the map's
    # own questions dict; the .ikuexam has the same answer keys but
    # we read from the map for consistency with what the backend
    # uses at /evaluate time.
    pages_map = map_data.get("pages", []) or []

    sn = {"A": SN_A, "B": SN_B, "C": SN_A}[archetype]  # SN_A for C; we'll degrade in the SN renderer

    doc = open_blank(slug)
    if len(doc) != len(pages_map):
        print(
            f"[warn] blank PDF has {len(doc)} pages but map has "
            f"{len(pages_map)} — proceeding with min(pages, mapPages).",
            file=sys.stderr,
        )

    n_pages = min(len(doc), len(pages_map))
    for page_idx in range(n_pages):
        page = doc[page_idx]
        page_map = pages_map[page_idx]
        map_w = float(page_map.get("pageWidth", 756))
        map_h = float(page_map.get("pageHeight", 1086))
        # Use the .page CSS dimensions (200mm × 287mm = 567×813.5 pt)
        # for the SCALE component. The PDF page is A4 (596×842 pt) and
        # Chrome's print engine CENTERS .page within it, so we also need
        # an offset. Compute the offset by measuring the printed
        # bullseye-anchor positions in the actual PDF and matching them
        # to the map's recorded anchor centers — that way we're robust
        # to any future change in how the renderer aligns .page within
        # the print canvas.
        offset_x, offset_y = _measure_page_offset(page, page_map)
        scaler = Scaler(
            map_w, map_h, PAGE_W_PT, PAGE_H_PT,
            offset_x=offset_x, offset_y=offset_y,
        )

        # Student number on whichever page carries the SN region (page 1
        # in the v6 schema).
        if page_map.get("studentNumberRegion") or page_map.get("studentNumberBoxes"):
            render_student_number(page, page_map, scaler, sn, archetype)

        # Full Name — only present on page 1 of v2+ maps. The captured
        # box covers the entire .student-name-field container, which
        # has a "FULL NAME" header label up top and a writing line at
        # the bottom. We carve out the bottom 2/3 so the cursive lands
        # on the writing line rather than overlapping the header label.
        name_box = page_map.get("studentNameField")
        if name_box:
            full_name = NAMES.get(archetype, "")
            if archetype == "C":
                full_name = full_name.split(" ")[0]
            if full_name:
                box = scaler.box(name_box)
                # Bottom 60% of the box → above the writing line, below
                # the "FULL NAME" header.
                writing_top = box.y0 + box.height * 0.40
                writing_rect = fitz.Rect(
                    box.x0 + 6, writing_top, box.x1 - 6, box.y1 - 2,
                )
                draw_handwriting(
                    page, writing_rect, full_name,
                    font_path=HAND_FONT,
                    fontsize=min(writing_rect.height * 0.75, 14),
                    seed=ord(archetype),
                    align_to_rules=False,
                )

        # Per-question overlays.
        questions = page_map.get("questions", {}) or {}
        for qNum, qmap in questions.items():
            expected = qmap.get("expectedAnswer", {}) or {}
            qtype = qmap.get("type", "")
            student_answer, scribbled = perturb(qtype, expected, archetype, int(qNum))
            render_question(
                page, qNum, qmap, expected, student_answer,
                scribbled, scaler, archetype,
            )
            # Debug outlines: stroke every target rect so we can
            # visually confirm the map → PDF mapping is right.
            if debug_outlines:
                if qtype in ("multiple_choice", "multi_select"):
                    for _opt, b in (qmap.get("options") or {}).items():
                        _debug_outline(page, scaler.box(b))
                if qtype in ("matching", "fill_blanks"):
                    for _idx, b in (qmap.get("answerBoxes") or {}).items():
                        _debug_outline(page, scaler.box(b))
                if qtype == "open_ended":
                    sa = qmap.get("solutionArea")
                    if sa:
                        _debug_outline(page, scaler.box(sa), color=(0, 0.5, 0))
        if debug_outlines:
            for b in (page_map.get("studentNumberBoxes") or []):
                _debug_outline(page, scaler.box(b), color=(0, 0, 1))
            nf = page_map.get("studentNameField")
            if nf:
                _debug_outline(page, scaler.box(nf), color=(0, 0, 1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{slug}-student-{archetype}.pdf"
    doc.save(str(out_path), deflate=True)
    doc.close()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "slug",
        nargs="?",
        help="Exam id (e.g. mock-allTypes-2026). Omit for all 3 known slugs.",
    )
    parser.add_argument(
        "--archetypes", default="A,B,C",
        help="Comma list of archetypes to render (default: A,B,C)",
    )
    parser.add_argument(
        "--debug-outlines", action="store_true",
        help="Draw thin colored rects around every target box (red=MC/answer, blue=SN/name, green=open) so you can visually verify the map → PDF mapping is correct.",
    )
    args = parser.parse_args()

    DEFAULT_SLUGS = [
        "mock-allTypes-2026",
        "cse301-ds-quiz-2026",
        "mat102-calc2-midterm-2026",
    ]
    slugs = [args.slug] if args.slug else DEFAULT_SLUGS
    arches = [a.strip().upper() for a in args.archetypes.split(",") if a.strip()]

    for slug in slugs:
        for a in arches:
            try:
                p = make_archetype(slug, a, debug_outlines=args.debug_outlines)
                print(f"[ok] {p.relative_to(REPO)}  ({p.stat().st_size // 1024} KB)")
            except FileNotFoundError as e:
                print(f"[skip] {slug} student-{a}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
