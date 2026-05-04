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

# Cursive-ish font baked into PyMuPDF's HIVE-DOC table is "Courier", and
# fitz can't load Segoe Script directly without insertFont. We register
# Segoe Script per page on insertion below if the file exists.
WIN_FONTS = Path("C:/Windows/Fonts")
HAND_FONT = WIN_FONTS / "segoesc.ttf"
HAND_FONT_BOLD = WIN_FONTS / "segoescb.ttf"


# ── Per-archetype student numbers ───────────────────────────────────
SN_A = "2400000001"
SN_B = "2400000002"
SN_C_FILLED_DIGITS = "2400......"  # only first 4 digits bubbled


# ── Coordinate helpers ──────────────────────────────────────────────

class Scaler:
    """Convert map.json's pixel coords (e.g. 756 × 1086) into the blank
    PDF page's point coords (e.g. 595 × 842 for A4 portrait). One
    Scaler per PDF page; we recompute it per page because the map's
    pageWidth / pageHeight may differ between pages in principle (in
    practice they don't, but be defensive)."""

    def __init__(self, map_w: float, map_h: float, pdf_w: float, pdf_h: float) -> None:
        self.sx = pdf_w / map_w
        self.sy = pdf_h / map_h
        self.map_w = map_w
        self.map_h = map_h
        self.pdf_w = pdf_w
        self.pdf_h = pdf_h

    def box(self, b: Dict[str, float]) -> fitz.Rect:
        return fitz.Rect(
            b["x"] * self.sx,
            b["y"] * self.sy,
            (b["x"] + b["w"]) * self.sx,
            (b["y"] + b["h"]) * self.sy,
        )

    def point(self, x: float, y: float) -> fitz.Point:
        return fitz.Point(x * self.sx, y * self.sy)

    def avg_scale(self) -> float:
        return (self.sx + self.sy) / 2


# ── Drawing primitives ──────────────────────────────────────────────

def fill_bubble(page: fitz.Page, rect: fitz.Rect, *, scribbled: bool = False) -> None:
    """Fill a bubble option box. When scribbled, draw the fill then
    cross it out with a couple of overlapping diagonal lines (mimics
    a student changing their mind and crossing the bubble out)."""
    cx = (rect.x0 + rect.x1) / 2
    cy = (rect.y0 + rect.y1) / 2
    r = min(rect.width, rect.height) / 2 * 0.85
    page.draw_circle(
        fitz.Point(cx, cy), r,
        color=(0, 0, 0), fill=(0, 0, 0), width=0.5,
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


def draw_handwriting(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    *,
    font_name: str = "segoesc",
    font_path: Optional[Path] = None,
    fontsize: float = 14,
    seed: int = 0,
    color: Tuple[float, float, float] = (0.05, 0.05, 0.18),
) -> None:
    """Insert text inside `rect` using a handwriting font, with slight
    baseline jitter so it doesn't read as typed. `font_name` is the
    label PyMuPDF uses; we register Segoe Script on first use per page."""
    if font_path and font_path.exists():
        # Register the font on this page (idempotent — fitz handles repeat).
        try:
            page.insert_font(fontname=font_name, fontfile=str(font_path))
        except Exception:
            pass

    rng = random.Random(seed)
    # Word-wrap by characters (cheap, good enough for cursive).
    chars_per_line = max(1, int(rect.width / (fontsize * 0.45)))
    lines: List[str] = []
    line = ""
    for w in text.split():
        if len(line) + len(w) + 1 <= chars_per_line:
            line = (line + " " + w).strip()
        else:
            if line: lines.append(line)
            line = w
    if line: lines.append(line)

    line_h = fontsize * 1.25
    y = rect.y0 + fontsize + 2
    for ln in lines:
        if y > rect.y1: break
        # Try the registered handwriting font first; fall back to a
        # built-in if PyMuPDF can't find the glyph.
        try:
            page.insert_text(
                fitz.Point(rect.x0 + 4 + rng.uniform(-1.5, 1.5), y + rng.uniform(-1.5, 1.5)),
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
    """Fill the SN bubble grid. The real map's `studentNumberBoxes`
    array (when present) is per-digit-row; archetype C only fills the
    first 4 digits (rest left blank → OMR can't decode → unknown_X)."""
    boxes = page_map.get("studentNumberBoxes")
    if not boxes:
        return  # No SN region on this page.

    digits_to_fill = student_number
    if archetype == "C":
        digits_to_fill = student_number[:4]  # leave rest blank

    # Boxes layout: 10 cols × 10 rows. The schema we've seen lays out
    # one Box per *digit position*; if instead it lays out per
    # individual bubble (100 boxes), we treat it differently.
    if len(boxes) == 10:
        # One column rect per digit position. Draw the bubble at the
        # digit-th vertical slice of the column's height.
        for col, digit_char in enumerate(digits_to_fill):
            if not digit_char.isdigit():
                continue
            digit_val = int(digit_char)
            col_rect = scaler.box(boxes[col])
            cell_h = col_rect.height / 10
            cy = col_rect.y0 + cell_h * digit_val + cell_h / 2
            cx = (col_rect.x0 + col_rect.x1) / 2
            r = min(col_rect.width, cell_h) / 2 * 0.6
            page.draw_circle(
                fitz.Point(cx, cy), r,
                color=(0, 0, 0), fill=(0, 0, 0), width=0.4,
            )
    elif len(boxes) == 100:
        # Per-bubble. Index = col * 10 + row (assumed; map convention).
        for col, digit_char in enumerate(digits_to_fill):
            if not digit_char.isdigit(): continue
            row = int(digit_char)
            idx = col * 10 + row
            if idx >= len(boxes): continue
            r_box = scaler.box(boxes[idx])
            cx = (r_box.x0 + r_box.x1) / 2
            cy = (r_box.y0 + r_box.y1) / 2
            r = min(r_box.width, r_box.height) / 2 * 0.7
            page.draw_circle(
                fitz.Point(cx, cy), r,
                color=(0, 0, 0), fill=(0, 0, 0), width=0.4,
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
        chosen = student_answer.get("option")
        # Scribbled treatment: also fill the wrong neighbor first.
        if scribbled and chosen:
            wrong_letter = next((l for l in "ABCD" if l != chosen and l in opts), None)
            if wrong_letter:
                fill_bubble(page, scaler.box(opts[wrong_letter]), scribbled=True)
        if chosen and chosen in opts:
            fill_bubble(page, scaler.box(opts[chosen]))

    elif qtype == "multi_select":
        opts = qmap.get("options", {}) or {}
        selected = student_answer.get("options") or []
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
        # Mid-tier (B) gets a smaller font on the third question per
        # page to challenge the OCR.
        size = 13 if archetype == "B" and int(qNum) % 3 == 0 else 14
        draw_handwriting(
            page, scaler.box(sa), text,
            font_path=HAND_FONT, fontsize=size, seed=int(qNum) * 7,
        )

    elif qtype == "matching":
        boxes = qmap.get("answerBoxes", {}) or {}
        matches = student_answer.get("correctMatches", {}) or {}
        for k, letter in matches.items():
            if k not in boxes or not letter:
                continue
            r = scaler.box(boxes[k])
            # Center a single big cursive letter.
            draw_handwriting(
                page,
                fitz.Rect(r.x0, r.y0 - 2, r.x1, r.y1),
                letter,
                font_path=HAND_FONT_BOLD if HAND_FONT_BOLD.exists() else HAND_FONT,
                fontsize=min(r.height * 0.7, 22),
                seed=int(qNum) * 11 + ord(letter),
            )

    elif qtype == "fill_blanks":
        boxes = qmap.get("answerBoxes", {}) or {}
        blanks = student_answer.get("correctBlanks", {}) or {}
        for k, word in blanks.items():
            if k not in boxes or not word:
                continue
            r = scaler.box(boxes[k])
            draw_handwriting(
                page, r, word,
                font_path=HAND_FONT,
                fontsize=min(r.height * 0.55, 13),
                seed=int(qNum) * 13 + sum(ord(c) for c in str(k)),
            )


# ── Top-level per-archetype generator ───────────────────────────────

def make_archetype(slug: str, archetype: str) -> Path:
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
        scaler = Scaler(map_w, map_h, page.rect.width, page.rect.height)

        # Student number on whichever page carries the SN region (page 1
        # in the v6 schema).
        if page_map.get("studentNumberRegion") or page_map.get("studentNumberBoxes"):
            render_student_number(page, page_map, scaler, sn, archetype)

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
                p = make_archetype(slug, a)
                print(f"[ok] {p.relative_to(REPO)}  ({p.stat().st_size // 1024} KB)")
            except FileNotFoundError as e:
                print(f"[skip] {slug} student-{a}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
