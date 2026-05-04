"""
make_mock_student.py — generate a 15-question synthetic exam covering
all 5 question types (3 questions per type), spread across 3 pages.

Outputs (all under iku-exam-backend/samples/mock/):
  mock_filled_student.pdf         — 3-page filled "student submission"
  mock_filled.map.json            — backend map (long-form types)
  mock_filled.ikuexam             — frontend exam definition (short types)

The .ikuexam + .map.json pair are also COPIED into the Electron app's
userData (`%APPDATA%\\iku-exam-generator\\exams\\`) when run with
`--install`, so the exam appears in the Dashboard's My Exams grid
ready to Evaluate (no Results button — that surfaces only after a
real evaluation run produces .results.json).

Layout
  Page 1: 3× multiple_choice  + 3× multi_select   (bubble grids)
  Page 2: 3× fill_blanks      + 3× matching       (handwriting boxes)
  Page 3: 3× open_ended                           (essay-length cursive)

Run:
    .venv\\Scripts\\python.exe scripts\\make_mock_student.py
    .venv\\Scripts\\python.exe scripts\\make_mock_student.py --install
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont

# ── Paths ─────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT_DIR = REPO / "samples" / "mock"
OUT_PDF = OUT_DIR / "mock_filled_student.pdf"
OUT_MAP = OUT_DIR / "mock_filled.map.json"
OUT_IKU = OUT_DIR / "mock_filled.ikuexam"

EXAM_ID = "mock-allTypes-2026"

USER_DATA = Path(os.environ.get("APPDATA", "")) / "iku-exam-generator" / "exams"

# ── Page geometry ────────────────────────────────────────────────────
# The map.json's pageWidth/pageHeight stays at 756×1086 (the v6 sample
# convention). For *rendering* we work at a higher pixel density so the
# saved PDF reads as sharp at native size; alignment.py rescales to the
# map dimensions on the backend, so the coordinate system is unchanged.
PAGE_W = 756
PAGE_H = 1086
RENDER_SCALE = 3   # 3× = ~270 DPI on A4 — looks crisp in the workspace

ANCHORS = {
    "TL": {"center": {"x": 11, "y": 11}, "size": None},
    "TR": {"center": {"x": 745, "y": 11}, "size": None},
    "BL": {"center": {"x": 11, "y": 1075}, "size": None},
    "BR": {"center": {"x": 745, "y": 1075}, "size": None},
}
STUDENT_NUMBER_REGION = {"x": 421.4, "y": 109.7, "w": 324.6, "h": 72.3}

# ── Fonts ────────────────────────────────────────────────────────────
WIN_FONTS = Path("C:/Windows/Fonts")
HAND_FONT = next(p for p in [WIN_FONTS / "segoesc.ttf", WIN_FONTS / "LHANDW.TTF",
                              WIN_FONTS / "comic.ttf"] if p.exists())
PRINT_FONT = next(p for p in [WIN_FONTS / "calibri.ttf", WIN_FONTS / "arial.ttf",
                               WIN_FONTS / "segoeui.ttf"] if p.exists())

MOCK_STUDENT_NUMBER = "2400123456"

# ── Archetype profiles (Phase 3) ─────────────────────────────────
# When CLI passes --archetype A|B|C, we override every question's
# studentPick / studentPicks / studentBlanks / studentAnswer derived
# from `correct` according to the archetype's accuracy distribution.
# Same input → same output (deterministic via qNum hash).
ARCHETYPE: str = "DEFAULT"
ARCHETYPE_SN = {
    "A": "2400000001",  # Top scorer — clean SN
    "B": "2400000002",  # Mid-tier
    "C": "2400......",  # Struggling — only first 4 digits filled (degraded)
    "DEFAULT": MOCK_STUDENT_NUMBER,
}

import hashlib as _hashlib  # noqa: E402

def _qhash(archetype: str, qNum: int) -> int:
    h = _hashlib.md5(f"{archetype}:{qNum}".encode()).hexdigest()
    return int(h[:8], 16) % 100  # 0-99


def derive_student_answer(qtype: str, qdata: Dict[str, Any], qNum: int) -> Dict[str, Any]:
    """Return a {studentPick / studentPicks / studentBlanks / studentAnswer}
    dict overriding qdata for the active archetype. For 'DEFAULT' (no
    archetype) returns the original hardcoded values."""
    if ARCHETYPE == "DEFAULT":
        return {}  # caller uses qdata as-is

    h = _qhash(ARCHETYPE, qNum)
    correct = qdata.get("correct")

    # Archetype A — all correct
    if ARCHETYPE == "A":
        if qtype == "multiple_choice":
            return {"studentPick": correct}
        if qtype == "multi_select":
            return {"studentPicks": list(correct)}
        if qtype == "fill_blanks":
            return {"studentBlanks": dict(qdata["blanks"])}
        if qtype == "matching":
            return {"studentPicks": dict(qdata["correct"])}
        if qtype == "open_ended":
            return {"studentAnswer": qdata["expected"]}
        return {}

    # Archetype B — ~70% correct, with one mid-tier OCR challenge
    if ARCHETYPE == "B":
        if h < 70:
            # Same as A
            return derive_student_answer(qtype, qdata, qNum) if False else (
                {"studentPick": correct} if qtype == "multiple_choice" else
                {"studentPicks": list(correct)} if qtype == "multi_select" else
                {"studentBlanks": dict(qdata["blanks"])} if qtype == "fill_blanks" else
                {"studentPicks": dict(qdata["correct"])} if qtype == "matching" else
                {"studentAnswer": qdata["expected"]} if qtype == "open_ended" else {}
            )
        # 30% wrong-but-plausible
        if qtype == "multiple_choice":
            i = "ABCD".index(correct) if correct in "ABCD" else 0
            return {"studentPick": "ABCD"[(i + 1) % 4]}
        if qtype == "multi_select":
            extras = [c for c in "ABCD" if c not in correct]
            extra = extras[0] if extras else None
            picks = list(correct) + ([extra] if extra else [])
            return {"studentPicks": picks}
        if qtype == "fill_blanks":
            # Truncate one blank
            b = dict(qdata["blanks"])
            keys = sorted(b.keys())
            if keys:
                b[keys[-1]] = b[keys[-1]][:max(2, len(b[keys[-1]]) // 2)]
            return {"studentBlanks": b}
        if qtype == "matching":
            m = dict(qdata["correct"])
            keys = sorted(m.keys())
            if len(keys) >= 2:
                m[keys[0]], m[keys[1]] = m[keys[1]], m[keys[0]]
            return {"studentPicks": m}
        if qtype == "open_ended":
            short = qdata["expected"][: max(20, len(qdata["expected"]) // 2)]
            return {"studentAnswer": short}
        return {}

    # Archetype C — 30% correct, 30% wrong, 40% blank
    if ARCHETYPE == "C":
        if h < 30:
            return (
                {"studentPick": correct} if qtype == "multiple_choice" else
                {"studentPicks": list(correct)} if qtype == "multi_select" else
                {"studentBlanks": dict(qdata["blanks"])} if qtype == "fill_blanks" else
                {"studentPicks": dict(qdata["correct"])} if qtype == "matching" else
                {"studentAnswer": qdata["expected"]} if qtype == "open_ended" else {}
            )
        if h < 60:
            # Wrong (same logic as B's wrong branch)
            if qtype == "multiple_choice":
                i = "ABCD".index(correct) if correct in "ABCD" else 0
                return {"studentPick": "ABCD"[(i + 2) % 4]}
            if qtype == "multi_select":
                # Pick wrong subset
                wrongs = [c for c in "ABCD" if c not in correct][:2]
                return {"studentPicks": wrongs}
            if qtype == "fill_blanks":
                # All wrong words
                b = {k: "wrong" for k in qdata["blanks"]}
                return {"studentBlanks": b}
            if qtype == "matching":
                m = dict(qdata["correct"])
                # Cycle all by one
                keys = sorted(m.keys())
                vals = [m[k] for k in keys]
                shifted = vals[1:] + vals[:1]
                return {"studentPicks": dict(zip(keys, shifted))}
            if qtype == "open_ended":
                return {"studentAnswer": "I don't know"}
            return {}
        # 40% blank
        if qtype == "multiple_choice":
            return {"studentPick": ""}
        if qtype == "multi_select":
            return {"studentPicks": []}
        if qtype == "fill_blanks":
            return {"studentBlanks": {k: "" for k in qdata["blanks"]}}
        if qtype == "matching":
            return {"studentPicks": {k: "" for k in qdata["correct"]}}
        if qtype == "open_ended":
            return {"studentAnswer": ""}
        return {}
    return {}


# ── Question pool — 3 of each type ───────────────────────────────────
# Each entry has:
#   prompt, expected, studentAnswer(s), points
# Geometry (boundingBox, options/answerBoxes/solutionArea) is computed
# in build_layout() so the script controls the grid.

POOL: Dict[str, List[Dict[str, Any]]] = {
    "multiple_choice": [
        {"prompt": "Which Flutter widget rebuilds when its state changes?",
         "options": ["StatelessWidget", "StatefulWidget", "InheritedWidget", "RenderObject"],
         "correct": "B", "studentPick": "B"},
        {"prompt": "Which command fetches Flutter package dependencies?",
         "options": ["flutter run", "flutter doctor", "flutter pub get", "flutter clean"],
         "correct": "C", "studentPick": "C"},
        {"prompt": "Dart's null-safety operator that asserts non-null is:",
         "options": ["?.", "??", "!", "?:"],
         "correct": "C", "studentPick": "B"},  # wrong on purpose
    ],
    "multi_select": [
        {"prompt": "Which of the following are valid Dart collection types? (select all)",
         "options": ["List", "Set", "Tuple", "Map"],
         "correct": ["A", "B", "D"], "studentPicks": ["A", "B", "D"]},  # all correct
        {"prompt": "Which are mobile platforms Flutter targets natively? (select all)",
         "options": ["iOS", "Android", "Windows Phone", "BlackBerry OS"],
         "correct": ["A", "B"], "studentPicks": ["A", "B", "C"]},  # one extra → partial
        {"prompt": "Which are immutable data types in Dart? (select all)",
         "options": ["int", "String", "List", "double"],
         "correct": ["A", "B", "D"], "studentPicks": ["A", "B"]},  # one missing → partial
    ],
    "fill_blanks": [
        {"prompt": "Flutter's package manifest is named ____ ; the command 'flutter ____' fetches deps.",
         "blanks": {"1": "pubspec.yaml", "2": "pub get"},
         "studentBlanks": {"1": "pubspec.yaml", "2": "pub get"}},
        {"prompt": "The keyword ____ declares an immutable variable; ____ declares one resolved at compile time.",
         "blanks": {"1": "final", "2": "const"},
         "studentBlanks": {"1": "final", "2": "const"}},
        {"prompt": "A Flutter widget tree is rebuilt by calling ____; the framework re-runs the ____ method.",
         "blanks": {"1": "setState()", "2": "build"},
         "studentBlanks": {"1": "setState", "2": "build"}},  # minor typo
    ],
    "matching": [
        {"prompt": "Match each concept (1–3) with its definition (A–C).",
         "leftItems": {"1": "StatelessWidget", "2": "setState()", "3": "Hot Reload"},
         "rightOptions": {"A": "Triggers a widget rebuild",
                          "B": "Injects updated source into the running app",
                          "C": "An immutable widget — never rebuilds itself"},
         "correct": {"1": "C", "2": "A", "3": "B"},
         "studentPicks": {"1": "C", "2": "A", "3": "B"}},
        {"prompt": "Match each Dart keyword (1–3) with its purpose (A–C).",
         "leftItems": {"1": "final", "2": "async", "3": "extends"},
         "rightOptions": {"A": "Marks a function returning a Future",
                          "B": "Class inheritance declaration",
                          "C": "A variable that can be set only once"},
         "correct": {"1": "C", "2": "A", "3": "B"},
         "studentPicks": {"1": "C", "2": "B", "3": "B"}},  # one wrong
        {"prompt": "Match each lifecycle method (1–3) with when it runs (A–C).",
         "leftItems": {"1": "initState", "2": "build", "3": "dispose"},
         "rightOptions": {"A": "Every time the widget rebuilds",
                          "B": "Once when widget is removed from the tree",
                          "C": "Once when widget is inserted into the tree"},
         "correct": {"1": "C", "2": "A", "3": "B"},
         "studentPicks": {"1": "C", "2": "A", "3": "B"}},
    ],
    "open_ended": [
        {"prompt": "Define what 'callback function' means in JavaScript (one sentence).",
         "expected": "A function passed as an argument to another function and invoked later.",
         "studentAnswer": "A function passed to another function as an argument."},
        {"prompt": "Briefly explain the difference between StatelessWidget and StatefulWidget in Flutter.",
         "expected": "StatelessWidget cannot change after build; StatefulWidget can update its internal state and rebuild.",
         "studentAnswer": "Stateless cannot change after build, Stateful can update its data and rebuild."},
        {"prompt": "What is virtual memory and why do operating systems use it?",
         "expected": "Virtual memory is a memory-management technique that gives processes the illusion of more memory than is physically available, by paging unused regions to disk.",
         "studentAnswer": "Virtual memory lets the OS pretend there is more RAM than actually exists by swapping data to disk."},
    ],
}

# ── Drawing primitives ───────────────────────────────────────────────

class ScaledDraw:
    """Proxy around ImageDraw that auto-multiplies every coordinate by
    RENDER_SCALE. The rest of the script writes coordinates in the
    map's native 756×1086 system (which keeps the map.json untouched
    when we change scale); the actual pixels land at SCALE× resolution.

    Only the geometry methods we use are forwarded — keep this list in
    sync if new draw calls are added.
    """
    def __init__(self, d, scale: int) -> None:
        self._d = d
        self._s = scale

    def _scale_xy(self, xy):
        if isinstance(xy, (list, tuple)):
            # Supports both (x, y) point and (x0, y0, x1, y1) bbox.
            return tuple(v * self._s for v in xy)
        return xy

    def text(self, xy, *a, **kw):  return self._d.text(self._scale_xy(xy), *a, **kw)
    def rectangle(self, xy, *a, **kw):  return self._d.rectangle(self._scale_xy(xy), *a, **kw)
    def ellipse(self, xy, *a, **kw):  return self._d.ellipse(self._scale_xy(xy), *a, **kw)
    def line(self, xy, *a, **kw):  return self._d.line(self._scale_xy(xy), *a, **kw)


def font(path, size: int) -> ImageFont.FreeTypeFont:
    """Font factory that pre-multiplies size by RENDER_SCALE so glyphs
    stay proportional to the scaled image."""
    return ImageFont.truetype(str(path), int(size * RENDER_SCALE))


def _wrap(text: str, n: int) -> List[str]:
    out, line = [], ""
    for w in text.split():
        if len(line) + len(w) + 1 <= n:
            line = (line + " " + w).strip()
        else:
            if line: out.append(line)
            line = w
    if line: out.append(line)
    return out


def draw_bullseye(d, cx, cy, r=14):
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill="black")
    d.ellipse((cx - r + 4, cy - r + 4, cx + r - 4, cy + r - 4), fill="white")
    d.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill="black")


def draw_anchors(d):
    for k, a in ANCHORS.items():
        draw_bullseye(d, int(a["center"]["x"]), int(a["center"]["y"]))


def draw_student_grid(d, region, number, label_font):
    """Render the student number as 10 large handwritten digits, one
    per column. The backend's `handwriting.read_student_number`
    runs a digit CNN on each per-column box (the `studentNumberBoxes`
    array we emit in the map), NOT bubble-fill detection. So we
    have to draw GLYPHS, not filled circles. Cursive font roughly
    matches what the CNN was trained on (handwritten digits).

    Archetype C falls through `number` having fewer than 10 digits —
    those columns are left blank intentionally to trigger the
    edit-student-number ✎ workflow."""
    x0, y0, w, h = region["x"], region["y"], region["w"], region["h"]
    cw = w / 10
    digit_font = font(HAND_FONT, 16) if 'font' in globals() else None
    if digit_font is None:
        from PIL import ImageFont as _IF
        digit_font = _IF.truetype(str(HAND_FONT), 16)
    for c in range(10):
        cx = x0 + c * cw + cw / 2
        digit_char = number[c] if c < len(number) else None
        if digit_char and digit_char.isdigit():
            d.text((cx - 6, y0 + 4), digit_char,
                   fill="black", font=digit_font)


def page_header(d, page_num, total_pages, font, sub):
    d.text((PAGE_W // 2 - 220, 28),
           "İSTANBUL KÜLTÜR ÜNİVERSİTESİ — Mock Combined Exam (All Types)",
           fill="black", font=font)
    d.text((PAGE_W // 2 - 60, 50),
           f"Page {page_num} / {total_pages}",
           fill="#666", font=sub)


def card(d, x, y, w, h, type_label, qNum, prompt, badge_color, p_font, q_font):
    d.rectangle((x, y, x + w, y + h), outline="#bdbdbd", width=1)
    d.rectangle((x + 6, y + 6, x + 78, y + 26), fill=badge_color[0], outline=badge_color[1])
    d.text((x + 11, y + 9), f"Q{qNum} · {type_label[:5]}", fill=badge_color[2], font=q_font)
    cy = y + 32
    for line in _wrap(prompt, max(20, int((w - 16) / 6.3)))[:3]:
        d.text((x + 8, cy), line, fill="#222", font=p_font)
        cy += 14
    return cy


BADGE = {
    "open_ended": ("#fff3e6", "#e08e3a", "#b85a00"),
    "fill_blanks": ("#e8f4ff", "#3a8de0", "#0040b8"),
    "matching": ("#fff3f3", "#e07a7a", "#b80000"),
    "multiple_choice": ("#eaffea", "#5cae5c", "#1a661a"),
    "multi_select": ("#f0eaff", "#7a5cae", "#3a1a66"),
}


def draw_bubble(d, x, y, w, h, filled):
    cx, cy = x + w / 2, y + h / 2
    r = min(w, h) / 2
    d.ellipse((cx - r, cy - r, cx + r, cy + r),
              outline="black",
              fill="black" if filled else "white",
              width=1)


def draw_handwriting(d, x, y, w, h, text, hand_font, seed=0):
    rng = random.Random(seed)
    lines = _wrap(text, max(1, int(w / 13)))
    line_h = max(20, int(h / max(1, len(lines))))
    for i, line in enumerate(lines):
        d.text((x + 8 + rng.randint(-2, 2),
                y + 4 + i * line_h + rng.randint(-1, 2)),
               line, fill="#1a1a1a", font=hand_font)


# ── Page builders + map question entries ─────────────────────────────

def build_pages_and_map():
    """Return (list of PIL.Image, list of map-page dicts).
    qNum is global across pages (1..15)."""
    p_font = font(PRINT_FONT, 11)
    q_font = font(PRINT_FONT, 11)
    title_font = font(PRINT_FONT, 14)
    sub_font = font(PRINT_FONT, 10)
    label_font = font(PRINT_FONT, 9)
    hand_font = font(HAND_FONT, 22)
    big_hand = font(HAND_FONT, 26)

    images: List[Image.Image] = []
    map_pages: List[Dict[str, Any]] = []
    qNum = 1

    # ── Page 1: 3 MC + 3 MS ─────────────────────────────────────────
    img1 = Image.new("RGB", (PAGE_W * RENDER_SCALE, PAGE_H * RENDER_SCALE), "white")
    d1 = ScaledDraw(ImageDraw.Draw(img1), RENDER_SCALE)
    page_header(d1, 1, 3, title_font, sub_font)
    draw_anchors(d1)
    d1.text((STUDENT_NUMBER_REGION["x"], STUDENT_NUMBER_REGION["y"] - 18),
            "STUDENT NUMBER", fill="black", font=sub_font)
    draw_student_grid(d1, STUDENT_NUMBER_REGION, MOCK_STUDENT_NUMBER, label_font)

    page1_questions: Dict[str, Any] = {}
    # 3 MC, then 3 MS — 6 cards in a 2×3 grid (left/right × 3 rows)
    grid_top = 220
    card_w, card_h = 363, 130
    col_x = [10, 383]
    pool_seq = (
        [("multiple_choice", q) for q in POOL["multiple_choice"]]
        + [("multi_select", q) for q in POOL["multi_select"]]
    )
    for idx, (qtype, qdata) in enumerate(pool_seq):
        col = idx % 2
        row = idx // 2
        x = col_x[col]
        y = grid_top + row * (card_h + 10)
        body_y = card(d1, x, y, card_w, card_h, qtype, str(qNum), qdata["prompt"],
                       BADGE[qtype], p_font, q_font)
        # 4 options stacked
        opts: Dict[str, Dict[str, float]] = {}
        opt_label_font = font(PRINT_FONT, 10)
        # Archetype override (or {} for DEFAULT)
        ov = derive_student_answer(qtype, qdata, qNum)
        sp = ov.get("studentPick", qdata.get("studentPick"))
        sps = ov.get("studentPicks", qdata.get("studentPicks", []))
        for i, letter in enumerate(("A", "B", "C", "D")):
            ox = x + 14
            oy = body_y + 6 + i * 16
            opts[letter] = {"x": float(ox), "y": float(oy), "w": 12, "h": 12}
            if qtype == "multiple_choice":
                filled = sp == letter
            else:
                filled = letter in sps
            draw_bubble(d1, ox, oy, 12, 12, filled)
            d1.text((ox + 18, oy - 1),
                    f"{letter}) {qdata['options'][i]}",
                    fill="#222", font=opt_label_font)
        # map entry
        if qtype == "multiple_choice":
            page1_questions[str(qNum)] = {
                "type": "multiple_choice",
                "boundingBox": {"x": float(x), "y": float(y), "w": float(card_w), "h": float(card_h)},
                "scoring": {"points": 10, "penaltyPerItem": 0, "itemCount": 4},
                "expectedAnswer": {"correctOption": qdata["correct"]},
                "options": opts,
            }
        else:
            page1_questions[str(qNum)] = {
                "type": "multi_select",
                "boundingBox": {"x": float(x), "y": float(y), "w": float(card_w), "h": float(card_h)},
                "scoring": {"points": 10, "penaltyPerItem": 0, "itemCount": 4},
                "expectedAnswer": {"correctOptions": qdata["correct"]},
                "options": opts,
            }
        qNum += 1
    images.append(img1)
    # Per-digit boxes for the student-number bubble grid. Without
    # these, pipeline.py:225 skips student-number reading entirely
    # and every PDF lands as `unknown_1`. Layout matches v6 sample's
    # 10 evenly-spaced columns, each ~28×34 px.
    sn_w = STUDENT_NUMBER_REGION["w"] / 10
    student_number_boxes = [
        {
            "x": float(STUDENT_NUMBER_REGION["x"] + i * sn_w + 2),
            "y": float(STUDENT_NUMBER_REGION["y"] + 6),
            "w": float(sn_w - 4),
            "h": float(STUDENT_NUMBER_REGION["h"] - 12),
        }
        for i in range(10)
    ]

    map_pages.append({
        "pageId": "page_1",
        "pageWidth": PAGE_W,
        "pageHeight": PAGE_H,
        "anchors": ANCHORS,
        "studentNumberRegion": STUDENT_NUMBER_REGION,
        "studentNumberBoxes": student_number_boxes,
        "questions": page1_questions,
    })

    # ── Page 2: 3 fill_blanks + 3 matching ─────────────────────────
    img2 = Image.new("RGB", (PAGE_W * RENDER_SCALE, PAGE_H * RENDER_SCALE), "white")
    d2 = ScaledDraw(ImageDraw.Draw(img2), RENDER_SCALE)
    page_header(d2, 2, 3, title_font, sub_font)
    draw_anchors(d2)

    page2_questions: Dict[str, Any] = {}
    # 3 fill_blanks at top
    fill_top = 100
    fill_card_h = 130
    for i, qdata in enumerate(POOL["fill_blanks"]):
        x = 10 + (i % 2) * 373
        y = fill_top + (i // 2) * (fill_card_h + 14)
        # last row of the 2×2 spans both columns visually but we keep it left
        if i == 2:
            x = 10
            y = fill_top + 1 * (fill_card_h + 14)
        body_y = card(d2, x, y, 363 if i != 2 else 736, fill_card_h, "fill_blanks",
                      str(qNum), qdata["prompt"], BADGE["fill_blanks"], p_font, q_font)
        # Archetype override
        ov = derive_student_answer("fill_blanks", qdata, qNum)
        student_blanks = ov.get("studentBlanks", qdata.get("studentBlanks", {}))
        # 2 answer boxes
        boxes: Dict[str, Any] = {}
        bx_start = x + 14
        for bi, key in enumerate(("1", "2")):
            bx = bx_start + bi * 175
            by = body_y + 28
            bw, bh = 165, 26
            boxes[key] = {"x": float(bx), "y": float(by), "w": float(bw), "h": float(bh)}
            d2.rectangle((bx, by, bx + bw, by + bh), outline="#888", width=1)
            d2.text((bx, by - 12), f"Blank #{key}", fill="#555", font=label_font)
            ans = student_blanks.get(key, "")
            if ans:
                draw_handwriting(d2, bx, by, bw, bh, ans, hand_font,
                                 seed=qNum * 11 + bi)
        # answerSection (loose enclosing rect)
        ax0 = min(b["x"] for b in boxes.values()) - 4
        ay0 = min(b["y"] for b in boxes.values()) - 4
        ax1 = max(b["x"] + b["w"] for b in boxes.values()) + 4
        ay1 = max(b["y"] + b["h"] for b in boxes.values()) + 4
        page2_questions[str(qNum)] = {
            "type": "fill_blanks",
            "boundingBox": {"x": float(x), "y": float(y),
                            "w": float(363 if i != 2 else 736), "h": float(fill_card_h)},
            "scoring": {"points": 10, "penaltyPerItem": 0, "itemCount": len(boxes)},
            "expectedAnswer": {"correctBlanks": qdata["blanks"]},
            "answerBoxes": boxes,
            "answerSection": {"x": float(ax0), "y": float(ay0),
                               "w": float(ax1 - ax0), "h": float(ay1 - ay0)},
            "fillBlanks": {k: {"x": float(0), "y": float(0), "w": 50, "h": 12}
                            for k in boxes},  # decorative
        }
        qNum += 1

    # 3 matching at bottom of page 2
    match_top = fill_top + 2 * (fill_card_h + 14) + 24
    match_h = 160
    for i, qdata in enumerate(POOL["matching"]):
        x = 10
        y = match_top + i * (match_h + 12)
        if y + match_h > PAGE_H - 60:
            break  # don't run past page bounds
        body_y = card(d2, x, y, 736, match_h, "matching", str(qNum),
                      qdata["prompt"], BADGE["matching"], p_font, q_font)
        item_font = font(PRINT_FONT, 11)
        for j, (k, v) in enumerate(qdata["leftItems"].items()):
            d2.text((x + 16, body_y + 8 + j * 16), f"{k}. {v}",
                    fill="#222", font=item_font)
        for j, (k, v) in enumerate(qdata["rightOptions"].items()):
            d2.text((x + 380, body_y + 8 + j * 16), f"{k}. {v}",
                    fill="#222", font=item_font)
        # Archetype override for matching
        ov_m = derive_student_answer("matching", qdata, qNum)
        student_picks_m = ov_m.get("studentPicks", qdata.get("studentPicks", {}))
        # 3 answer boxes for letters
        boxes_m: Dict[str, Any] = {}
        for ai, key in enumerate(("1", "2", "3")):
            bx = x + 60 + ai * 80
            by = body_y + 8 + 3 * 16 + 14
            bw, bh = 80, 40
            boxes_m[key] = {"x": float(bx), "y": float(by), "w": float(bw), "h": float(bh)}
            d2.rectangle((bx, by, bx + bw, by + bh), outline="#888", width=1)
            small = font(PRINT_FONT, 10)
            d2.text((bx + bw / 2 - 4, by - 14), key, fill="#444", font=small)
            ans = student_picks_m.get(key, "")
            if ans:
                d2.text((bx + bw / 2 - 8, by + 2), ans,
                        fill="#1a1a1a", font=big_hand)
        ax0 = min(b["x"] for b in boxes_m.values()) - 4
        ay0 = min(b["y"] for b in boxes_m.values()) - 4
        ax1 = max(b["x"] + b["w"] for b in boxes_m.values()) + 4
        ay1 = max(b["y"] + b["h"] for b in boxes_m.values()) + 4
        page2_questions[str(qNum)] = {
            "type": "matching",
            "boundingBox": {"x": float(x), "y": float(y), "w": 736.0, "h": float(match_h)},
            "scoring": {"points": 10, "penaltyPerItem": 0, "itemCount": 3},
            "expectedAnswer": {"correctMatches": qdata["correct"]},
            "answerBoxes": boxes_m,
            "answerSection": {"x": float(ax0), "y": float(ay0),
                               "w": float(ax1 - ax0), "h": float(ay1 - ay0)},
        }
        qNum += 1
    images.append(img2)
    map_pages.append({
        "pageId": "page_2",
        "pageWidth": PAGE_W,
        "pageHeight": PAGE_H,
        "anchors": ANCHORS,
        "questions": page2_questions,
    })

    # ── Page 3: 3 open_ended ───────────────────────────────────────
    img3 = Image.new("RGB", (PAGE_W * RENDER_SCALE, PAGE_H * RENDER_SCALE), "white")
    d3 = ScaledDraw(ImageDraw.Draw(img3), RENDER_SCALE)
    page_header(d3, 3, 3, title_font, sub_font)
    draw_anchors(d3)

    page3_questions: Dict[str, Any] = {}
    open_top = 100
    open_h = 280
    for i, qdata in enumerate(POOL["open_ended"]):
        x = 10
        y = open_top + i * (open_h + 14)
        body_y = card(d3, x, y, 736, open_h, "open_ended", str(qNum),
                      qdata["prompt"], BADGE["open_ended"], p_font, q_font)
        # solutionArea — large
        sa_x, sa_y = x + 10, body_y + 20
        sa_w, sa_h = 716, open_h - 60 - (body_y - y - 32)
        d3.rectangle((sa_x, sa_y, sa_x + sa_w, sa_y + sa_h), outline="#888", width=1)
        d3.text((sa_x + 4, sa_y - 9), f"SolutionArea{qNum}", fill="#888", font=label_font)
        # Archetype override for open-ended
        ov_o = derive_student_answer("open_ended", qdata, qNum)
        student_text = ov_o.get("studentAnswer", qdata.get("studentAnswer", ""))
        if student_text:
            draw_handwriting(d3, sa_x, sa_y, sa_w, sa_h,
                             student_text, hand_font, seed=qNum * 7)
        page3_questions[str(qNum)] = {
            "type": "open_ended",
            "boundingBox": {"x": float(x), "y": float(y), "w": 736.0, "h": float(open_h)},
            "scoring": {"points": 10, "penaltyPerItem": None},
            "expectedAnswer": {"text": qdata["expected"]},
            "solutionArea": {"x": float(sa_x), "y": float(sa_y),
                              "w": float(sa_w), "h": float(sa_h)},
        }
        qNum += 1
    images.append(img3)
    map_pages.append({
        "pageId": "page_3",
        "pageWidth": PAGE_W,
        "pageHeight": PAGE_H,
        "anchors": ANCHORS,
        "questions": page3_questions,
    })

    return images, map_pages


# ── .ikuexam (frontend exam definition) ──────────────────────────────

# Frontend short type code ↔ backend long form.
SHORT = {
    "open_ended": "open",
    "fill_blanks": "fill",
    "matching": "match",
    "multiple_choice": "mc",
    "multi_select": "ms",
}


def build_ikuexam(map_pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Translate the map.json questions into the .ikuexam wire shape that
    the QuestionEditor / Dashboard load."""
    questions = []
    qNum = 1
    # Walk the pool in the same order the layout uses so qNums align.
    pool_seq: List[tuple] = (
        [("multiple_choice", q) for q in POOL["multiple_choice"]]
        + [("multi_select", q) for q in POOL["multi_select"]]
        + [("fill_blanks", q) for q in POOL["fill_blanks"]]
        + [("matching", q) for q in POOL["matching"]]
        + [("open_ended", q) for q in POOL["open_ended"]]
    )
    for qtype, q in pool_seq:
        base = {
            "type": SHORT[qtype],
            "text": q["prompt"],
            "imagePreview": None,
            "options": [], "correctAnswer": 0, "correctAnswers": [],
            "correctText": "", "correctImagePreview": None, "spaceId": 0,
            "matchLeft": [], "matchRight": [], "matchCorrect": [],
            "fillText": "", "fillAnswers": [],
            "points": 10, "penaltyPerItem": 0,
        }
        if qtype == "multiple_choice":
            base["options"] = q["options"]
            base["correctAnswer"] = "ABCD".index(q["correct"])
        elif qtype == "multi_select":
            base["options"] = q["options"]
            base["correctAnswers"] = sorted("ABCD".index(c) for c in q["correct"])
        elif qtype == "fill_blanks":
            # Use a single fillText with __ markers + per-blank answers.
            base["fillText"] = q["prompt"]
            base["fillAnswers"] = [q["blanks"][k] for k in sorted(q["blanks"])]
        elif qtype == "matching":
            base["matchLeft"] = list(q["leftItems"].values())
            base["matchRight"] = list(q["rightOptions"].values())
            # correct[i] = 0-indexed position of right item that matches left[i]
            right_keys = list(q["rightOptions"].keys())
            base["matchCorrect"] = [right_keys.index(q["correct"][k])
                                     for k in q["leftItems"].keys()]
        elif qtype == "open_ended":
            base["correctText"] = q["expected"]
        questions.append(base)
        qNum += 1

    return {
        "version": 1,
        "exam": {
            "faculty": "Engineering",
            "department": "Computer Science",
            "course": "Mock Combined Exam — All Question Types",
            "courseCode": "MOCK-ALL",
            "examType": "Midterm",
            "date": "2026-05-02",
            "duration": "60",
            "instructions": (
                "Synthesized exam covering all 5 question types. "
                "Click Evaluate and drop mock_filled_student.pdf to run "
                "the full pipeline end-to-end."
            ),
        },
        "questions": questions,
        "updatedAt": "2026-05-02T00:00:00.000Z",
    }


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    global ARCHETYPE
    install = "--install" in sys.argv

    # --archetype A | B | C → per-student-archetype generation.
    # If omitted, behaviour is unchanged (DEFAULT keeps the hardcoded
    # POOL answers — used when scripted before archetypes existed).
    arch = "DEFAULT"
    for i, arg in enumerate(sys.argv):
        if arg == "--archetype" and i + 1 < len(sys.argv):
            arch = sys.argv[i + 1].upper()
            break
    ARCHETYPE = arch

    # Per-archetype student number (Top=clean, Mid=clean, Struggling=garbled)
    sn = ARCHETYPE_SN.get(ARCHETYPE, MOCK_STUDENT_NUMBER)
    # Mutate the module-level constant the renderer reads.
    globals()["MOCK_STUDENT_NUMBER"] = sn

    images, map_pages = build_pages_and_map()

    # Per-archetype output filename so we can ship 3 student PDFs.
    out_pdf = OUT_DIR / (
        f"mock_filled_student_{ARCHETYPE}.pdf"
        if ARCHETYPE != "DEFAULT" else "mock_filled_student.pdf"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out_pdf, "PDF", resolution=150.0,
        save_all=True, append_images=images[1:],
    )
    print(f"[ok] wrote {out_pdf.relative_to(REPO)}  ({out_pdf.stat().st_size // 1024} KB, {len(images)} pages)  archetype={ARCHETYPE}")

    map_data = {
        "examId": EXAM_ID,
        # `totalPages` tells pipeline.split_by_students how many PDF pages
        # belong to one student. Without it, the pipeline assumes 1 page
        # per student and a 3-page PDF gets sliced as 3 separate students.
        "totalPages": len(map_pages),
        "pages": map_pages,
    }
    OUT_MAP.write_text(json.dumps(map_data, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    print(f"[ok] wrote {OUT_MAP.relative_to(REPO)}")

    iku = build_ikuexam(map_pages)
    OUT_IKU.write_text(json.dumps(iku, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    print(f"[ok] wrote {OUT_IKU.relative_to(REPO)}")

    print()
    print(f"Total questions: {sum(len(p['questions']) for p in map_pages)}")
    by_type: Dict[str, int] = {}
    for p in map_pages:
        for q in p["questions"].values():
            by_type[q["type"]] = by_type.get(q["type"], 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"  {t:<16} × {n}")

    if install:
        if not USER_DATA.exists():
            print(f"[warn] {USER_DATA} does not exist — Electron app probably "
                  f"hasn't run yet. Run `npm run dev` once, then re-run with --install.")
            return
        target_iku = USER_DATA / f"{EXAM_ID}.ikuexam"
        target_map = USER_DATA / f"{EXAM_ID}.map.json"
        target_iku.write_text(OUT_IKU.read_text(encoding="utf-8"), encoding="utf-8")
        target_map.write_text(OUT_MAP.read_text(encoding="utf-8"), encoding="utf-8")
        # Pointedly DO NOT write a results.json — that's what gates the
        # "Results" button in the dashboard. With it absent, the user
        # sees only Edit / Evaluate / Duplicate / Delete on the card.
        results_path = USER_DATA / f"{EXAM_ID}.results.json"
        if results_path.exists():
            results_path.unlink()
            print(f"[ok] removed stale {results_path.name}")
        print(f"[ok] installed exam to {USER_DATA}")
        print(f"     → restart the Electron app or refresh the Dashboard to see it.")


if __name__ == "__main__":
    main()
