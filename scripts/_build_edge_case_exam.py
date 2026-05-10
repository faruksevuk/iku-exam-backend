"""Build a single-page exam (matching + MC + multi-select + matching + fill)
plus a 5-student solution PDF that exercises mild edge cases:

  Student 1 (2200000001) — normal/baseline, control
  Student 2 (2200000002) — page rotated 2° counterclockwise
  Student 3 (2200000003) — QR code scribbled over (forces sequential fallback)
  Student 4 (2200000004) — random pencil scribbles in margins / empty areas
  Student 5 (2200000005) — light bubble fill (borderline OMR) + off-center handwriting

All five students give correct answers, so any score variance comes purely from
the edge-case interference.

Outputs:
  C:/Users/faruk/Downloads/bil101-edge-cases-blank.pdf
  C:/Users/faruk/Downloads/bil101-edge-cases-students.pdf
  C:/Users/faruk/Downloads/bil101-edge-cases-expected.json
  %APPDATA%/iku-exam-generator/exams/bil101-edge-cases-2026.{ikuexam,map.json}
"""
import io
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF
import qrcode
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Outputs ──────────────────────────────────────────────────────
EXAM_ID = "bil101-edge-cases-2026"
DOWNLOADS = Path(r"C:/Users/faruk/Downloads")
BLANK_PDF = DOWNLOADS / f"{EXAM_ID}-blank.pdf"
STUDENTS_PDF = DOWNLOADS / f"{EXAM_ID}-students.pdf"
EXPECTED_JSON = DOWNLOADS / f"{EXAM_ID}-expected.json"

APPDATA = Path(os.environ["APPDATA"]) / "iku-exam-generator"
EXAMS_DIR = APPDATA / "exams"
EXAMS_DIR.mkdir(parents=True, exist_ok=True)

# ── Page geometry (matches existing convention) ─────────────────
PAGE_W, PAGE_H = 756, 1086        # logical units used by the app's blanks
MARGIN = 40
PRINT_DPI = 200                   # used when rasterising for PDF embedding
BULLSEYE_PX = 16

# ── Exam definition ──────────────────────────────────────────────
EXAM = {
    "id": EXAM_ID,
    "courseCode": "BIL101",
    "course": "Introduction to Programming",
    "examType": "Quiz",
    "date": "2026-05-15",
    "faculty": "Faculty of Engineering",
    "department": "Computer Engineering",
    "questions": [
        # Q1 — Matching (4 cells): match algorithm to its complexity
        # Items (left)       Letters (right)
        #  1) Binary Search   A) O(n)
        #  2) Bubble Sort     B) O(log n)
        #  3) Linear Search   C) O(n log n)
        #  4) Merge Sort      D) O(n^2)
        # Correct: 1→B, 2→D, 3→A, 4→C
        {
            "type": "matching",
            "prompt": "Match each algorithm with its average-case time complexity:",
            "points": 10,
            "items_left": ["Binary Search", "Bubble Sort", "Linear Search", "Merge Sort"],
            "items_right": ["A) O(n)", "B) O(log n)", "C) O(n log n)", "D) O(n²)"],
            "correct": {"1": "B", "2": "D", "3": "A", "4": "C"},
        },
        # Q2 — Multiple choice (1 of 4): correct B
        {
            "type": "multiple_choice",
            "prompt": "Which data structure follows the Last-In-First-Out (LIFO) principle?",
            "points": 10,
            "options": ["Queue", "Stack", "Linked List", "Binary Tree"],
            "correct": "B",
        },
        # Q3 — Multi-select (2 of 4): correct A and C
        {
            "type": "multi_select",
            "prompt": "Select all primitive data types in Python (mark all that apply):",
            "points": 10,
            "options": ["int", "list", "float", "dict"],
            "correct": ["A", "C"],
        },
        # Q4 — Matching (4 cells): match language to its primary paradigm
        #  1) Haskell    A) Object-Oriented
        #  2) C          B) Functional
        #  3) Java       C) Procedural
        #  4) Prolog     D) Logic
        # Correct: 1→B, 2→C, 3→A, 4→D
        {
            "type": "matching",
            "prompt": "Match each language with its primary paradigm:",
            "points": 10,
            "items_left": ["Haskell", "C", "Java", "Prolog"],
            "items_right": ["A) Object-Oriented", "B) Functional",
                            "C) Procedural", "D) Logic"],
            "correct": {"1": "B", "2": "C", "3": "A", "4": "D"},
        },
        # Q5 — Fill in the blank (1 word): correct "Python"
        {
            "type": "fill_blanks",
            "prompt": ("Name the high-level interpreted programming language whose "
                       "reference implementation is CPython:"),
            "points": 10,
            "correct": "Python",
        },
    ],
}

# Total = 50 points
TOTAL_POINTS = sum(q["points"] for q in EXAM["questions"])

# ── Layout coordinates (page-space) ─────────────────────────────
LAYOUT = {
    "anchors": {
        "TL": {"cx": 18, "cy": 18},
        "TR": {"cx": PAGE_W - 18, "cy": 18},
        "BL": {"cx": 18, "cy": PAGE_H - 18},
        "BR": {"cx": PAGE_W - 18, "cy": PAGE_H - 18},
    },
    "header": {"y": 50},
    "name_field": {"x": MARGIN, "y": 105, "w": 360, "h": 26},
    "student_number_boxes": [
        {"x": MARGIN + i * 32, "y": 155, "w": 28, "h": 36} for i in range(10)
    ],
    "qr": {"x": PAGE_W - 130, "y": 50, "size": 95, "label_y_offset": 100},
    # Question Y positions (rough; refined inside the renderer)
    "q1_text_y": 220,    # Matching #1 prompt
    "q1_items_y": 250,   # listing of items
    "q1_boxes_y": 350,   # answer boxes row
    "q2_text_y": 410,    # MC prompt
    "q2_options_y": 440, # vertical bubble list
    "q3_text_y": 580,    # MS prompt
    "q3_options_y": 610,
    "q4_text_y": 740,    # Matching #2 prompt
    "q4_items_y": 770,
    "q4_boxes_y": 870,
    "q5_text_y": 930,    # Fill prompt
    "q5_box_y": 990,
}

# Answer-box geometry (page-space)
MATCH_BOX = {"w": 50, "h": 30}
MC_BUBBLE = {"d": 13}                       # outer diameter; inner ring radius -2
FILL_BOX = {"w": 200, "h": 30}

# ── Helpers ──────────────────────────────────────────────────────
def get_font(size: int, bold: bool = False, handwriting: bool = False) -> ImageFont.FreeTypeFont:
    """Return a TTF font handle. Falls back gracefully when not on Windows."""
    candidates_normal = [
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    candidates_bold = [
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
    ]
    candidates_handwriting = [
        "C:/Windows/Fonts/segoesc.ttf",  # Segoe Script
        "C:/Windows/Fonts/inkfree.ttf",  # Ink Free
        "C:/Windows/Fonts/comic.ttf",    # Comic Sans (close-enough handwriting fallback)
        "C:/Windows/Fonts/segoepr.ttf",  # Segoe Print
    ]
    pool = candidates_handwriting if handwriting else (candidates_bold if bold else candidates_normal)
    for p in pool:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_bullseye(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int = 16) -> None:
    r = size / 2
    # Outer black
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill="black")
    # Middle white
    r2 = r * 0.6
    draw.ellipse((cx - r2, cy - r2, cx + r2, cy + r2), fill="white")
    # Inner black
    r3 = r * 0.3
    draw.ellipse((cx - r3, cy - r3, cx + r3, cy + r3), fill="black")


def render_qr(text: str, size_px: int) -> Image.Image:
    """Return a B/W PIL Image of `text` as a QR code at exactly `size_px` square."""
    qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_L,
                       box_size=4, border=1)
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    return img.resize((size_px, size_px), Image.NEAREST)


def draw_centered_text(draw: ImageDraw.ImageDraw, xy_rect, text: str,
                       font: ImageFont.FreeTypeFont, fill: str = "black") -> None:
    x, y, w, h = xy_rect
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x + (w - tw) / 2, y + (h - th) / 2 - 2), text, font=font, fill=fill)


# ─────────────────────────────────────────────────────────────────
# Build the blank page (PIL canvas at print resolution)
# ─────────────────────────────────────────────────────────────────
def render_blank_page(canvas_scale: float = PRINT_DPI / 72.0) -> Image.Image:
    """Render the blank exam page as a PIL Image.

    canvas_scale converts page-space units (assumed ~72 DPI logical) into pixel
    units at PRINT_DPI. We render larger so embedded text / lines look crisp
    after PDF rasterisation; the map.json keeps the page-space coordinates.
    """
    W = int(PAGE_W * canvas_scale)
    H = int(PAGE_H * canvas_scale)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    s = canvas_scale  # short alias
    f_title = get_font(int(18 * s), bold=True)
    f_sub = get_font(int(11 * s))
    f_label = get_font(int(10 * s))
    f_question = get_font(int(11 * s), bold=True)
    f_body = get_font(int(10 * s))
    f_small = get_font(int(9 * s))

    # 1. Corner anchors
    for a in LAYOUT["anchors"].values():
        draw_bullseye(d, int(a["cx"] * s), int(a["cy"] * s), size=int(BULLSEYE_PX * s))

    # 2. Header
    d.text((MARGIN * s, 40 * s),
           f"{EXAM['courseCode']} — {EXAM['course']}",
           font=f_title, fill="black")
    d.text((MARGIN * s, 70 * s),
           f"{EXAM['examType']}   •   {EXAM['date']}   •   Total: {TOTAL_POINTS} pts",
           font=f_sub, fill="black")

    # 3. Student name field
    nf = LAYOUT["name_field"]
    d.text((nf["x"] * s, (nf["y"] - 16) * s), "Name / Surname:", font=f_label, fill="black")
    d.rectangle((nf["x"] * s, nf["y"] * s,
                 (nf["x"] + nf["w"]) * s, (nf["y"] + nf["h"]) * s),
                outline="black", width=max(1, int(s)))

    # 4. Student number cells (10 digits)
    d.text((MARGIN * s, 138 * s), "Student Number:", font=f_label, fill="black")
    for box in LAYOUT["student_number_boxes"]:
        d.rectangle((box["x"] * s, box["y"] * s,
                     (box["x"] + box["w"]) * s, (box["y"] + box["h"]) * s),
                    outline="black", width=max(1, int(s)))

    # 5. QR code + label
    qr_text = f"P1_{EXAM_ID}"
    qr_img = render_qr(qr_text, int(LAYOUT["qr"]["size"] * s))
    qr_x = int(LAYOUT["qr"]["x"] * s)
    qr_y = int(LAYOUT["qr"]["y"] * s)
    img.paste(qr_img, (qr_x, qr_y))
    d.text((qr_x, qr_y + int(LAYOUT["qr"]["label_y_offset"] * s)), qr_text,
           font=f_small, fill="#555")

    # ── Q1 — Matching #1
    q = EXAM["questions"][0]
    d.text((MARGIN * s, LAYOUT["q1_text_y"] * s),
           f"Q1. (10 pts)  {q['prompt']}", font=f_question, fill="black")
    # Two columns of items
    left_x = (MARGIN + 10) * s
    right_x = (PAGE_W / 2 + 10) * s
    for i, (lt, rt) in enumerate(zip(q["items_left"], q["items_right"])):
        d.text((left_x, (LAYOUT["q1_items_y"] + i * 19) * s),
               f"{i + 1}) {lt}", font=f_body, fill="black")
        d.text((right_x, (LAYOUT["q1_items_y"] + i * 19) * s),
               rt, font=f_body, fill="black")
    # Answer-box row labels + boxes
    d.text((MARGIN * s, (LAYOUT["q1_boxes_y"] - 18) * s),
           "Answer:", font=f_label, fill="black")
    for i in range(4):
        bx = MARGIN + 70 + i * (MATCH_BOX["w"] + 28)
        by = LAYOUT["q1_boxes_y"]
        d.text((bx * s, (by - 14) * s), f"{i + 1}:", font=f_label, fill="black")
        d.rectangle((bx * s, by * s,
                     (bx + MATCH_BOX["w"]) * s, (by + MATCH_BOX["h"]) * s),
                    outline="black", width=max(1, int(s)))

    # ── Q2 — Multiple choice
    q = EXAM["questions"][1]
    d.text((MARGIN * s, LAYOUT["q2_text_y"] * s),
           f"Q2. (10 pts)  {q['prompt']}", font=f_question, fill="black")
    for i, opt in enumerate(q["options"]):
        cx = (MARGIN + 22) * s
        cy = (LAYOUT["q2_options_y"] + i * 26 + 7) * s
        r = (MC_BUBBLE["d"] / 2) * s
        d.ellipse((cx - r, cy - r, cx + r, cy + r),
                  outline="black", width=max(1, int(s)))
        letter = chr(ord("A") + i)
        d.text(((MARGIN + 45) * s, (LAYOUT["q2_options_y"] + i * 26) * s),
               f"{letter})  {opt}", font=f_body, fill="black")

    # ── Q3 — Multi-select
    q = EXAM["questions"][2]
    d.text((MARGIN * s, LAYOUT["q3_text_y"] * s),
           f"Q3. (10 pts)  {q['prompt']}", font=f_question, fill="black")
    for i, opt in enumerate(q["options"]):
        cx = (MARGIN + 22) * s
        cy = (LAYOUT["q3_options_y"] + i * 26 + 7) * s
        r = (MC_BUBBLE["d"] / 2) * s
        d.ellipse((cx - r, cy - r, cx + r, cy + r),
                  outline="black", width=max(1, int(s)))
        letter = chr(ord("A") + i)
        d.text(((MARGIN + 45) * s, (LAYOUT["q3_options_y"] + i * 26) * s),
               f"{letter})  {opt}", font=f_body, fill="black")

    # ── Q4 — Matching #2
    q = EXAM["questions"][3]
    d.text((MARGIN * s, LAYOUT["q4_text_y"] * s),
           f"Q4. (10 pts)  {q['prompt']}", font=f_question, fill="black")
    for i, (lt, rt) in enumerate(zip(q["items_left"], q["items_right"])):
        d.text((left_x, (LAYOUT["q4_items_y"] + i * 19) * s),
               f"{i + 1}) {lt}", font=f_body, fill="black")
        d.text((right_x, (LAYOUT["q4_items_y"] + i * 19) * s),
               rt, font=f_body, fill="black")
    d.text((MARGIN * s, (LAYOUT["q4_boxes_y"] - 18) * s),
           "Answer:", font=f_label, fill="black")
    for i in range(4):
        bx = MARGIN + 70 + i * (MATCH_BOX["w"] + 28)
        by = LAYOUT["q4_boxes_y"]
        d.text((bx * s, (by - 14) * s), f"{i + 1}:", font=f_label, fill="black")
        d.rectangle((bx * s, by * s,
                     (bx + MATCH_BOX["w"]) * s, (by + MATCH_BOX["h"]) * s),
                    outline="black", width=max(1, int(s)))

    # ── Q5 — Fill in the blank
    q = EXAM["questions"][4]
    d.text((MARGIN * s, LAYOUT["q5_text_y"] * s),
           f"Q5. (10 pts)  {q['prompt']}", font=f_question, fill="black")
    d.text((MARGIN * s, (LAYOUT["q5_box_y"] - 18) * s),
           "Answer:", font=f_label, fill="black")
    fbx = MARGIN + 60
    fby = LAYOUT["q5_box_y"]
    d.rectangle((fbx * s, fby * s,
                 (fbx + FILL_BOX["w"]) * s, (fby + FILL_BOX["h"]) * s),
                outline="black", width=max(1, int(s)))

    return img


# ─────────────────────────────────────────────────────────────────
# Build map.json + .ikuexam (consumed by the backend pipeline)
# ─────────────────────────────────────────────────────────────────
def build_map() -> dict:
    """Build the map.json describing question coordinates + expected answers."""

    # Page-space coordinates (no rendering scale here — backend will resize)
    questions = {}

    # Q1 — Matching #1
    q1 = {
        "type": "matching",
        "boundingBox": {"x": MARGIN, "y": LAYOUT["q1_boxes_y"] - 20,
                        "w": PAGE_W - 2 * MARGIN,
                        "h": MATCH_BOX["h"] + 30},
        "scoring": {"points": 10, "penaltyPerItem": 0, "itemCount": 4},
        "expectedAnswer": {"correctMatches": EXAM["questions"][0]["correct"]},
        "answerBoxes": {},
    }
    for i in range(4):
        bx = MARGIN + 70 + i * (MATCH_BOX["w"] + 28)
        q1["answerBoxes"][str(i + 1)] = {
            "x": float(bx), "y": float(LAYOUT["q1_boxes_y"]),
            "w": float(MATCH_BOX["w"]), "h": float(MATCH_BOX["h"]),
        }
    questions["1"] = q1

    # Q2 — Multiple choice
    q2 = {
        "type": "multiple_choice",
        "boundingBox": {"x": MARGIN, "y": LAYOUT["q2_options_y"] - 10,
                        "w": 400, "h": 4 * 26 + 10},
        "scoring": {"points": 10, "penaltyPerItem": 0, "itemCount": 4},
        "expectedAnswer": {"correctOption": EXAM["questions"][1]["correct"]},
        "options": {},
    }
    for i, _opt in enumerate(EXAM["questions"][1]["options"]):
        letter = chr(ord("A") + i)
        cx = MARGIN + 22
        cy = LAYOUT["q2_options_y"] + i * 26 + 7
        q2["options"][letter] = {
            "x": float(cx - MC_BUBBLE["d"] / 2),
            "y": float(cy - MC_BUBBLE["d"] / 2),
            "w": float(MC_BUBBLE["d"]),
            "h": float(MC_BUBBLE["d"]),
        }
    questions["2"] = q2

    # Q3 — Multi-select
    q3 = {
        "type": "multi_select",
        "boundingBox": {"x": MARGIN, "y": LAYOUT["q3_options_y"] - 10,
                        "w": 400, "h": 4 * 26 + 10},
        "scoring": {"points": 10, "penaltyPerItem": 0, "itemCount": 4},
        "expectedAnswer": {"correctOptions": EXAM["questions"][2]["correct"]},
        "options": {},
    }
    for i, _opt in enumerate(EXAM["questions"][2]["options"]):
        letter = chr(ord("A") + i)
        cx = MARGIN + 22
        cy = LAYOUT["q3_options_y"] + i * 26 + 7
        q3["options"][letter] = {
            "x": float(cx - MC_BUBBLE["d"] / 2),
            "y": float(cy - MC_BUBBLE["d"] / 2),
            "w": float(MC_BUBBLE["d"]),
            "h": float(MC_BUBBLE["d"]),
        }
    questions["3"] = q3

    # Q4 — Matching #2
    q4 = {
        "type": "matching",
        "boundingBox": {"x": MARGIN, "y": LAYOUT["q4_boxes_y"] - 20,
                        "w": PAGE_W - 2 * MARGIN,
                        "h": MATCH_BOX["h"] + 30},
        "scoring": {"points": 10, "penaltyPerItem": 0, "itemCount": 4},
        "expectedAnswer": {"correctMatches": EXAM["questions"][3]["correct"]},
        "answerBoxes": {},
    }
    for i in range(4):
        bx = MARGIN + 70 + i * (MATCH_BOX["w"] + 28)
        q4["answerBoxes"][str(i + 1)] = {
            "x": float(bx), "y": float(LAYOUT["q4_boxes_y"]),
            "w": float(MATCH_BOX["w"]), "h": float(MATCH_BOX["h"]),
        }
    questions["4"] = q4

    # Q5 — Fill blank
    fbx = MARGIN + 60
    q5 = {
        "type": "fill_blanks",
        "boundingBox": {"x": fbx - 5, "y": LAYOUT["q5_box_y"] - 5,
                        "w": FILL_BOX["w"] + 10, "h": FILL_BOX["h"] + 10},
        "scoring": {"points": 10, "penaltyPerItem": 0, "itemCount": 1},
        "expectedAnswer": {"correctBlanks": {"1": EXAM["questions"][4]["correct"]}},
        "answerBoxes": {
            "1": {"x": float(fbx), "y": float(LAYOUT["q5_box_y"]),
                  "w": float(FILL_BOX["w"]), "h": float(FILL_BOX["h"])}
        },
    }
    questions["5"] = q5

    # Anchors
    anchors = {
        edge: {"type": "bullseye",
               "center": {"x": a["cx"], "y": a["cy"]},
               "diameter": BULLSEYE_PX}
        for edge, a in LAYOUT["anchors"].items()
    }

    # Student-number boxes
    sn_boxes = [
        {"x": float(b["x"]), "y": float(b["y"]),
         "w": float(b["w"]), "h": float(b["h"])}
        for b in LAYOUT["student_number_boxes"]
    ]

    page = {
        "pageId": "p1",
        "pageIndex": 0,
        "pageWidth": PAGE_W,
        "pageHeight": PAGE_H,
        "anchors": anchors,
        "studentNameField": {"x": LAYOUT["name_field"]["x"],
                             "y": LAYOUT["name_field"]["y"],
                             "w": LAYOUT["name_field"]["w"],
                             "h": LAYOUT["name_field"]["h"]},
        "studentNumberRegion": {
            "x": LAYOUT["student_number_boxes"][0]["x"],
            "y": LAYOUT["student_number_boxes"][0]["y"],
            "w": (LAYOUT["student_number_boxes"][-1]["x"]
                  + LAYOUT["student_number_boxes"][-1]["w"]
                  - LAYOUT["student_number_boxes"][0]["x"]),
            "h": LAYOUT["student_number_boxes"][0]["h"],
        },
        "studentNumberBoxes": sn_boxes,
        "questions": questions,
    }

    return {
        "version": 2,
        "examId": EXAM_ID,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "courseCode": EXAM["courseCode"],
        "examType": EXAM["examType"],
        "date": EXAM["date"],
        "totalPages": 1,
        "pages": [page],
    }


def build_ikuexam() -> dict:
    """Build the .ikuexam metadata file (what the app's store carries)."""
    return {
        "id": EXAM_ID,
        "courseCode": EXAM["courseCode"],
        "course": EXAM["course"],
        "examType": EXAM["examType"],
        "date": EXAM["date"],
        "faculty": EXAM["faculty"],
        "department": EXAM["department"],
        "questionCount": len(EXAM["questions"]),
        "totalPoints": TOTAL_POINTS,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        # The full questions array isn't strictly required for evaluation, but
        # keeps the exam openable inside the editor.
        "questions": EXAM["questions"],
    }


# ─────────────────────────────────────────────────────────────────
# Student answer rendering — draws on a fresh copy of the blank
# ─────────────────────────────────────────────────────────────────
def fill_bubble(d: ImageDraw.ImageDraw, cx: float, cy: float, d_px: float,
                darkness: float = 0.95) -> None:
    """Fill an answer bubble with a circle of the given darkness (0..1)."""
    fill_int = int(255 * (1.0 - darkness))
    color = (fill_int, fill_int, fill_int)
    r = d_px / 2 - max(1, d_px * 0.08)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)


def jitter(v: float, amt: float = 1.5) -> float:
    return v + random.uniform(-amt, amt)


def draw_student_letter(d: ImageDraw.ImageDraw, box, letter: str,
                        font: ImageFont.FreeTypeFont, scale: float,
                        offset: tuple[float, float] = (0, 0)) -> None:
    """Draw a single letter into a matching answer box (centred-ish, slight skew)."""
    bx, by = box["x"] * scale, box["y"] * scale
    bw, bh = box["w"] * scale, box["h"] * scale
    bbox = d.textbbox((0, 0), letter, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = bx + (bw - tw) / 2 + offset[0] * scale
    y = by + (bh - th) / 2 - 2 + offset[1] * scale
    d.text((jitter(x, 1.0), jitter(y, 1.0)), letter, font=font, fill="black")


def draw_student_word(d: ImageDraw.ImageDraw, box, word: str,
                      font: ImageFont.FreeTypeFont, scale: float,
                      offset: tuple[float, float] = (0, 0)) -> None:
    bx, by = box["x"] * scale, box["y"] * scale
    bw, bh = box["w"] * scale, box["h"] * scale
    bbox = d.textbbox((0, 0), word, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = bx + 10 * scale + offset[0] * scale
    y = by + (bh - th) / 2 - 2 + offset[1] * scale
    d.text((x, y), word, font=font, fill="black")


def render_student_page(student_idx: int, student_number: str,
                        edge_case: str, blank_template: Image.Image,
                        scale: float = PRINT_DPI / 72.0) -> Image.Image:
    """Return a PIL image of one filled student's page (no rotation applied here)."""
    page = blank_template.copy()
    d = ImageDraw.Draw(page)

    f_digit = get_font(int(20 * scale), handwriting=True)
    f_letter = get_font(int(18 * scale), handwriting=True)
    f_word = get_font(int(16 * scale), handwriting=True)
    f_scribble = get_font(int(12 * scale), handwriting=True)

    # ── Student number — write each digit in its own box
    for i, digit in enumerate(student_number):
        box = LAYOUT["student_number_boxes"][i]
        bbox = d.textbbox((0, 0), digit, font=f_digit)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = box["x"] * scale + (box["w"] * scale - tw) / 2
        y = box["y"] * scale + (box["h"] * scale - th) / 2 - 2
        d.text((jitter(x, 1.0), jitter(y, 1.0)), digit, font=f_digit, fill="black")

    # ── Q1 — Matching #1
    correct = EXAM["questions"][0]["correct"]
    for slot, letter in correct.items():
        bx = MARGIN + 70 + (int(slot) - 1) * (MATCH_BOX["w"] + 28)
        by = LAYOUT["q1_boxes_y"]
        box = {"x": bx, "y": by, "w": MATCH_BOX["w"], "h": MATCH_BOX["h"]}
        draw_student_letter(d, box, letter, f_letter, scale)

    # ── Q2 — Multiple choice (correct = B by default)
    correct = EXAM["questions"][1]["correct"]
    letter_idx = ord(correct) - ord("A")
    cx = (MARGIN + 22) * scale
    cy = (LAYOUT["q2_options_y"] + letter_idx * 26 + 7) * scale
    fill_bubble(d, cx, cy, MC_BUBBLE["d"] * scale,
                darkness=(0.30 if edge_case == "light_bubble" else 0.92))

    # ── Q3 — Multi-select (correct = A and C)
    for letter in EXAM["questions"][2]["correct"]:
        idx = ord(letter) - ord("A")
        cx = (MARGIN + 22) * scale
        cy = (LAYOUT["q3_options_y"] + idx * 26 + 7) * scale
        fill_bubble(d, cx, cy, MC_BUBBLE["d"] * scale, darkness=0.92)

    # ── Q4 — Matching #2
    correct = EXAM["questions"][3]["correct"]
    for slot, letter in correct.items():
        bx = MARGIN + 70 + (int(slot) - 1) * (MATCH_BOX["w"] + 28)
        by = LAYOUT["q4_boxes_y"]
        box = {"x": bx, "y": by, "w": MATCH_BOX["w"], "h": MATCH_BOX["h"]}
        draw_student_letter(d, box, letter, f_letter, scale)

    # ── Q5 — Fill blank (correct = "Python")
    fbx = MARGIN + 60
    fby = LAYOUT["q5_box_y"]
    fbox = {"x": fbx, "y": fby, "w": FILL_BOX["w"], "h": FILL_BOX["h"]}
    word = EXAM["questions"][4]["correct"]
    off_x, off_y = (0, 0)
    if edge_case == "off_center_handwriting":
        off_x, off_y = (50, -4)   # nudge right + slightly up; grazes the box edge
    draw_student_word(d, fbox, word, f_word, scale, offset=(off_x, off_y))

    # ── Edge-case-specific overlays ─────────────────────────────
    if edge_case == "scribbled_qr":
        # Draw thick pencil-like scribbles across the QR area
        qr_x = LAYOUT["qr"]["x"] * scale
        qr_y = LAYOUT["qr"]["y"] * scale
        qr_s = LAYOUT["qr"]["size"] * scale
        for _ in range(70):
            x1 = random.uniform(qr_x, qr_x + qr_s)
            y1 = random.uniform(qr_y, qr_y + qr_s)
            x2 = x1 + random.uniform(-qr_s * 0.35, qr_s * 0.35)
            y2 = y1 + random.uniform(-qr_s * 0.35, qr_s * 0.35)
            d.line((x1, y1, x2, y2), fill="#1c1c1c", width=int(2.5 * scale))

    if edge_case == "marginal_scribbles":
        # Random pencil notes in empty areas (right margin between Q2 and Q3, etc.)
        scribble_zones = [
            # (x, y, w, h) in page space — empty stretches
            (420, 450, 280, 100),    # right of Q2
            (420, 620, 280, 100),    # right of Q3
            (420, 940, 280, 80),     # right of Q5
            (40, 1010, 700, 35),     # bottom margin
        ]
        notes = [
            "?? maybe queue",
            "stack uses LIFO",
            "skip Q4",
            "review later",
            "* check answer",
            "Q3 → A & C",
            "study chapter 4",
        ]
        for zx, zy, zw, zh in scribble_zones[:3]:
            note = random.choice(notes)
            tx = (zx + random.uniform(0, zw * 0.2)) * scale
            ty = (zy + random.uniform(0, zh * 0.3)) * scale
            d.text((tx, ty), note, font=f_scribble, fill="#2a2a2a")
        # Bottom-margin doodle: a wavy underline
        bz = scribble_zones[3]
        prev_x = bz[0] * scale
        prev_y = (bz[1] + bz[3] / 2) * scale
        for k in range(40):
            nx = prev_x + 18 * scale
            ny = (bz[1] + bz[3] / 2 + (k % 2) * 6) * scale
            d.line((prev_x, prev_y, nx, ny), fill="#2a2a2a", width=int(1.5 * scale))
            prev_x, prev_y = nx, ny

    return page


def apply_rotation(img: Image.Image, degrees: float) -> Image.Image:
    """Apply a small rotation, preserving canvas size, with white fill."""
    return img.rotate(degrees, resample=Image.BILINEAR,
                      expand=False, fillcolor="white")


# ─────────────────────────────────────────────────────────────────
# Build everything
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    random.seed(2026)

    # 1) Render the blank page (used for the standalone blank PDF + as the
    #    template every student page draws on top of).
    blank_img = render_blank_page()

    # 2) Save the standalone blank PDF.
    blank_img.save(str(BLANK_PDF), "PDF", resolution=PRINT_DPI)
    print(f"Blank PDF        → {BLANK_PDF}  ({BLANK_PDF.stat().st_size / 1024:.1f} KB)")

    # 3) Save the map.json + .ikuexam into the app's store.
    exam_map = build_map()
    ikuexam = build_ikuexam()

    map_path = EXAMS_DIR / f"{EXAM_ID}.map.json"
    iku_path = EXAMS_DIR / f"{EXAM_ID}.ikuexam"
    map_path.write_text(json.dumps(exam_map, indent=2, ensure_ascii=False), encoding="utf-8")
    iku_path.write_text(json.dumps(ikuexam, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"map.json         → {map_path}")
    print(f".ikuexam         → {iku_path}")

    # 4) Render 5 student pages, one per edge case.
    plan = [
        ("2200000001", "normal"),
        ("2200000002", "rotated_2deg"),
        ("2200000003", "scribbled_qr"),
        ("2200000004", "marginal_scribbles"),
        ("2200000005", "light_bubble_and_offcenter"),
    ]

    pages: list[Image.Image] = []
    for idx, (sn, case) in enumerate(plan):
        # For case "light_bubble_and_offcenter" we apply both sub-cases
        if case == "light_bubble_and_offcenter":
            page = render_student_page(idx, sn, "light_bubble", blank_img)
            # then overlay off-center handwriting on top of the existing page
            d = ImageDraw.Draw(page)
            scale = PRINT_DPI / 72.0
            f_word = get_font(int(16 * scale), handwriting=True)
            fbx = MARGIN + 60
            fby = LAYOUT["q5_box_y"]
            fbox = {"x": fbx, "y": fby, "w": FILL_BOX["w"], "h": FILL_BOX["h"]}
            # Erase the existing word (cover with white) then redraw with nudge
            # (cleaner than over-stamping)
            d.rectangle(
                (fbox["x"] * scale + 1, fbox["y"] * scale + 1,
                 (fbox["x"] + fbox["w"]) * scale - 1, (fbox["y"] + fbox["h"]) * scale - 1),
                fill="white",
            )
            draw_student_word(d, fbox, EXAM["questions"][4]["correct"],
                              f_word, scale, offset=(60, -3))
        else:
            page = render_student_page(idx, sn, case, blank_img)

        # Rotation post-step
        if case == "rotated_2deg":
            page = apply_rotation(page, +2.0)   # positive = counter-clockwise in PIL

        pages.append(page)
        print(f"  Student {idx + 1} ({sn}) — {case}")

    # 5) Stack all 5 pages into a single PDF.
    pages[0].save(
        str(STUDENTS_PDF),
        "PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=PRINT_DPI,
    )
    print(f"Students PDF     → {STUDENTS_PDF}  ({STUDENTS_PDF.stat().st_size / 1024:.1f} KB)")

    # 6) Expected results — ground truth for every student.
    expected: dict = {
        "examId": EXAM_ID,
        "totalStudents": 5,
        "totalQuestions": len(EXAM["questions"]),
        "students": [],
    }
    for sn, case in plan:
        student_record = {
            "studentNumber": sn,
            "edgeCase": case,
            "totalScore": TOTAL_POINTS,
            "totalMaxPoints": TOTAL_POINTS,
            "questions": {
                "1": {"type": "matching",
                      "expectedMatches": EXAM["questions"][0]["correct"],
                      "expectedScore": 10, "shouldBeCorrect": True},
                "2": {"type": "multiple_choice",
                      "expectedOption": EXAM["questions"][1]["correct"],
                      "expectedScore": 10, "shouldBeCorrect": True},
                "3": {"type": "multi_select",
                      "expectedOptions": EXAM["questions"][2]["correct"],
                      "expectedScore": 10, "shouldBeCorrect": True},
                "4": {"type": "matching",
                      "expectedMatches": EXAM["questions"][3]["correct"],
                      "expectedScore": 10, "shouldBeCorrect": True},
                "5": {"type": "fill_blanks",
                      "expectedBlanks": {"1": EXAM["questions"][4]["correct"]},
                      "expectedScore": 10, "shouldBeCorrect": True},
            },
        }
        # Per-edge-case notes the comparator can use to interpret deviations.
        notes = {
            "normal": "Baseline / control. Any deviation here is a pipeline bug, "
                      "not an edge-case effect.",
            "rotated_2deg": "Page rotated +2° (CCW). Anchors should still resolve via "
                            "homography; expect alignment to absorb the tilt and "
                            "all 5 questions to score normally.",
            "scribbled_qr": "QR code obliterated by scribbles. QR detection will fail; "
                            "splitter must fall back to sequential page grouping. All "
                            "answers are otherwise normal.",
            "marginal_scribbles": "Random pencil scribbles in margins and empty zones. "
                                  "Should not affect question crops, since none of the "
                                  "scribbles enter the answer-box regions.",
            "light_bubble_and_offcenter": "Q2 bubble filled at ~30% darkness (above the "
                                          "10% empty threshold, well below 50%; OMR may "
                                          "flag as ambiguous). Q5 handwriting nudged "
                                          "60px right and 3px up — text starts close to "
                                          "the box's right edge but stays inside.",
        }
        student_record["notes"] = notes[case]
        expected["students"].append(student_record)

    EXPECTED_JSON.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Expected results → {EXPECTED_JSON}")


if __name__ == "__main__":
    main()
