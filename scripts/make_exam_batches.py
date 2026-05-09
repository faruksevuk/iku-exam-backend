"""make_exam_batches.py — generate 3 exams, each with 3 PDFs of 10
filled student papers, for sequential evaluation testing.

Output layout under C:/Users/faruk/Downloads/exam-batches/:

  <slug>/
    batch-1.pdf    10 student papers
    batch-2.pdf    10 student papers
    batch-3.pdf    10 student papers

Each blank exam PDF + its map.json + .ikuexam are also installed into
the app's userData/exams folder so the dashboard can pick them up.

Layout per exam:
  Page 1 — title, student-number row, Q1 (MC), Q2 (MC), Q3 (matching),
           Q4 (fill in the blank)
  Page 2 — Q5 (open ended, ruled answer area)

All student answers are CORRECT (matches the answer key) so the
pipeline gets clean ground-truth data for every reader path.
"""
from __future__ import annotations

import json
import os
import random
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont
import qrcode
import fitz  # PyMuPDF

# ── Paths ────────────────────────────────────────────────────────────
DOWNLOADS = Path(r"C:/Users/faruk/Downloads/exam-batches")
USER_DATA = Path(os.environ["APPDATA"]) / "iku-exam-generator" / "exams"
DOWNLOADS.mkdir(parents=True, exist_ok=True)
USER_DATA.mkdir(parents=True, exist_ok=True)

# ── Page geometry (canonical = renderer's .page CSS px) ─────────────
CANVAS_W = 756
CANVAS_H = 1086
SCALE = 2  # 2× for crisp anti-aliased anchors + bubble strokes
W = CANVAS_W * SCALE
H = CANVAS_H * SCALE
DPI = 96 * SCALE


# ── Fonts ────────────────────────────────────────────────────────────
FONT_DIR = r"C:/Windows/Fonts"
F_TITLE = ImageFont.truetype(os.path.join(FONT_DIR, "arialbd.ttf"), 16 * SCALE)
F_SUB = ImageFont.truetype(os.path.join(FONT_DIR, "arial.ttf"), 11 * SCALE)
F_TEXT = ImageFont.truetype(os.path.join(FONT_DIR, "arial.ttf"), 11 * SCALE)
F_LABEL = ImageFont.truetype(os.path.join(FONT_DIR, "arialbd.ttf"), 10 * SCALE)
F_LABEL_SM = ImageFont.truetype(os.path.join(FONT_DIR, "arialbd.ttf"), 9 * SCALE)
F_HAND = ImageFont.truetype(os.path.join(FONT_DIR, "segoepr.ttf"), 14 * SCALE)
F_HAND_SM = ImageFont.truetype(os.path.join(FONT_DIR, "segoepr.ttf"), 13 * SCALE)
F_HAND_FILL = ImageFont.truetype(os.path.join(FONT_DIR, "segoepr.ttf"), 16 * SCALE)
F_HAND_BIG = ImageFont.truetype(os.path.join(FONT_DIR, "segoeprb.ttf"), 22 * SCALE)
F_DIGIT = ImageFont.truetype(os.path.join(FONT_DIR, "comic.ttf"), 22 * SCALE)


def s(v: float) -> int:
    """canonical px → render px (the 2× scaled canvas)."""
    return int(round(v * SCALE))


# ── Anchor + QR helpers ──────────────────────────────────────────────
def draw_bullseye(d: ImageDraw.ImageDraw, cx: float, cy: float) -> None:
    cx_, cy_ = s(cx), s(cy)
    outer = 8 * SCALE
    inner = 3 * SCALE
    d.ellipse((cx_ - outer, cy_ - outer, cx_ + outer, cy_ + outer), fill="black")
    d.ellipse((cx_ - inner, cy_ - inner, cx_ + inner, cy_ + inner), fill="white")


def make_qr(text: str, size_canonical: int = 60) -> Image.Image:
    qr = qrcode.QRCode(border=1, box_size=3)
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    return img.resize((s(size_canonical), s(size_canonical)), Image.NEAREST)


# ── Layout constants (identical layout across all 3 exams; only text
# content + correct-answer keys differ) ──────────────────────────────
ANCHORS_CANONICAL = {
    "TL": (11, 11),
    "TR": (745, 11),
    "BL": (11, 1075),
    "BR": (745, 1075),
}

SN_BOX_W = 32
SN_BOX_H = 38
SN_GAP = 4
SN_X0 = 250
SN_Y0 = 130

NAME_X0 = 30
NAME_Y0 = 130
NAME_W = 200
NAME_H = 40

Q1_TOP = 240
Q1_LEFT = 25
Q1_W = 350
Q1_H = 130

Q2_TOP = 240
Q2_LEFT = 380
Q2_W = 350
Q2_H = 130

Q3_TOP = 380
Q3_LEFT = 25
Q3_W = 705
Q3_H = 220

Q4_TOP = 610
Q4_LEFT = 25
Q4_W = 705
Q4_H = 130

# Page 2 — open ended
Q5_TOP = 80
Q5_LEFT = 25
Q5_W = 705
Q5_H = 220


# ── Draw a question card frame ───────────────────────────────────────
def draw_card(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int,
              label: str, points: int) -> None:
    d.rectangle((s(x), s(y), s(x + w), s(y + h)), outline="#aaa", width=2)
    d.text((s(x + 8), s(y + 6)), label, font=F_LABEL, fill="black")
    d.text((s(x + w - 50), s(y + 6)), f"{points} pts", font=F_LABEL_SM, fill="#444")


# ── Per-exam configs ─────────────────────────────────────────────────
EXAMS: List[Dict[str, Any]] = [
    {
        "slug": "4012-se-midterm-2026",
        "courseCode": "4012",
        "courseName": "software engineering",
        "examType": "midterm",
        "date": "2026-05-20",
        "title": "[4012] Software Engineering — Midterm Exam",
        "duration": "90 min",
        "totalPts": 50,
        "q1": {
            "stem": "Which phase immediately follows requirements analysis in the SDLC?",
            "options": {"A": "testing", "B": "design", "C": "implementation", "D": "maintenance"},
            "correct": "B",
        },
        "q2": {
            "stem": "Which is a traditional, plan-driven (non-Agile) methodology?",
            "options": {"A": "Scrum", "B": "Extreme Programming", "C": "Waterfall", "D": "Kanban"},
            "correct": "C",
        },
        "q3": {
            "stem": "Match each concept on the left with its definition on the right.",
            "left": ["SDLC", "Clean Code", "Unit Test", "Agile"],
            "right": [
                "Structural process: planning, analysis, design, implementation, testing.",
                "Methodology: rapid adaptation to changing requirements; short cycles.",
                "Principles: small, readable functions/classes; single responsibility.",
                "Tests verifying the smallest testable unit, usually one function.",
            ],
            "correct": {"1": "A", "2": "C", "3": "D", "4": "B"},
        },
        "q4": {
            "stem": "Methodology aimed at rapid adaptation in short cycles is called ____",
            "correct": {"1": "Agile"},
        },
        "q5": {
            "stem": "What does the acronym SDLC stand for in software engineering?",
            "correct": "Software Development Life Cycle. It is the structural process that guides the entire software development from planning through testing and maintenance.",
        },
    },
    {
        "slug": "mat102-calc2-midterm-2026",
        "courseCode": "MAT102",
        "courseName": "calculus II",
        "examType": "midterm",
        "date": "2026-05-22",
        "title": "[MAT102] Calculus II — Midterm Exam",
        "duration": "90 min",
        "totalPts": 50,
        "q1": {
            "stem": "Which integral represents the area under f(x) from x=a to x=b?",
            "options": {
                "A": "f'(b) - f'(a)", "B": "integral of f(x) dx from a to b",
                "C": "f(b) + f(a)", "D": "lim f(x) as x to b",
            },
            "correct": "B",
        },
        "q2": {
            "stem": "The derivative of sin(x) with respect to x equals?",
            "options": {"A": "-cos(x)", "B": "-sin(x)", "C": "cos(x)", "D": "tan(x)"},
            "correct": "C",
        },
        "q3": {
            "stem": "Match each function with its derivative.",
            "left": ["x^2", "ln(x)", "e^x", "cos(x)"],
            "right": ["e^x", "2x", "-sin(x)", "1/x"],
            "correct": {"1": "B", "2": "D", "3": "A", "4": "C"},
        },
        "q4": {
            "stem": "The fundamental theorem links differentiation and ____",
            "correct": {"1": "integration"},
        },
        "q5": {
            "stem": "State the chain rule for the derivative of a composite function f(g(x)).",
            "correct": "The derivative equals f prime of g of x times g prime of x. The chain rule lets us differentiate composite functions by multiplying outer and inner derivatives.",
        },
    },
    {
        "slug": "phy201-physics-quiz-2026",
        "courseCode": "PHY201",
        "courseName": "general physics",
        "examType": "quiz",
        "date": "2026-05-24",
        "title": "[PHY201] General Physics — Quiz",
        "duration": "60 min",
        "totalPts": 50,
        "q1": {
            "stem": "Which is Newton's second law of motion?",
            "options": {"A": "F = m * v", "B": "F = m * a", "C": "F = m / a", "D": "F = a / m"},
            "correct": "B",
        },
        "q2": {
            "stem": "The SI unit of energy is the?",
            "options": {"A": "Newton", "B": "Watt", "C": "Joule", "D": "Pascal"},
            "correct": "C",
        },
        "q3": {
            "stem": "Match each quantity with its SI unit.",
            "left": ["Force", "Power", "Pressure", "Charge"],
            "right": ["Watt", "Pascal", "Coulomb", "Newton"],
            "correct": {"1": "D", "2": "A", "3": "B", "4": "C"},
        },
        "q4": {
            "stem": "Acceleration due to gravity near Earth's surface is approximately ____ m/s^2",
            "correct": {"1": "9.81"},
        },
        "q5": {
            "stem": "State the law of conservation of energy in one sentence.",
            "correct": "Energy cannot be created or destroyed, only transformed from one form to another. The total energy of an isolated system stays constant over time.",
        },
    },
]


# ── Student names + numbers (10 per archetype × 3 = 30 per exam) ────
TURKISH_NAMES = [
    "Ayse Yilmaz", "Mehmet Demir", "Fatma Kaya", "Ali Celik", "Zeynep Sahin",
    "Mustafa Ozdemir", "Elif Arslan", "Ahmet Yildiz", "Esra Polat", "Burak Aydin",
    "Selin Ozturk", "Emre Korkmaz", "Cansu Tas", "Hasan Erdogan", "Deniz Gunes",
    "Merve Akar", "Onur Kilic", "Sevda Karaca", "Tolga Bal", "Sibel Aksoy",
    "Kerem Avci", "Ipek Ergin", "Baris Kuzu", "Gulay Soylu", "Yusuf Ata",
    "Damla Kose", "Hakan Erol", "Buse Cetin", "Cem Bulut", "Naz Yagmur",
]


def build_blank(cfg: Dict[str, Any]) -> Tuple[Image.Image, Image.Image, Dict[str, Any]]:
    """Return (page1_img, page2_img, map_dict) for the given exam config."""
    p1 = Image.new("RGB", (W, H), "white")
    p2 = Image.new("RGB", (W, H), "white")
    d1 = ImageDraw.Draw(p1)
    d2 = ImageDraw.Draw(p2)

    # Anchors on both pages.
    for cx, cy in ANCHORS_CANONICAL.values():
        draw_bullseye(d1, cx, cy)
        draw_bullseye(d2, cx, cy)

    # QR + title on page 1.
    qr1 = make_qr(f"P1_{cfg['slug']}", size_canonical=60)
    p1.paste(qr1, (s(675), s(20)))
    qr2 = make_qr(f"P2_{cfg['slug']}", size_canonical=60)
    p2.paste(qr2, (s(675), s(20)))

    # University header
    d1.text((s(CANVAS_W // 2 - 130), s(45)), "T.C. Istanbul Kultur Universitesi",
            font=F_LABEL, fill="black")
    d1.text((s(CANVAS_W // 2 - 75), s(62)), "engineering — computer engineering",
            font=F_SUB, fill="black")
    # Title (centered)
    title = cfg["title"]
    tw = d1.textlength(title, font=F_TITLE)
    d1.text((s(CANVAS_W // 2) - tw // 2, s(85)),
            title, font=F_TITLE, fill="black")
    d1.text((s(30), s(110)), f"Date: {cfg['date']}", font=F_SUB, fill="black")
    d1.text((s(CANVAS_W // 2 - 35), s(110)), f"Duration: {cfg['duration']}",
            font=F_SUB, fill="black")
    d1.text((s(CANVAS_W - 100), s(110)), f"Total: {cfg['totalPts']} pts",
            font=F_SUB, fill="black")

    # Name field (left) + Student number row (right)
    d1.rectangle((s(NAME_X0), s(NAME_Y0), s(NAME_X0 + NAME_W), s(NAME_Y0 + NAME_H)),
                 outline="#666", width=2)
    d1.text((s(NAME_X0 + 4), s(NAME_Y0 - 14)), "FULL NAME",
            font=F_LABEL_SM, fill="black")

    d1.text((s(SN_X0), s(SN_Y0 - 14)), "STUDENT NUMBER",
            font=F_LABEL_SM, fill="black")
    sn_boxes = []
    for i in range(10):
        x = SN_X0 + i * (SN_BOX_W + SN_GAP)
        y = SN_Y0
        d1.rectangle((s(x), s(y), s(x + SN_BOX_W), s(y + SN_BOX_H)),
                     outline="#666", width=2)
        sn_boxes.append({"x": float(x), "y": float(y),
                         "w": float(SN_BOX_W), "h": float(SN_BOX_H)})

    name_field = {"x": float(NAME_X0), "y": float(NAME_Y0),
                  "w": float(NAME_W), "h": float(NAME_H)}

    # ── Q1 (MC) ──
    draw_card(d1, Q1_LEFT, Q1_TOP, Q1_W, Q1_H, "Question 1", 10)
    # Stem
    _wrap_text(d1, cfg["q1"]["stem"], Q1_LEFT + 8, Q1_TOP + 26, Q1_W - 16, F_TEXT)
    q1_options = _draw_mc_grid(d1, Q1_LEFT + 8, Q1_TOP + 70, cfg["q1"]["options"])

    # ── Q2 (MC) ──
    draw_card(d1, Q2_LEFT, Q2_TOP, Q2_W, Q2_H, "Question 2", 10)
    _wrap_text(d1, cfg["q2"]["stem"], Q2_LEFT + 8, Q2_TOP + 26, Q2_W - 16, F_TEXT)
    q2_options = _draw_mc_grid(d1, Q2_LEFT + 8, Q2_TOP + 70, cfg["q2"]["options"])

    # ── Q3 (matching) ──
    draw_card(d1, Q3_LEFT, Q3_TOP, Q3_W, Q3_H, "Question 3 (matching)", 10)
    _wrap_text(d1, cfg["q3"]["stem"], Q3_LEFT + 8, Q3_TOP + 26, Q3_W - 16, F_TEXT)

    # Two-column item table
    table_top = Q3_TOP + 50
    col_left_x = Q3_LEFT + 16
    col_right_x = Q3_LEFT + 200
    for i, (left_item, right_item) in enumerate(zip(cfg["q3"]["left"], cfg["q3"]["right"])):
        ry = table_top + i * 22
        d1.text((s(col_left_x), s(ry)), f"{i+1}. {left_item}",
                font=F_TEXT, fill="black")
        letter = "ABCD"[i]
        d1.text((s(col_right_x), s(ry)), f"{letter}. {right_item[:60]}",
                font=F_TEXT, fill="black")

    # 4 answer boxes
    ANS_BOX_W = 55
    ANS_BOX_H = 34
    ans_y = Q3_TOP + Q3_H - 50
    q3_answer_boxes: Dict[str, Dict[str, float]] = {}
    for i in range(4):
        bx = Q3_LEFT + 30 + i * (ANS_BOX_W + 28)
        d1.text((s(bx - 16), s(ans_y + 8)), f"{i+1}:",
                font=F_LABEL_SM, fill="black")
        d1.rectangle((s(bx), s(ans_y),
                      s(bx + ANS_BOX_W), s(ans_y + ANS_BOX_H)),
                     outline="#888", width=2)
        q3_answer_boxes[str(i + 1)] = {
            "x": float(bx), "y": float(ans_y),
            "w": float(ANS_BOX_W), "h": float(ANS_BOX_H),
        }

    # ── Q4 (fill blank) ──
    draw_card(d1, Q4_LEFT, Q4_TOP, Q4_W, Q4_H, "Question 4 (fill in the blank)", 10)
    _wrap_text(d1, cfg["q4"]["stem"], Q4_LEFT + 8, Q4_TOP + 26, Q4_W - 16, F_TEXT)
    FILL_W = 130
    FILL_H = 34
    fill_y = Q4_TOP + Q4_H - 50
    fill_x = Q4_LEFT + 30
    d1.text((s(fill_x - 16), s(fill_y + 8)), "1:",
            font=F_LABEL_SM, fill="black")
    d1.rectangle((s(fill_x), s(fill_y),
                  s(fill_x + FILL_W), s(fill_y + FILL_H)),
                 outline="#888", width=2)
    q4_answer_boxes = {"1": {"x": float(fill_x), "y": float(fill_y),
                              "w": float(FILL_W), "h": float(FILL_H)}}

    # ── Q5 (open ended) on page 2 ──
    draw_card(d2, Q5_LEFT, Q5_TOP, Q5_W, Q5_H, "Question 5 (open ended)", 10)
    _wrap_text(d2, cfg["q5"]["stem"], Q5_LEFT + 8, Q5_TOP + 26, Q5_W - 16, F_TEXT)
    # Solution area with ruled lines
    sa_top = Q5_TOP + 60
    sa_bottom = Q5_TOP + Q5_H - 10
    sa_x0 = Q5_LEFT + 14
    sa_x1 = Q5_LEFT + Q5_W - 14
    d2.rectangle((s(sa_x0), s(sa_top), s(sa_x1), s(sa_bottom)),
                 outline="#bbb", width=1)
    # Ruled lines every 28 px
    line_y = sa_top + 28
    while line_y < sa_bottom - 6:
        d2.line((s(sa_x0 + 4), s(line_y), s(sa_x1 - 4), s(line_y)),
                fill="#ddd", width=1)
        line_y += 28

    q5_solution_area = {"x": float(sa_x0), "y": float(sa_top),
                        "w": float(sa_x1 - sa_x0), "h": float(sa_bottom - sa_top)}

    # ── Build map dict ──
    map_dict = {
        "version": 2,
        "examId": cfg["slug"],
        "generatedAt": "2026-05-09T00:00:00.000Z",
        "courseCode": cfg["courseCode"],
        "examType": cfg["examType"],
        "date": cfg["date"],
        "totalPages": 2,
        "pages": [
            {
                "pageId": f"P1_{cfg['slug']}",
                "pageIndex": 0,
                "pageWidth": CANVAS_W,
                "pageHeight": CANVAS_H,
                "anchors": {
                    name: {"type": "bullseye",
                           "center": {"x": cx, "y": cy},
                           "diameter": 16}
                    for name, (cx, cy) in ANCHORS_CANONICAL.items()
                },
                "studentNameField": name_field,
                "studentNumberRegion": {
                    "x": float(SN_X0 - 6), "y": float(SN_Y0 - 6),
                    "w": float(10 * (SN_BOX_W + SN_GAP) + 12),
                    "h": float(SN_BOX_H + 12),
                },
                "studentNumberBoxes": sn_boxes,
                "questions": {
                    "1": {
                        "type": "multiple_choice",
                        "boundingBox": {"x": float(Q1_LEFT), "y": float(Q1_TOP),
                                        "w": float(Q1_W), "h": float(Q1_H)},
                        "scoring": {"points": 10, "penaltyPerItem": 0, "itemCount": 4},
                        "expectedAnswer": {"correctOption": cfg["q1"]["correct"]},
                        "options": q1_options,
                    },
                    "2": {
                        "type": "multiple_choice",
                        "boundingBox": {"x": float(Q2_LEFT), "y": float(Q2_TOP),
                                        "w": float(Q2_W), "h": float(Q2_H)},
                        "scoring": {"points": 10, "penaltyPerItem": 0, "itemCount": 4},
                        "expectedAnswer": {"correctOption": cfg["q2"]["correct"]},
                        "options": q2_options,
                    },
                    "3": {
                        "type": "matching",
                        "boundingBox": {"x": float(Q3_LEFT), "y": float(Q3_TOP),
                                        "w": float(Q3_W), "h": float(Q3_H)},
                        "scoring": {"points": 10, "penaltyPerItem": 0, "itemCount": 4},
                        "expectedAnswer": {"correctMatches": cfg["q3"]["correct"]},
                        "answerBoxes": q3_answer_boxes,
                    },
                    "4": {
                        "type": "fill_blanks",
                        "boundingBox": {"x": float(Q4_LEFT), "y": float(Q4_TOP),
                                        "w": float(Q4_W), "h": float(Q4_H)},
                        "scoring": {"points": 10, "penaltyPerItem": 0, "itemCount": 1},
                        "expectedAnswer": {"correctBlanks": cfg["q4"]["correct"]},
                        "answerBoxes": q4_answer_boxes,
                    },
                },
            },
            {
                "pageId": f"P2_{cfg['slug']}",
                "pageIndex": 1,
                "pageWidth": CANVAS_W,
                "pageHeight": CANVAS_H,
                "anchors": {
                    name: {"type": "bullseye",
                           "center": {"x": cx, "y": cy},
                           "diameter": 16}
                    for name, (cx, cy) in ANCHORS_CANONICAL.items()
                },
                "questions": {
                    "5": {
                        "type": "open_ended",
                        "boundingBox": {"x": float(Q5_LEFT), "y": float(Q5_TOP),
                                        "w": float(Q5_W), "h": float(Q5_H)},
                        "scoring": {"points": 10, "penaltyPerItem": None},
                        "expectedAnswer": {"text": cfg["q5"]["correct"]},
                        "solutionArea": q5_solution_area,
                    },
                },
            },
        ],
    }

    return p1, p2, map_dict


def _wrap_text(d: ImageDraw.ImageDraw, text: str, x: int, y: int,
               max_w: int, font: ImageFont.FreeTypeFont) -> int:
    """Word-wrap text into max_w pixels (canonical). Returns final y."""
    if not text:
        return y
    words = text.split()
    cur = ""
    line_h = 16
    cy = y
    for w in words:
        cand = (cur + " " + w).strip() if cur else w
        if d.textlength(cand, font=font) > max_w * SCALE:
            d.text((s(x), s(cy)), cur, font=font, fill="black")
            cy += line_h
            cur = w
        else:
            cur = cand
    if cur:
        d.text((s(x), s(cy)), cur, font=font, fill="black")
        cy += line_h
    return cy


def _draw_mc_grid(d: ImageDraw.ImageDraw, x0: int, y0: int,
                  options: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """2×2 grid of empty option circles + labels. Returns the geometry
    map keyed by letter."""
    out: Dict[str, Dict[str, float]] = {}
    items = list(options.items())
    for i, (letter, txt) in enumerate(items):
        col = i % 2
        row = i // 2
        cx = x0 + col * 170
        cy = y0 + row * 22
        d.ellipse((s(cx), s(cy), s(cx + 12), s(cy + 12)),
                  outline="#222", width=2)
        d.text((s(cx + 18), s(cy - 2)), f"{letter}) {txt[:30]}",
               font=F_TEXT, fill="black")
        out[letter] = {"x": float(cx), "y": float(cy), "w": 12.0, "h": 12.0}
    return out


# ── Filling student answers on top of a blank ────────────────────────
def fill_student(blank_pages: Tuple[Image.Image, Image.Image],
                 cfg: Dict[str, Any], map_dict: Dict[str, Any],
                 student_number: str, full_name: str,
                 seed: int) -> Tuple[Image.Image, Image.Image]:
    """Return new (p1, p2) with student answers overlaid. All correct."""
    p1 = blank_pages[0].copy()
    p2 = blank_pages[1].copy()
    d1 = ImageDraw.Draw(p1)
    d2 = ImageDraw.Draw(p2)
    rng = random.Random(seed)

    pages = map_dict["pages"]
    page1_map = pages[0]
    page2_map = pages[1]

    # Full name in name field
    name_field = page1_map["studentNameField"]
    name_x = name_field["x"] + 6
    name_y = name_field["y"] + name_field["h"] - 22
    d1.text((s(name_x + rng.randint(-2, 2)), s(name_y + rng.randint(-1, 1))),
            full_name, font=F_HAND, fill=(15, 25, 95))

    # Student number digits
    for i, ch in enumerate(student_number):
        if i >= len(page1_map["studentNumberBoxes"]):
            break
        box = page1_map["studentNumberBoxes"][i]
        bx = box["x"] + box["w"] / 2
        by = box["y"] + box["h"] / 2
        cw = d1.textlength(ch, font=F_DIGIT)
        d1.text((s(bx) - cw // 2 + rng.randint(-2, 2),
                 s(by) - 18 * SCALE + rng.randint(-2, 2)),
                ch, font=F_DIGIT, fill=(15, 25, 95))

    # Q1 — fill correct option bubble
    q1_correct = cfg["q1"]["correct"]
    q1_opts = page1_map["questions"]["1"]["options"]
    _fill_bubble(d1, q1_opts[q1_correct])

    # Q2 — fill correct option bubble
    q2_correct = cfg["q2"]["correct"]
    q2_opts = page1_map["questions"]["2"]["options"]
    _fill_bubble(d1, q2_opts[q2_correct])

    # Q3 — write correct letters in matching answer boxes
    q3_correct = cfg["q3"]["correct"]
    q3_boxes = page1_map["questions"]["3"]["answerBoxes"]
    for k, letter in q3_correct.items():
        box = q3_boxes[k]
        bx = box["x"] + box["w"] / 2
        by = box["y"] + box["h"] / 2
        cw = d1.textlength(letter, font=F_HAND_BIG)
        d1.text((s(bx) - cw // 2 + rng.randint(-3, 3),
                 s(by) - 16 * SCALE + rng.randint(-2, 2)),
                letter, font=F_HAND_BIG, fill=(15, 25, 95))

    # Q4 — write fill word (bigger font; small fonts confuse TrOCR)
    q4_correct = cfg["q4"]["correct"]
    q4_boxes = page1_map["questions"]["4"]["answerBoxes"]
    for k, word in q4_correct.items():
        box = q4_boxes[k]
        # Vertical-center the text inside the box.
        text_x = box["x"] + 8
        text_y = box["y"] + (box["h"] - 18) / 2
        _hand_text(d1, word, text_x, text_y, F_HAND_FILL, rng, jitter=0)

    # Q5 — write open-ended answer on ruled lines
    q5_text = cfg["q5"]["correct"]
    sa = page2_map["questions"]["5"]["solutionArea"]
    _hand_paragraph(d2, q5_text, sa["x"] + 8, sa["y"] + 6,
                    sa["w"] - 16, sa["h"] - 12, F_HAND_SM, rng)

    return p1, p2


def _fill_bubble(d: ImageDraw.ImageDraw, box: Dict[str, float]) -> None:
    cx = box["x"] + box["w"] / 2
    cy = box["y"] + box["h"] / 2
    r = box["w"] / 2 - 1.5
    d.ellipse((s(cx - r), s(cy - r), s(cx + r), s(cy + r)),
              fill="black")


def _hand_text(d: ImageDraw.ImageDraw, text: str, x: float, y: float,
               font: ImageFont.FreeTypeFont, rng: random.Random,
               jitter: int = 1) -> None:
    cx = s(x)
    cy = s(y)
    for ch in text:
        dx = rng.randint(-jitter, jitter) * SCALE
        dy = rng.randint(-jitter, jitter) * SCALE
        d.text((cx + dx, cy + dy), ch, font=font, fill=(15, 25, 95))
        cw = d.textlength(ch, font=font)
        cx += int(cw)


def _hand_paragraph(d: ImageDraw.ImageDraw, text: str,
                    x: float, y: float, w: float, h: float,
                    font: ImageFont.FreeTypeFont, rng: random.Random) -> None:
    """Word-wrap text into the given box, baseline-aligned to ~28-px ruled lines."""
    words = (text or "").split()
    line_h = 28  # canonical px ≈ ruled-line spacing
    inner_w = w - 8
    cur = ""
    cy = y + 4
    for word in words:
        cand = (cur + " " + word).strip() if cur else word
        if d.textlength(cand, font=font) > inner_w * SCALE:
            _hand_text(d, cur, x + 4, cy, font, rng)
            cy += line_h
            cur = word
            if cy + line_h > y + h:
                break
        else:
            cur = cand
    if cur and cy + line_h <= y + h + 4:
        _hand_text(d, cur, x + 4, cy, font, rng)


# ── Build .ikuexam from config ───────────────────────────────────────
def build_ikuexam(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "version": 1,
        "exam": {
            "faculty": "engineering",
            "department": "computer engineering",
            "course": cfg["courseName"],
            "courseCode": cfg["courseCode"],
            "examType": cfg["examType"],
            "date": cfg["date"],
            "duration": cfg["duration"],
            "instructions": "",
        },
        "questions": [
            {
                "type": "mc",
                "text": cfg["q1"]["stem"],
                "imagePreview": None,
                "options": list(cfg["q1"]["options"].values()),
                "correctAnswer": "ABCD".index(cfg["q1"]["correct"]),
                "correctAnswers": [], "correctText": "", "correctImagePreview": None,
                "spaceId": 0,
                "matchLeft": ["", ""], "matchRight": ["", ""], "matchCorrect": [],
                "fillText": "", "fillAnswers": [],
                "points": 10, "penaltyPerItem": 0,
            },
            {
                "type": "mc",
                "text": cfg["q2"]["stem"],
                "imagePreview": None,
                "options": list(cfg["q2"]["options"].values()),
                "correctAnswer": "ABCD".index(cfg["q2"]["correct"]),
                "correctAnswers": [], "correctText": "", "correctImagePreview": None,
                "spaceId": 0,
                "matchLeft": ["", ""], "matchRight": ["", ""], "matchCorrect": [],
                "fillText": "", "fillAnswers": [],
                "points": 10, "penaltyPerItem": 0,
            },
            {
                "type": "match",
                "text": cfg["q3"]["stem"],
                "imagePreview": None,
                "options": ["", "", "", ""],
                "correctAnswer": 0,
                "correctAnswers": [], "correctText": "", "correctImagePreview": None,
                "spaceId": 0,
                "matchLeft": cfg["q3"]["left"],
                "matchRight": cfg["q3"]["right"],
                "matchCorrect": [cfg["q3"]["correct"][str(i + 1)] for i in range(4)],
                "fillText": "", "fillAnswers": [],
                "points": 10, "penaltyPerItem": 0,
            },
            {
                "type": "fill",
                "text": "",
                "imagePreview": None,
                "options": ["", "", "", ""],
                "correctAnswer": 0,
                "correctAnswers": [], "correctText": "", "correctImagePreview": None,
                "spaceId": 0,
                "matchLeft": ["", ""], "matchRight": ["", ""], "matchCorrect": [],
                "fillText": cfg["q4"]["stem"],
                "fillAnswers": [cfg["q4"]["correct"]["1"]],
                "points": 10, "penaltyPerItem": 0,
            },
            {
                "type": "open",
                "text": cfg["q5"]["stem"],
                "imagePreview": None,
                "options": ["", "", "", ""],
                "correctAnswer": 0,
                "correctAnswers": [], "correctText": cfg["q5"]["correct"],
                "correctImagePreview": None,
                "spaceId": 0,
                "matchLeft": ["", ""], "matchRight": ["", ""], "matchCorrect": [],
                "fillText": "", "fillAnswers": [],
                "points": 10, "penaltyPerItem": 0,
            },
        ],
        "updatedAt": "2026-05-09T00:00:00.000Z",
    }


# ── PIL pages → in-memory PDF (PyMuPDF docs concatenable) ───────────
def pages_to_pdf_bytes(pages: List[Image.Image]) -> bytes:
    """Save a list of PIL pages to a single multi-page PDF and return bytes."""
    if not pages:
        return b""
    buf = BytesIO()
    pages[0].save(buf, "PDF", resolution=DPI, save_all=True,
                  append_images=pages[1:])
    return buf.getvalue()


# ── Driver ───────────────────────────────────────────────────────────
def run() -> None:
    for cfg in EXAMS:
        slug = cfg["slug"]
        print(f"\n=== {slug} ===")
        # Build blank pages once.
        blank_p1, blank_p2, map_dict = build_blank(cfg)

        # Install map.json + .ikuexam in userData/exams.
        (USER_DATA / f"{slug}.map.json").write_text(
            json.dumps(map_dict, indent=2), encoding="utf-8",
        )
        (USER_DATA / f"{slug}.ikuexam").write_text(
            json.dumps(build_ikuexam(cfg), indent=2), encoding="utf-8",
        )
        print(f"  installed {slug}.ikuexam + .map.json into {USER_DATA}")

        # Save blank.pdf (single copy) for visual reference.
        out_dir = DOWNLOADS / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        blank_pdf = pages_to_pdf_bytes([blank_p1, blank_p2])
        (out_dir / "_blank.pdf").write_bytes(blank_pdf)

        # Generate 30 students = 3 batches × 10. Sequential names + SNs.
        # Student numbers: 24 + course-code-prefix (last 2 digits) + 4-digit seq
        course_prefix = "".join(c for c in cfg["courseCode"] if c.isdigit())[:2].rjust(2, "0")
        for batch_idx in range(3):
            student_pages: List[Image.Image] = []
            for student_in_batch in range(10):
                idx = batch_idx * 10 + student_in_batch  # 0..29
                sn_seq = idx + 1
                student_number = f"24{course_prefix}{str(sn_seq).zfill(6)}"  # 10 digits
                full_name = TURKISH_NAMES[idx % len(TURKISH_NAMES)]
                p1, p2 = fill_student(
                    (blank_p1, blank_p2), cfg, map_dict,
                    student_number=student_number,
                    full_name=full_name,
                    seed=hash((slug, idx)) & 0xFFFFFFFF,
                )
                student_pages.append(p1)
                student_pages.append(p2)

            batch_path = out_dir / f"batch-{batch_idx + 1}.pdf"
            batch_path.write_bytes(pages_to_pdf_bytes(student_pages))
            kb = batch_path.stat().st_size // 1024
            print(f"  wrote {batch_path.name}  ({len(student_pages)//2} students, {kb} KB)")

    print(f"\nDone. All output under {DOWNLOADS}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run()
