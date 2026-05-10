"""Overlay 5 edge-case students onto the app-generated blank PDF.

v4 changes vs v3:
  - **All marks land cleanly inside their boxes** via PIL's anchor system
    ("mm" centre-centre for matching cells and digits, "lm" left-middle
    for the fill-blank word). No more bbox arithmetic guesswork.
  - Handwriting fonts everywhere a student would write — Segoe Script
    primary, Ink Free / Comic Sans fallbacks.
  - Bubble pencil-scribble fill recentred on the bubble's geometric
    centre so the scribbled disc never bleeds outside the printed circle.
  - Same five edge cases (CW tilt, scribbled QR, margin writing,
    scribble blot, blackened anchors).

Outputs (Downloads):
  bil101-edge-cases-2026-blank.pdf
  bil101-edge-cases-2026-students.pdf
  bil101-edge-cases-2026-expected.json
"""
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXAM_ID = "bil101-edge-cases-2026"
APPDATA = Path(os.environ["APPDATA"]) / "iku-exam-generator"
MAP_PATH = APPDATA / "exams" / f"{EXAM_ID}.map.json"
SOURCE_BLANK = Path(r"D:/repos/ExamGeneration/iku-exam-backend/samples/blanks") / f"{EXAM_ID}.pdf"

DOWNLOADS = Path(r"C:/Users/faruk/Downloads")
OUT_BLANK = DOWNLOADS / f"{EXAM_ID}-blank.pdf"
OUT_STUDENTS = DOWNLOADS / f"{EXAM_ID}-students.pdf"
OUT_EXPECTED = DOWNLOADS / f"{EXAM_ID}-expected.json"

RENDER_DPI = 200


# ── Handwriting fonts ─────────────────────────────────────────
def hand_font(size: int) -> ImageFont.FreeTypeFont:
    """A handwriting / casual font. Tries Segoe Print first because it
    has good legibility in answer boxes, then falls back."""
    candidates = [
        "C:/Windows/Fonts/segoepr.ttf",       # Segoe Print — clean handwriting
        "C:/Windows/Fonts/segoeprb.ttf",      # Segoe Print Bold
        "C:/Windows/Fonts/inkfree.ttf",       # Ink Free
        "C:/Windows/Fonts/segoesc.ttf",       # Segoe Script
        "C:/Windows/Fonts/comic.ttf",         # Comic Sans (last resort)
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ── Loading ───────────────────────────────────────────────────
def load_map() -> dict:
    return json.loads(MAP_PATH.read_text(encoding="utf-8"))


def rasterise_blank() -> Image.Image:
    doc = fitz.open(str(SOURCE_BLANK))
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=RENDER_DPI, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    img = Image.fromarray(arr, mode="RGB")
    doc.close()
    return img


def scales(blank_img: Image.Image, exam_map: dict) -> tuple[float, float]:
    page = exam_map["pages"][0]
    return blank_img.width / page["pageWidth"], blank_img.height / page["pageHeight"]


# ── Placement primitives — anchor-based, pixel-clean ─────────
def draw_centered(d: ImageDraw.ImageDraw, box_px, text: str,
                  font: ImageFont.FreeTypeFont,
                  jitter_px: float = 0.0) -> None:
    """Draw text centred horizontally + vertically in a (x, y, w, h) box.

    Uses PIL anchor='mm' so we place the visual middle of the glyph at
    the box's middle. This is the only way to get reliable centring
    across fonts — bbox arithmetic over-corrects on fonts with extra
    line-height padding.
    """
    bx, by, bw, bh = box_px
    cx = bx + bw / 2
    cy = by + bh / 2
    if jitter_px > 0:
        cx += random.uniform(-jitter_px, jitter_px)
        cy += random.uniform(-jitter_px, jitter_px)
    d.text((cx, cy), text, font=font, fill="black", anchor="mm")


def draw_left_middle(d: ImageDraw.ImageDraw, box_px, text: str,
                     font: ImageFont.FreeTypeFont,
                     pad_px: float = 10.0) -> None:
    """Draw text vertically centred with a left-pad in a box."""
    bx, by, bw, bh = box_px
    cx = bx + pad_px
    cy = by + bh / 2
    d.text((cx, cy), text, font=font, fill="black", anchor="lm")


def fill_bubble_pencil(d: ImageDraw.ImageDraw,
                       cx_px: float, cy_px: float, r_px: float,
                       darkness: float = 1.0, seed: int = 0) -> None:
    """Pencil-scribble fill, perfectly centred on (cx_px, cy_px)."""
    rng = random.Random(seed)
    inner_r = r_px - 1.5
    # Soft base grey
    base = int(255 - 130 * darkness)
    d.ellipse((cx_px - inner_r, cy_px - inner_r, cx_px + inner_r, cy_px + inner_r),
              fill=(base, base, base))
    # Layered strokes — denser & darker as darkness rises
    n = max(2, int(round(16 * darkness)))
    stroke = max(0, int(40 - 40 * darkness))
    pw = max(1, int(r_px * 0.18))
    for _ in range(n):
        ang = rng.uniform(0, math.pi)
        ca, sa = math.cos(ang), math.sin(ang)
        x1 = cx_px - ca * inner_r * 0.94
        y1 = cy_px - sa * inner_r * 0.94
        x2 = cx_px + ca * inner_r * 0.94
        y2 = cy_px + sa * inner_r * 0.94
        d.line((x1, y1, x2, y2), fill=(stroke, stroke, stroke), width=pw)


def page_to_px(box: dict, sx: float, sy: float):
    return (box["x"] * sx, box["y"] * sy, box["w"] * sx, box["h"] * sy)


# ── QR detection ─────────────────────────────────────────────
def find_qr_bbox(img: Image.Image):
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    det = cv2.QRCodeDetector()
    pts = None
    try:
        ret = det.detectAndDecodeMulti(arr)
        if len(ret) == 4:
            pts = ret[2]
        elif len(ret) == 3:
            pts = ret[1]
    except Exception:
        try:
            _, p, _ = det.detectAndDecode(arr)
            pts = np.array([p]) if p is not None else None
        except Exception:
            pts = None
    if pts is None or len(pts) == 0:
        return None
    p = np.array(pts[0]).reshape(-1, 2)
    return (int(p[:, 0].min()), int(p[:, 1].min()),
            int(p[:, 0].max() - p[:, 0].min()),
            int(p[:, 1].max() - p[:, 1].min()))


# ── Answer key ──────────────────────────────────────────────
ANSWERS = {
    "1": {"type": "matching",
          "correctMatches": {"1": "B", "2": "D", "3": "A", "4": "C"}},
    "2": {"type": "multiple_choice", "correctOption": "B"},
    "3": {"type": "multi_select", "correctOptions": ["A", "C"]},
    "4": {"type": "matching",
          "correctMatches": {"1": "B", "2": "C", "3": "A", "4": "D"}},
    "5": {"type": "fill_blanks", "correctBlanks": {"1": "Python"}},
}


def render_student(blank_img: Image.Image, exam_map: dict,
                   sn: str, edge_case: str,
                   qr_bbox_px) -> Image.Image:
    img = blank_img.copy()
    d = ImageDraw.Draw(img)
    sx, sy = scales(img, exam_map)
    page = exam_map["pages"][0]

    # ── Font sizing (proportional to box heights in pixels) ──
    sn_box_h_px = page["studentNumberBoxes"][0]["h"] * sy
    match_box_h_px = page["questions"]["1"]["answerBoxes"]["1"]["h"] * sy
    fill_box_h_px = page["questions"]["5"]["answerBoxes"]["1"]["h"] * sy

    f_digit = hand_font(int(sn_box_h_px * 0.62))
    f_letter = hand_font(int(match_box_h_px * 0.62))
    f_word = hand_font(int(fill_box_h_px * 0.62))

    # ── Student number digits ─────────────────────────────
    for i, digit in enumerate(sn[: len(page["studentNumberBoxes"])]):
        box_px = page_to_px(page["studentNumberBoxes"][i], sx, sy)
        draw_centered(d, box_px, digit, f_digit, jitter_px=0.35)

    # ── Q1 matching ───────────────────────────────────────
    for slot, letter in ANSWERS["1"]["correctMatches"].items():
        box_px = page_to_px(page["questions"]["1"]["answerBoxes"][slot], sx, sy)
        draw_centered(d, box_px, letter, f_letter, jitter_px=0.4)

    # ── Q2 multiple choice (single bubble) ────────────────
    opt = page["questions"]["2"]["options"][ANSWERS["2"]["correctOption"]]
    bcx = (opt["x"] + opt["w"] / 2) * sx
    bcy = (opt["y"] + opt["h"] / 2) * sy
    br = min(opt["w"] * sx, opt["h"] * sy) / 2 - 1.5
    fill_bubble_pencil(d, bcx, bcy, br, darkness=1.0, seed=int(sn[-1]))

    # ── Q3 multi-select bubbles ───────────────────────────
    for letter in ANSWERS["3"]["correctOptions"]:
        opt = page["questions"]["3"]["options"][letter]
        bcx = (opt["x"] + opt["w"] / 2) * sx
        bcy = (opt["y"] + opt["h"] / 2) * sy
        br = min(opt["w"] * sx, opt["h"] * sy) / 2 - 1.5
        fill_bubble_pencil(d, bcx, bcy, br, darkness=1.0, seed=ord(letter))

    # ── Q4 matching ───────────────────────────────────────
    for slot, letter in ANSWERS["4"]["correctMatches"].items():
        box_px = page_to_px(page["questions"]["4"]["answerBoxes"][slot], sx, sy)
        draw_centered(d, box_px, letter, f_letter, jitter_px=0.4)

    # ── Q5 fill blank — "Python" left-justified ──────────
    box_px = page_to_px(page["questions"]["5"]["answerBoxes"]["1"], sx, sy)
    draw_left_middle(d, box_px, ANSWERS["5"]["correctBlanks"]["1"], f_word,
                     pad_px=12.0)

    # ── Edge-case overlays ───────────────────────────────
    if edge_case == "scribbled_qr" and qr_bbox_px is not None:
        qx, qy, qw, qh = qr_bbox_px
        for _ in range(120):
            x1 = random.uniform(qx, qx + qw)
            y1 = random.uniform(qy, qy + qh)
            x2 = x1 + random.uniform(-qw * 0.35, qw * 0.35)
            y2 = y1 + random.uniform(-qh * 0.35, qh * 0.35)
            d.line((x1, y1, x2, y2), fill=(15, 15, 15),
                   width=max(3, int(qw * 0.02)))

    if edge_case == "margin_writing":
        f_note = hand_font(int(14 * sy))
        # Page-space zones safely below all answer regions and in the
        # bottom margin — well away from anything the pipeline reads.
        marks = [
            (60, 870, "stack = LIFO"),
            (520, 880, "Q2 -> B"),
            (60, 1020, "review later *"),
            (300, 1020, "study chapter 4"),
        ]
        for px, py, txt in marks:
            d.text((px * sx, py * sy), txt, font=f_note, fill=(50, 50, 50))
        # Small arrow + star bottom margin
        ax, ay = 600 * sx, 1025 * sy
        d.line((ax, ay, ax + 30, ay), fill=(50, 50, 50), width=2)
        d.line((ax + 30, ay, ax + 22, ay - 5), fill=(50, 50, 50), width=2)
        d.line((ax + 30, ay, ax + 22, ay + 5), fill=(50, 50, 50), width=2)
        sx_star, sy_star = 700 * sx, 1025 * sy
        for ang in [0, 72, 144, 216, 288]:
            x2 = sx_star + math.cos(math.radians(ang - 90)) * 9
            y2 = sy_star + math.sin(math.radians(ang - 90)) * 9
            d.line((sx_star, sy_star, x2, y2),
                   fill=(50, 50, 50), width=max(1, int(1.4 * sx)))

    if edge_case == "scribble_blot":
        # Heavy pencil-scribble blot, clearly in an empty zone below the
        # answer regions (between Q4/Q5 row and the bottom anchors).
        blot_cx = 540 * sx
        blot_cy = 900 * sy
        rng = random.Random(909)
        for _ in range(220):
            ang = rng.uniform(0, 2 * math.pi)
            r1 = rng.uniform(0, 26 * sx)
            r2 = rng.uniform(0, 26 * sx)
            x1 = blot_cx + math.cos(ang) * r1
            y1 = blot_cy + math.sin(ang) * r1
            x2 = blot_cx + math.cos(ang + 0.7) * r2
            y2 = blot_cy + math.sin(ang + 0.7) * r2
            d.line((x1, y1, x2, y2), fill=(20, 20, 20),
                   width=max(2, int(2.5 * sx)))

    if edge_case == "anchors_blackened":
        # The four corner bullseyes are inked into solid black discs.
        # The map.json anchor coords don't always match the printed
        # position because CSS layout offsets aren't reflected in the
        # map. Detect the actual bullseye centres in the rasterised
        # blank by scanning the four corner quadrants for the centroid
        # of dark pixels (the bullseye is the only dark shape there).
        bw_h, bw_w = img.height, img.width
        gray = np.array(img.convert("L"))
        quad_size = int(min(bw_w, bw_h) * 0.06)
        for cx_corner, cy_corner in [
            (0, 0), (bw_w - quad_size, 0),
            (0, bw_h - quad_size), (bw_w - quad_size, bw_h - quad_size),
        ]:
            q = gray[cy_corner:cy_corner + quad_size,
                     cx_corner:cx_corner + quad_size]
            mask = q < 100
            if mask.sum() < 10:
                continue
            ys, xs = np.where(mask)
            cx = cx_corner + xs.mean()
            cy = cy_corner + ys.mean()
            # Use a radius slightly larger than the bullseye's own size
            r = quad_size * 0.30
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill="black")

    return img


def apply_rotation(img: Image.Image, degrees: float) -> Image.Image:
    """PIL rotates CCW for positive degrees; pass negative for CW."""
    return img.rotate(degrees, resample=Image.BILINEAR, expand=False, fillcolor="white")


def main() -> None:
    random.seed(2026)

    if not SOURCE_BLANK.exists():
        sys.exit(f"Missing blank PDF: {SOURCE_BLANK}")
    if not MAP_PATH.exists():
        sys.exit(f"Missing map.json: {MAP_PATH}")

    exam_map = load_map()
    blank_img = rasterise_blank()
    sx, sy = scales(blank_img, exam_map)
    print(f"Blank: {blank_img.size}  scale=({sx:.3f}, {sy:.3f})")

    qr_bbox = find_qr_bbox(blank_img)
    if qr_bbox is None:
        qr_bbox = (int(blank_img.width * 0.82), int(blank_img.height * 0.03),
                   int(blank_img.width * 0.14), int(blank_img.width * 0.14))
    print(f"QR bbox px = {qr_bbox}")

    plan = [
        ("2200000001", "tilted_2deg_cw"),
        ("2200000002", "scribbled_qr"),
        ("2200000003", "margin_writing"),
        ("2200000004", "scribble_blot"),
        ("2200000005", "anchors_blackened"),
    ]

    pages = []
    for idx, (sn, case) in enumerate(plan):
        page = render_student(blank_img, exam_map, sn, case, qr_bbox)
        if case == "tilted_2deg_cw":
            page = apply_rotation(page, -2.0)
        pages.append(page)
        print(f"  Student {idx + 1} ({sn}) — {case}")

    pages[0].save(
        str(OUT_STUDENTS), "PDF",
        save_all=True, append_images=pages[1:],
        resolution=RENDER_DPI,
    )
    print(f"\nStudents → {OUT_STUDENTS}  ({OUT_STUDENTS.stat().st_size / 1024:.1f} KB)")

    shutil.copy2(str(SOURCE_BLANK), str(OUT_BLANK))
    print(f"Blank    → {OUT_BLANK}  ({OUT_BLANK.stat().st_size / 1024:.1f} KB)")

    notes_map = {
        "tilted_2deg_cw":
            "Page rotated -2° (clockwise / sağa). Anchors intact, so the "
            "alignment module should resolve via homography and absorb the "
            "tilt; all answers should score normally.",
        "scribbled_qr":
            "QR code obliterated by pencil scribbles. QR detection fails "
            "on this page; splitter falls back to sequential page grouping "
            "(pages_per_exam=1). Answers themselves are clean.",
        "margin_writing":
            "Random pencil notes / arrow / star scattered in the bottom "
            "margin and below all answer regions. None of the marks "
            "intersect any answer box, so per-question crops should be "
            "unaffected.",
        "scribble_blot":
            "One heavy pencil scribble blot in the empty area below Q4/Q5 "
            "(centred around page-coord 540, 900). The blot sits well "
            "outside every answer box.",
        "anchors_blackened":
            "All four corner anchors inked into solid black discs — no "
            "bullseye pattern. The bullseye detector will likely fail; "
            "alignment may fall back to <2-anchor pure-resize mode and "
            "answer crops may shift a few pixels.",
    }

    expected = {
        "examId": EXAM_ID,
        "totalStudents": 5,
        "totalQuestions": 5,
        "totalMaxPoints": 50,
        "notes": ("Every student answers all 5 questions correctly. Any "
                  "score variance comes purely from the edge case applied "
                  "to that page."),
        "answerKey": {
            "Q1_matching": ANSWERS["1"]["correctMatches"],
            "Q2_multiple_choice": ANSWERS["2"]["correctOption"],
            "Q3_multi_select": ANSWERS["3"]["correctOptions"],
            "Q4_matching": ANSWERS["4"]["correctMatches"],
            "Q5_fill_blank": ANSWERS["5"]["correctBlanks"]["1"],
        },
        "students": [],
    }
    for sn, case in plan:
        expected["students"].append({
            "studentNumber": sn,
            "edgeCase": case,
            "expectedTotalScore": 50,
            "expectedMaxPoints": 50,
            "shouldAllQuestionsBeCorrect": True,
            "expectedQuestions": {
                "1": {"type": "matching",
                      "expectedMatches": ANSWERS["1"]["correctMatches"],
                      "expectedScore": 10},
                "2": {"type": "multiple_choice",
                      "expectedOption": ANSWERS["2"]["correctOption"],
                      "expectedScore": 10},
                "3": {"type": "multi_select",
                      "expectedOptions": ANSWERS["3"]["correctOptions"],
                      "expectedScore": 10},
                "4": {"type": "matching",
                      "expectedMatches": ANSWERS["4"]["correctMatches"],
                      "expectedScore": 10},
                "5": {"type": "fill_blanks",
                      "expectedBlanks": ANSWERS["5"]["correctBlanks"],
                      "expectedScore": 10},
            },
            "edgeCaseNotes": notes_map[case],
        })

    OUT_EXPECTED.write_text(json.dumps(expected, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"Expected → {OUT_EXPECTED}  ({OUT_EXPECTED.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
