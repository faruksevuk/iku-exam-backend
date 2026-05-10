"""Overlay 5 edge-case students onto the app-generated blank PDF.

v3 changes over v2:
  - All answers are now drawn properly inside their boxes (block letters
    instead of cursive script, sized to ~70-75% of box height, centred
    with sub-pixel jitter only).
  - Bubble fills look like real pencil scribbles (multiple cross-hatched
    strokes), not uniform grey discs.
  - Five edge cases match the request:
      1) Student 1 — +2° clockwise rotation (sağa)
      2) Student 2 — QR scribbled
      3) Student 3 — random pencil notes/marks in blank zones
      4) Student 4 — one heavy scribble blot in an empty area
      5) Student 5 — anchor points filled to solid black dots

Outputs (all in Downloads):
  bil101-edge-cases-2026-blank.pdf      (copy of the real blank)
  bil101-edge-cases-2026-students.pdf   (5-page solution PDF)
  bil101-edge-cases-2026-expected.json  (ground truth + edge-case notes)
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


# ── Fonts ───────────────────────────────────────────────────────
def find_font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont:
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def block_font(size: int) -> ImageFont.FreeTypeFont:
    """Bold block-letter style — what most students actually use for
    answer boxes. Reads cleanly to OCR + neat to the human eye."""
    return find_font([
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/seguibl.ttf",
    ], size)


def script_font(size: int) -> ImageFont.FreeTypeFont:
    """Lightly script for the fill-blank word — still very readable."""
    return find_font([
        "C:/Windows/Fonts/segoesc.ttf",
        "C:/Windows/Fonts/inkfree.ttf",
        "C:/Windows/Fonts/comic.ttf",
    ], size)


# ── Map / blank loading ─────────────────────────────────────────
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


def compute_scales(blank_img: Image.Image, exam_map: dict) -> tuple[float, float]:
    page = exam_map["pages"][0]
    return (
        blank_img.width / page["pageWidth"],
        blank_img.height / page["pageHeight"],
    )


# ── Drawing primitives — centred placement ───────────────────────
def center_text_in_box(d: ImageDraw.ImageDraw, box_px: tuple[float, float, float, float],
                       text: str, font: ImageFont.FreeTypeFont,
                       jitter_px: float = 0.5,
                       offset: tuple[float, float] = (0, 0)) -> None:
    """Draw text centred inside a box specified in pixel coords (x, y, w, h)."""
    bx, by, bw, bh = box_px
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # textbbox already accounts for the baseline; we want pure visual centring,
    # so also offset by the bbox top.
    cx = bx + (bw - tw) / 2 - bbox[0] + offset[0]
    cy = by + (bh - th) / 2 - bbox[1] + offset[1]
    if jitter_px > 0:
        cx += random.uniform(-jitter_px, jitter_px)
        cy += random.uniform(-jitter_px, jitter_px)
    d.text((cx, cy), text, font=font, fill="black")


def left_text_in_box(d: ImageDraw.ImageDraw, box_px: tuple[float, float, float, float],
                     text: str, font: ImageFont.FreeTypeFont,
                     pad_px: float = 6.0,
                     offset: tuple[float, float] = (0, 0)) -> None:
    """Left-justify text inside a box with a small left padding."""
    bx, by, bw, bh = box_px
    bbox = d.textbbox((0, 0), text, font=font)
    th = bbox[3] - bbox[1]
    cx = bx + pad_px - bbox[0] + offset[0]
    cy = by + (bh - th) / 2 - bbox[1] + offset[1]
    d.text((cx, cy), text, font=font, fill="black")


def fill_bubble_pencil(d: ImageDraw.ImageDraw,
                       cx_px: float, cy_px: float, r_px: float,
                       darkness: float = 1.0,
                       seed: int = 0) -> None:
    """Fill a bubble with a pencil-scribble pattern of overlapping strokes.

    darkness=1.0 → heavy, fully marked; darkness=0.3 → light, ambiguous.
    """
    rng = random.Random(seed)
    # First a base soft-grey disc so darker strokes blend over it
    base_grey = int(255 - 100 * darkness)
    inner_r = r_px - 1.5
    d.ellipse((cx_px - inner_r, cy_px - inner_r, cx_px + inner_r, cy_px + inner_r),
              fill=(base_grey, base_grey, base_grey))
    # Then layered diagonal strokes — n grows with darkness so a light
    # fill has visibly fewer strokes (and is therefore lighter to OMR).
    n = max(2, int(round(14 * darkness)))
    stroke_color = int(60 - 50 * darkness)
    stroke_color = max(0, stroke_color)
    pen_width = max(1, int(r_px * 0.16))
    for i in range(n):
        ang = rng.uniform(0, math.pi)
        cosA, sinA = math.cos(ang), math.sin(ang)
        # Stroke endpoints on the bubble interior
        x1 = cx_px - cosA * inner_r * 0.92
        y1 = cy_px - sinA * inner_r * 0.92
        x2 = cx_px + cosA * inner_r * 0.92
        y2 = cy_px + sinA * inner_r * 0.92
        d.line((x1, y1, x2, y2),
               fill=(stroke_color, stroke_color, stroke_color), width=pen_width)


def page_to_px(box: dict, sx: float, sy: float) -> tuple[float, float, float, float]:
    return (box["x"] * sx, box["y"] * sy, box["w"] * sx, box["h"] * sy)


# ── QR detection ────────────────────────────────────────────────
def find_qr_bbox(img: Image.Image) -> tuple[int, int, int, int] | None:
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    detector = cv2.QRCodeDetector()
    points = None
    try:
        ret = detector.detectAndDecodeMulti(arr)
        if len(ret) == 4:
            points = ret[2]
        elif len(ret) == 3:
            points = ret[1]
    except Exception:
        try:
            _, pts, _ = detector.detectAndDecode(arr)
            points = np.array([pts]) if pts is not None else None
        except Exception:
            points = None
    if points is None or len(points) == 0:
        return None
    pts = np.array(points[0]).reshape(-1, 2)
    x = int(pts[:, 0].min())
    y = int(pts[:, 1].min())
    w = int(pts[:, 0].max() - x)
    h = int(pts[:, 1].max() - y)
    return x, y, w, h


# ── Answer key ──────────────────────────────────────────────────
ANSWERS = {
    "1": {"type": "matching", "correctMatches": {"1": "B", "2": "D", "3": "A", "4": "C"}},
    "2": {"type": "multiple_choice", "correctOption": "B"},
    "3": {"type": "multi_select", "correctOptions": ["A", "C"]},
    "4": {"type": "matching", "correctMatches": {"1": "B", "2": "C", "3": "A", "4": "D"}},
    "5": {"type": "fill_blanks", "correctBlanks": {"1": "Python"}},
}


# ── Student page render ────────────────────────────────────────
def render_student(blank_img: Image.Image, exam_map: dict,
                   student_number: str, edge_case: str,
                   qr_bbox_px: tuple[int, int, int, int] | None) -> Image.Image:
    """Draw one student's answers cleanly inside every box, then apply
    the edge-case overlay."""
    img = blank_img.copy()
    d = ImageDraw.Draw(img)
    sx, sy = compute_scales(img, exam_map)
    page = exam_map["pages"][0]

    # ── Student number digits ─────────────────────────────────
    sn_boxes = page["studentNumberBoxes"]
    digit_box_h = sn_boxes[0]["h"] * sy
    f_digit = block_font(int(digit_box_h * 0.65))
    for i, digit in enumerate(student_number[: len(sn_boxes)]):
        box_px = page_to_px(sn_boxes[i], sx, sy)
        center_text_in_box(d, box_px, digit, f_digit, jitter_px=0.4)

    # ── Q1 matching ───────────────────────────────────────────
    q1 = page["questions"]["1"]
    box_h = q1["answerBoxes"]["1"]["h"] * sy
    f_letter = block_font(int(box_h * 0.70))
    for slot, letter in ANSWERS["1"]["correctMatches"].items():
        box_px = page_to_px(q1["answerBoxes"][slot], sx, sy)
        center_text_in_box(d, box_px, letter, f_letter, jitter_px=0.4)

    # ── Q2 multiple choice ───────────────────────────────────
    q2 = page["questions"]["2"]
    correct2 = ANSWERS["2"]["correctOption"]
    opt = q2["options"][correct2]
    cx = (opt["x"] + opt["w"] / 2) * sx
    cy = (opt["y"] + opt["h"] / 2) * sy
    r = min(opt["w"] * sx, opt["h"] * sy) / 2 - 1.5
    fill_bubble_pencil(d, cx, cy, r, darkness=1.0, seed=int(student_number[-1]))

    # ── Q3 multi-select ──────────────────────────────────────
    q3 = page["questions"]["3"]
    for letter in ANSWERS["3"]["correctOptions"]:
        opt = q3["options"][letter]
        cx = (opt["x"] + opt["w"] / 2) * sx
        cy = (opt["y"] + opt["h"] / 2) * sy
        r = min(opt["w"] * sx, opt["h"] * sy) / 2 - 1.5
        fill_bubble_pencil(d, cx, cy, r, darkness=1.0, seed=ord(letter))

    # ── Q4 matching ──────────────────────────────────────────
    q4 = page["questions"]["4"]
    for slot, letter in ANSWERS["4"]["correctMatches"].items():
        box_px = page_to_px(q4["answerBoxes"][slot], sx, sy)
        center_text_in_box(d, box_px, letter, f_letter, jitter_px=0.4)

    # ── Q5 fill blank — "Python" ─────────────────────────────
    q5 = page["questions"]["5"]
    fbox = q5["answerBoxes"]["1"]
    box_px = page_to_px(fbox, sx, sy)
    f_word = script_font(int(box_px[3] * 0.62))
    left_text_in_box(d, box_px, ANSWERS["5"]["correctBlanks"]["1"], f_word,
                     pad_px=8.0)

    # ── Edge-case-specific overlays ─────────────────────────
    if edge_case == "scribbled_qr" and qr_bbox_px is not None:
        qx, qy, qw, qh = qr_bbox_px
        # Heavy scribble across the QR square
        for _ in range(120):
            x1 = random.uniform(qx, qx + qw)
            y1 = random.uniform(qy, qy + qh)
            x2 = x1 + random.uniform(-qw * 0.35, qw * 0.35)
            y2 = y1 + random.uniform(-qh * 0.35, qh * 0.35)
            d.line((x1, y1, x2, y2), fill=(15, 15, 15),
                   width=max(3, int(qw * 0.02)))

    if edge_case == "margin_writing":
        # Diverse marks in empty zones — short notes, a tiny calculation,
        # an arrow, a star — all OUTSIDE answer regions.
        rng = random.Random(303)
        f_small = script_font(int(13 * sy))
        f_tiny = block_font(int(11 * sy))
        # Page-space zones safely between answer regions (right margin)
        marks = [
            (575, 240, "stack = LIFO"),
            (575, 360, "Q2 -> B"),
            (575, 460, "int + float"),
            (575, 760, "review *"),
            (320, 670, "2 + 3 = 5"),
        ]
        for px, py, txt in marks:
            d.text((px * sx, py * sy), txt, font=f_small, fill=(45, 45, 45))
        # Small arrow pointing toward Q3
        ax, ay = 560 * sx, 540 * sy
        d.line((ax, ay, ax - 40, ay + 8), fill=(45, 45, 45), width=2)
        d.line((ax - 40, ay + 8, ax - 30, ay - 2), fill=(45, 45, 45), width=2)
        d.line((ax - 40, ay + 8, ax - 30, ay + 18), fill=(45, 45, 45), width=2)
        # A tiny star in the bottom margin
        star_cx = 700 * sx
        star_cy = 1030 * sy
        for ang in [0, 72, 144, 216, 288]:
            x2 = star_cx + math.cos(math.radians(ang - 90)) * 9
            y2 = star_cy + math.sin(math.radians(ang - 90)) * 9
            d.line((star_cx, star_cy, x2, y2),
                   fill=(45, 45, 45), width=max(1, int(1.4 * sx)))

    if edge_case == "scribble_blot":
        # One heavy scribble blot in an empty area (right of Q3/Q4)
        # — like a student got distracted and scribbled a chunk of ink.
        blot_cx = 660 * sx
        blot_cy = 620 * sy
        rng = random.Random(909)
        for _ in range(180):
            ang = rng.uniform(0, 2 * math.pi)
            r1 = rng.uniform(0, 22 * sx)
            r2 = rng.uniform(0, 22 * sx)
            x1 = blot_cx + math.cos(ang) * r1
            y1 = blot_cy + math.sin(ang) * r1
            x2 = blot_cx + math.cos(ang + 0.6) * r2
            y2 = blot_cy + math.sin(ang + 0.6) * r2
            d.line((x1, y1, x2, y2), fill=(20, 20, 20),
                   width=max(2, int(2.5 * sx)))

    if edge_case == "anchors_blackened":
        # The four corner bullseye anchors are inked over into solid
        # black dots — no inner-ring pattern anymore. We over-paint
        # a black disc on top of each.
        for a in page["anchors"].values():
            cx = a["center"]["x"] * sx
            cy = a["center"]["y"] * sy
            r = (a["diameter"] / 2) * max(sx, sy) * 1.05    # slight overdraw
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill="black")

    return img


def apply_rotation(img: Image.Image, degrees: float) -> Image.Image:
    """Rotate keeping canvas size; white background fill.
    Note: PIL rotates CCW for positive degrees, so pass negative for CW."""
    return img.rotate(degrees, resample=Image.BILINEAR, expand=False, fillcolor="white")


# ── Main ────────────────────────────────────────────────────
def main() -> None:
    random.seed(2026)

    if not SOURCE_BLANK.exists():
        sys.exit(f"Missing blank PDF: {SOURCE_BLANK}")
    if not MAP_PATH.exists():
        sys.exit(f"Missing map.json: {MAP_PATH}")

    exam_map = load_map()
    blank_img = rasterise_blank()
    sx, sy = compute_scales(blank_img, exam_map)
    print(f"Blank rasterised: {blank_img.size}  scale=({sx:.3f}, {sy:.3f})")

    qr_bbox = find_qr_bbox(blank_img)
    if qr_bbox is None:
        qr_bbox = (int(blank_img.width * 0.82), int(blank_img.height * 0.03),
                   int(blank_img.width * 0.14), int(blank_img.width * 0.14))
        print(f"QR not detected — fallback bbox {qr_bbox}")
    else:
        print(f"QR bbox px = {qr_bbox}")

    # ── Five edge cases, exactly as requested ─────────────
    plan = [
        ("2200000001", "tilted_2deg_cw"),   # +2° sağa (clockwise)
        ("2200000002", "scribbled_qr"),
        ("2200000003", "margin_writing"),
        ("2200000004", "scribble_blot"),
        ("2200000005", "anchors_blackened"),
    ]

    pages: list[Image.Image] = []
    for idx, (sn, case) in enumerate(plan):
        page = render_student(blank_img, exam_map, sn, case, qr_bbox)
        if case == "tilted_2deg_cw":
            page = apply_rotation(page, -2.0)   # negative = CW in PIL
        pages.append(page)
        print(f"  Student {idx + 1} ({sn}) — {case}")

    pages[0].save(
        str(OUT_STUDENTS), "PDF",
        save_all=True, append_images=pages[1:],
        resolution=RENDER_DPI,
    )
    print(f"\nStudents PDF → {OUT_STUDENTS}  ({OUT_STUDENTS.stat().st_size / 1024:.1f} KB)")

    shutil.copy2(str(SOURCE_BLANK), str(OUT_BLANK))
    print(f"Blank PDF    → {OUT_BLANK}  ({OUT_BLANK.stat().st_size / 1024:.1f} KB)")

    # ── Expected outputs (ground truth) ──────────────────
    notes_map = {
        "tilted_2deg_cw": (
            "Page rotated -2° (clockwise / sağa). Four corner anchors are "
            "intact, so the alignment module should resolve via homography "
            "and absorb the tilt; all 5 answers should score normally."
        ),
        "scribbled_qr": (
            "QR code obliterated by scribbles. QR detection fails on this "
            "page; the splitter falls back to sequential page grouping "
            "(pages_per_exam=1). Answers themselves are otherwise normal."
        ),
        "margin_writing": (
            "Random pencil notes / arrows / a tiny star scattered through "
            "the right margin and between-question gaps. None of the marks "
            "enter any answer-box region, so per-question crops should be "
            "unaffected."
        ),
        "scribble_blot": (
            "One heavy pencil scribble blot in an empty area to the right "
            "of Q3/Q4. The blot sits well outside every answer box, so "
            "per-question crops should be clean."
        ),
        "anchors_blackened": (
            "All four corner anchors have been inked into solid black "
            "discs — no inner bullseye pattern anymore. The bullseye "
            "detector will likely fail; if alignment falls back to "
            "<2 anchors → pure resize, some boxes may shift a few pixels. "
            "Worth watching whether OMR / matching crops still land "
            "correctly under the degraded alignment."
        ),
    }

    expected = {
        "examId": EXAM_ID,
        "totalStudents": 5,
        "totalQuestions": 5,
        "totalMaxPoints": 50,
        "notes": (
            "Every student answers all 5 questions correctly. Any score "
            "variance in the evaluation comes purely from the edge case "
            "applied to that page."
        ),
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
    print(f"Expected     → {OUT_EXPECTED}  ({OUT_EXPECTED.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
