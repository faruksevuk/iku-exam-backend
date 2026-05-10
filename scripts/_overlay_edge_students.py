"""Overlay 5 edge-case students onto the app-generated blank PDF.

Uses the REAL coordinates from the map.json the app's HeadlessExporter
produced, so the answers land exactly where the pipeline will look for
them. We never spawn Electron here — we only read the artefacts the app
already wrote and produce three files in Downloads:

  bil101-edge-cases-2026-blank.pdf      (copy of the real blank, for reference)
  bil101-edge-cases-2026-students.pdf   (the 5-student answers PDF to evaluate)
  bil101-edge-cases-2026-expected.json  (ground truth for comparison later)
"""
import json
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

# App's bulk export writes here (paths are baked into ipc.ts)
SOURCE_BLANK = Path(r"D:/repos/ExamGeneration/iku-exam-backend/samples/blanks") / f"{EXAM_ID}.pdf"

DOWNLOADS = Path(r"C:/Users/faruk/Downloads")
OUT_BLANK = DOWNLOADS / f"{EXAM_ID}-blank.pdf"
OUT_STUDENTS = DOWNLOADS / f"{EXAM_ID}-students.pdf"
OUT_EXPECTED = DOWNLOADS / f"{EXAM_ID}-expected.json"

RENDER_DPI = 200    # rasterise the blank at this DPI


# ── Fonts ─────────────────────────────────────────────────────────
def get_font(size: int, handwriting: bool = False) -> ImageFont.FreeTypeFont:
    handwriting_candidates = [
        "C:/Windows/Fonts/segoesc.ttf",
        "C:/Windows/Fonts/inkfree.ttf",
        "C:/Windows/Fonts/comic.ttf",
        "C:/Windows/Fonts/segoepr.ttf",
    ]
    normal_candidates = [
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    pool = handwriting_candidates if handwriting else normal_candidates
    for p in pool:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ── Map + blank loading ───────────────────────────────────────────
def load_map() -> dict:
    return json.loads(MAP_PATH.read_text(encoding="utf-8"))


def rasterise_blank() -> tuple[Image.Image, float, float]:
    """Render the app's blank PDF at RENDER_DPI. Returns (image, sx, sy)
    where sx, sy convert page-space coords into pixel coords."""
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
    """Return (sx, sy) converting page coords -> pixel coords."""
    page = exam_map["pages"][0]
    sx = blank_img.width / page["pageWidth"]
    sy = blank_img.height / page["pageHeight"]
    return sx, sy


# ── Drawing helpers ──────────────────────────────────────────────
def jitter(v: float, amt: float = 1.2) -> float:
    return v + random.uniform(-amt, amt)


def fill_bubble(d: ImageDraw.ImageDraw, opt: dict, sx: float, sy: float,
                darkness: float = 0.95) -> None:
    cx = (opt["x"] + opt["w"] / 2) * sx
    cy = (opt["y"] + opt["h"] / 2) * sy
    r = min(opt["w"] * sx, opt["h"] * sy) / 2 - max(1, opt["w"] * sx * 0.10)
    fill_int = int(255 * (1.0 - darkness))
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(fill_int, fill_int, fill_int))


def draw_letter_in_box(d: ImageDraw.ImageDraw, box: dict, letter: str,
                       font: ImageFont.FreeTypeFont,
                       sx: float, sy: float,
                       offset_xy: tuple[float, float] = (0, 0)) -> None:
    bx = box["x"] * sx
    by = box["y"] * sy
    bw = box["w"] * sx
    bh = box["h"] * sy
    bbox = d.textbbox((0, 0), letter, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cx = bx + (bw - tw) / 2 + offset_xy[0]
    cy = by + (bh - th) / 2 - 2 + offset_xy[1]
    d.text((jitter(cx, 0.8), jitter(cy, 0.8)), letter, font=font, fill="black")


def draw_word_in_box(d: ImageDraw.ImageDraw, box: dict, word: str,
                     font: ImageFont.FreeTypeFont,
                     sx: float, sy: float,
                     offset_xy: tuple[float, float] = (0, 0)) -> None:
    bx = box["x"] * sx
    by = box["y"] * sy
    bw = box["w"] * sx
    bh = box["h"] * sy
    bbox = d.textbbox((0, 0), word, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cx = bx + 8 + offset_xy[0]      # left-justified inside the box (with small pad)
    cy = by + (bh - th) / 2 - 2 + offset_xy[1]
    d.text((cx, cy), word, font=font, fill="black")


def write_student_number(d: ImageDraw.ImageDraw, boxes: list[dict],
                         sn: str, font: ImageFont.FreeTypeFont,
                         sx: float, sy: float) -> None:
    for i, digit in enumerate(sn[: len(boxes)]):
        box = boxes[i]
        bx = box["x"] * sx
        by = box["y"] * sy
        bw = box["w"] * sx
        bh = box["h"] * sy
        bbox = d.textbbox((0, 0), digit, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx = bx + (bw - tw) / 2
        cy = by + (bh - th) / 2 - 2
        d.text((jitter(cx, 1.0), jitter(cy, 1.0)), digit, font=font, fill="black")


# ── QR detection (so we know exactly where to scribble) ─────────
def find_qr_bbox(img: Image.Image) -> tuple[int, int, int, int] | None:
    """Return (x, y, w, h) bbox of the QR in the rendered blank, in pixels.

    OpenCV's QRCodeDetector return signature shifts between versions; we
    fall back through both shapes (3-tuple vs 4-tuple) before giving up.
    """
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    detector = cv2.QRCodeDetector()
    points = None
    try:
        ret = detector.detectAndDecodeMulti(arr)
        # Newer OpenCV: (retval, decoded_info, points, straight_qrcode)
        # Older OpenCV: (decoded_info, points, straight_qrcode)
        if len(ret) == 4:
            points = ret[2]
        elif len(ret) == 3:
            points = ret[1]
    except Exception:
        try:
            # Single-QR fallback
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


# ── Answers (ground truth) ────────────────────────────────────────
ANSWERS = {
    # Question key (string) -> per-type answer payload
    "1": {"type": "matching", "correctMatches": {"1": "B", "2": "D", "3": "A", "4": "C"}},
    "2": {"type": "multiple_choice", "correctOption": "B"},
    "3": {"type": "multi_select", "correctOptions": ["A", "C"]},
    "4": {"type": "matching", "correctMatches": {"1": "B", "2": "C", "3": "A", "4": "D"}},
    "5": {"type": "fill_blanks", "correctBlanks": {"1": "Python"}},
}


def render_student(blank_img: Image.Image, exam_map: dict,
                   student_number: str, edge_case: str,
                   qr_bbox_px: tuple[int, int, int, int] | None) -> Image.Image:
    """Return a new image with one student's answers drawn on top of the blank."""
    img = blank_img.copy()
    d = ImageDraw.Draw(img)
    sx, sy = compute_scales(img, exam_map)
    page = exam_map["pages"][0]

    # Pick handwriting font sizes proportional to page render
    f_digit = get_font(int(22 * sy), handwriting=True)
    f_letter = get_font(int(20 * sy), handwriting=True)
    f_word = get_font(int(18 * sy), handwriting=True)
    f_scribble = get_font(int(13 * sy), handwriting=True)

    # ── Student number boxes
    write_student_number(d, page["studentNumberBoxes"], student_number, f_digit, sx, sy)

    # ── Q1 matching
    q1 = page["questions"]["1"]
    correct1 = ANSWERS["1"]["correctMatches"]
    for slot, letter in correct1.items():
        draw_letter_in_box(d, q1["answerBoxes"][slot], letter, f_letter, sx, sy)

    # ── Q2 multiple choice
    q2 = page["questions"]["2"]
    correct2 = ANSWERS["2"]["correctOption"]
    darkness2 = 0.30 if edge_case == "light_bubble_and_offcenter" else 0.92
    fill_bubble(d, q2["options"][correct2], sx, sy, darkness=darkness2)

    # ── Q3 multi-select
    q3 = page["questions"]["3"]
    for letter in ANSWERS["3"]["correctOptions"]:
        fill_bubble(d, q3["options"][letter], sx, sy, darkness=0.92)

    # ── Q4 matching
    q4 = page["questions"]["4"]
    correct4 = ANSWERS["4"]["correctMatches"]
    for slot, letter in correct4.items():
        draw_letter_in_box(d, q4["answerBoxes"][slot], letter, f_letter, sx, sy)

    # ── Q5 fill blank
    q5 = page["questions"]["5"]
    correct5 = ANSWERS["5"]["correctBlanks"]["1"]
    off_x = 90 if edge_case == "light_bubble_and_offcenter" else 0
    draw_word_in_box(d, q5["answerBoxes"]["1"], correct5, f_word, sx, sy,
                     offset_xy=(off_x, 0))

    # ── Edge-case overlays
    if edge_case == "scribbled_qr" and qr_bbox_px is not None:
        qx, qy, qw, qh = qr_bbox_px
        for _ in range(80):
            x1 = random.uniform(qx, qx + qw)
            y1 = random.uniform(qy, qy + qh)
            x2 = x1 + random.uniform(-qw * 0.3, qw * 0.3)
            y2 = y1 + random.uniform(-qh * 0.3, qh * 0.3)
            d.line((x1, y1, x2, y2), fill=(20, 20, 20), width=max(3, int(qw * 0.015)))

    if edge_case == "marginal_scribbles":
        # Find safe areas to scribble in — empty stretches of the page.
        # We use approximate page-coord zones that don't intersect any answer box.
        notes = [
            "stack = LIFO",
            "Q3 -> A & C",
            "review later",
            "check Q4",
            "skip 5?",
            "study ch.4",
        ]
        # Page-space scribble zones; positions hand-tuned to avoid the
        # answer regions of this particular layout (right margin and bottom).
        zones = [
            {"x": 600, "y": 240, "w": 130, "h": 50},    # right margin near top
            {"x": 600, "y": 360, "w": 130, "h": 50},    # right margin near Q1/Q2
            {"x": 250, "y": 660, "w": 200, "h": 30},    # between Q5 and Q4
            {"x": 600, "y": 760, "w": 130, "h": 50},    # right margin near Q4
            {"x": 40, "y": 1010, "w": 700, "h": 20},    # bottom margin
        ]
        random.seed(7)
        for z in zones[:-1]:
            note = random.choice(notes)
            tx = (z["x"] + random.uniform(0, z["w"] * 0.15)) * sx
            ty = (z["y"] + random.uniform(0, z["h"] * 0.4)) * sy
            d.text((tx, ty), note, font=f_scribble, fill=(40, 40, 40))
        # Bottom-margin wavy doodle
        bz = zones[-1]
        prev_x = bz["x"] * sx
        prev_y = (bz["y"] + bz["h"] / 2) * sy
        for k in range(35):
            nx = prev_x + 18 * sx
            ny = (bz["y"] + bz["h"] / 2 + (k % 2) * 6) * sy
            d.line((prev_x, prev_y, nx, ny), fill=(40, 40, 40),
                   width=max(1, int(1.5 * sx)))
            prev_x, prev_y = nx, ny

    return img


def apply_rotation(img: Image.Image, degrees: float) -> Image.Image:
    """Rotate keeping canvas size, white background fill."""
    return img.rotate(degrees, resample=Image.BILINEAR, expand=False, fillcolor="white")


# ── Main pipeline ───────────────────────────────────────────────
def main() -> None:
    random.seed(2026)

    # Pre-flight: required artefacts must exist
    if not SOURCE_BLANK.exists():
        sys.exit(f"Missing blank PDF: {SOURCE_BLANK}")
    if not MAP_PATH.exists():
        sys.exit(f"Missing map.json: {MAP_PATH}")

    exam_map = load_map()
    blank_img = rasterise_blank()
    sx, sy = compute_scales(blank_img, exam_map)
    print(f"Blank rasterised: {blank_img.size}, scale=({sx:.3f}, {sy:.3f})")

    qr_bbox = find_qr_bbox(blank_img)
    if qr_bbox is None:
        print("WARNING: QR not auto-detected on blank — scribble fallback to TR corner.")
        qr_bbox = (int(blank_img.width * 0.82), int(blank_img.height * 0.03),
                   int(blank_img.width * 0.14), int(blank_img.width * 0.14))
    else:
        print(f"QR bbox (px) = {qr_bbox}")

    # Plan
    plan = [
        ("2200000001", "normal"),
        ("2200000002", "rotated_2deg"),
        ("2200000003", "scribbled_qr"),
        ("2200000004", "marginal_scribbles"),
        ("2200000005", "light_bubble_and_offcenter"),
    ]

    pages: list[Image.Image] = []
    for idx, (sn, case) in enumerate(plan):
        page = render_student(blank_img, exam_map, sn, case, qr_bbox)
        if case == "rotated_2deg":
            page = apply_rotation(page, +2.0)   # +deg = CCW = "sola dönmüş"
        pages.append(page)
        print(f"  Student {idx + 1} ({sn}) — {case}")

    # Stack into PDF
    pages[0].save(
        str(OUT_STUDENTS), "PDF",
        save_all=True, append_images=pages[1:],
        resolution=RENDER_DPI,
    )
    print(f"\nStudents PDF → {OUT_STUDENTS}  ({OUT_STUDENTS.stat().st_size / 1024:.1f} KB)")

    # Copy the blank into Downloads for reference
    shutil.copy2(str(SOURCE_BLANK), str(OUT_BLANK))
    print(f"Blank PDF    → {OUT_BLANK}  ({OUT_BLANK.stat().st_size / 1024:.1f} KB)")

    # ── Expected output JSON ────────────────────────────────────
    expected = {
        "examId": EXAM_ID,
        "totalStudents": 5,
        "totalQuestions": 5,
        "totalMaxPoints": 50,
        "notes": (
            "Every student answers all 5 questions correctly. Any score variance "
            "in the evaluation is purely the effect of the edge case applied to "
            "that page."
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
    notes_map = {
        "normal": (
            "Baseline / control. Any deviation here is a pipeline bug, not an "
            "edge-case effect."
        ),
        "rotated_2deg": (
            "Page rotated +2° (counter-clockwise). Anchors should still resolve "
            "via homography; alignment should absorb the tilt and all questions "
            "should score normally."
        ),
        "scribbled_qr": (
            "QR code obliterated by scribbles. QR detection will fail on this "
            "page; the splitter should fall back to sequential page grouping "
            "(pages_per_exam=1). The answers themselves are otherwise normal."
        ),
        "marginal_scribbles": (
            "Random pencil scribbles in the right margin, between-question gaps, "
            "and bottom margin. None of the scribbles enter the answer-box "
            "regions, so per-question crops should be unaffected."
        ),
        "light_bubble_and_offcenter": (
            "Q2's correct bubble is filled at ~30% darkness (above the 10% empty "
            "threshold, well below 50% — OMR may flag this as ambiguous and "
            "either guess B or send it to review). Q5 handwriting is nudged "
            "~90 px right of its default position, starting closer to the right "
            "edge of the box; the word stays inside the box but is off-centre."
        ),
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
