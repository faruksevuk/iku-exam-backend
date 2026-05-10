"""Overlay 6 edge-case students onto the app-generated blank PDF.

v5 — fixes the root coordinate issue:
  The map.json's coordinates are in the editor's CSS coordinate space
  (756 × 1086), but the printed A4 PDF embeds the content at a different
  scale and with a ~40px inward offset (the page has implicit margins
  that aren't reflected in map.json). My naive `pageWidth ratio` scaling
  was therefore wrong by ~40 pixels in each direction — that's why the
  Q2 bubble and the Q5 fill word landed outside their boxes.

  Fix: detect the four bullseye centres in the rasterised blank, build
  a 3×3 perspective transform from map-space to actual pixel-space, and
  route every coordinate through it. This is exactly what the backend
  pipeline does for the scanned input — we now mirror that on the way
  in, so everything we draw lands precisely inside the printed boxes.

6 edge cases:
  1) Student 1 — +2° clockwise tilt (sağa)
  2) Student 2 — QR scribbled
  3) Student 3 — pencil notes in empty zones
  4) Student 4 — heavy scribble blot in an empty area
  5) Student 5 — 4 anchor bullseyes inked into solid black dots
  6) Student 6 — anchors blackened AND page tilted +2° CW
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


# ── Handwriting font ──────────────────────────────────────────
def hand_font(size: int) -> ImageFont.FreeTypeFont:
    for p in [
        "C:/Windows/Fonts/segoepr.ttf",
        "C:/Windows/Fonts/segoeprb.ttf",
        "C:/Windows/Fonts/inkfree.ttf",
        "C:/Windows/Fonts/segoesc.ttf",
        "C:/Windows/Fonts/comic.ttf",
    ]:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ── Loading / detection ──────────────────────────────────────
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


def detect_bullseye_centers(blank_img: Image.Image) -> dict[str, tuple[float, float]]:
    """Scan the 4 corner quadrants of the rasterised blank for the centroid
    of dark pixels. Each corner contains exactly one bullseye, so the dark
    centroid IS the bullseye centre."""
    arr = np.array(blank_img.convert("L"))
    h, w = arr.shape
    quad = int(min(w, h) * 0.06)
    out: dict[str, tuple[float, float]] = {}
    for edge, (cx0, cy0) in [
        ("TL", (0, 0)), ("TR", (w - quad, 0)),
        ("BL", (0, h - quad)), ("BR", (w - quad, h - quad)),
    ]:
        q = arr[cy0:cy0 + quad, cx0:cx0 + quad]
        mask = q < 100
        ys, xs = np.where(mask)
        if len(xs) == 0:
            # Fallback: corner of quadrant
            out[edge] = (cx0 + quad / 2, cy0 + quad / 2)
        else:
            out[edge] = (cx0 + float(xs.mean()), cy0 + float(ys.mean()))
    return out


# ── Map → pixel transform ────────────────────────────────────
def build_transform(blank_img: Image.Image, exam_map: dict) -> np.ndarray:
    """Return the 3×3 perspective transform that maps map-space points to
    actual pixel positions in the rasterised blank."""
    page = exam_map["pages"][0]
    a = page["anchors"]
    src = np.array([
        [a["TL"]["center"]["x"], a["TL"]["center"]["y"]],
        [a["TR"]["center"]["x"], a["TR"]["center"]["y"]],
        [a["BR"]["center"]["x"], a["BR"]["center"]["y"]],
        [a["BL"]["center"]["x"], a["BL"]["center"]["y"]],
    ], dtype=np.float32)
    detected = detect_bullseye_centers(blank_img)
    dst = np.array([
        detected["TL"], detected["TR"], detected["BR"], detected["BL"],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def map_pt(M: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Transform a single (x, y) from map-space to pixel-space."""
    out = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), M)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def map_box(M: np.ndarray, box: dict) -> tuple[float, float, float, float]:
    """Transform a (x, y, w, h) map-space box → pixel-space (px, py, pw, ph)
    via the 4-corner perspective transform."""
    pts = np.array([
        [[box["x"], box["y"]]],
        [[box["x"] + box["w"], box["y"]]],
        [[box["x"] + box["w"], box["y"] + box["h"]]],
        [[box["x"], box["y"] + box["h"]]],
    ], dtype=np.float32)
    out = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
    px = float(out[:, 0].min())
    py = float(out[:, 1].min())
    pw = float(out[:, 0].max()) - px
    ph = float(out[:, 1].max()) - py
    return (px, py, pw, ph)


# ── Placement primitives ────────────────────────────────────
def draw_centered(d: ImageDraw.ImageDraw, box_px, text: str,
                  font: ImageFont.FreeTypeFont,
                  jitter_px: float = 0.0) -> None:
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
    bx, by, bw, bh = box_px
    d.text((bx + pad_px, by + bh / 2), text, font=font, fill="black", anchor="lm")


def fill_bubble_pencil(d: ImageDraw.ImageDraw,
                       cx_px: float, cy_px: float, r_px: float,
                       darkness: float = 1.0, seed: int = 0) -> None:
    rng = random.Random(seed)
    inner_r = r_px - 1.5
    base = int(255 - 130 * darkness)
    d.ellipse((cx_px - inner_r, cy_px - inner_r, cx_px + inner_r, cy_px + inner_r),
              fill=(base, base, base))
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
                   M: np.ndarray,
                   sn: str, edge_case: str,
                   qr_bbox_px) -> Image.Image:
    img = blank_img.copy()
    d = ImageDraw.Draw(img)
    page = exam_map["pages"][0]

    # ── Compute font sizes from TRANSFORMED box dims (truth) ─
    sn_box_px = map_box(M, page["studentNumberBoxes"][0])
    match_box_px = map_box(M, page["questions"]["1"]["answerBoxes"]["1"])
    fill_box_px = map_box(M, page["questions"]["5"]["answerBoxes"]["1"])

    f_digit = hand_font(int(sn_box_px[3] * 0.60))
    f_letter = hand_font(int(match_box_px[3] * 0.62))
    f_word = hand_font(int(fill_box_px[3] * 0.60))

    # ── Student number digits ─────────────────────────────
    for i, digit in enumerate(sn[: len(page["studentNumberBoxes"])]):
        box_px = map_box(M, page["studentNumberBoxes"][i])
        draw_centered(d, box_px, digit, f_digit, jitter_px=0.35)

    # ── Q1 matching letters ───────────────────────────────
    for slot, letter in ANSWERS["1"]["correctMatches"].items():
        box_px = map_box(M, page["questions"]["1"]["answerBoxes"][slot])
        draw_centered(d, box_px, letter, f_letter, jitter_px=0.4)

    # ── Q2 multiple choice bubble (correct B) ─────────────
    opt = page["questions"]["2"]["options"][ANSWERS["2"]["correctOption"]]
    opt_px = map_box(M, opt)
    bcx = opt_px[0] + opt_px[2] / 2
    bcy = opt_px[1] + opt_px[3] / 2
    br = min(opt_px[2], opt_px[3]) / 2 - 1.5
    fill_bubble_pencil(d, bcx, bcy, br, darkness=1.0, seed=int(sn[-1]))

    # ── Q3 multi-select bubbles (correct A + C) ───────────
    for letter in ANSWERS["3"]["correctOptions"]:
        opt = page["questions"]["3"]["options"][letter]
        opt_px = map_box(M, opt)
        bcx = opt_px[0] + opt_px[2] / 2
        bcy = opt_px[1] + opt_px[3] / 2
        br = min(opt_px[2], opt_px[3]) / 2 - 1.5
        fill_bubble_pencil(d, bcx, bcy, br, darkness=1.0, seed=ord(letter))

    # ── Q4 matching letters ───────────────────────────────
    for slot, letter in ANSWERS["4"]["correctMatches"].items():
        box_px = map_box(M, page["questions"]["4"]["answerBoxes"][slot])
        draw_centered(d, box_px, letter, f_letter, jitter_px=0.4)

    # ── Q5 fill blank — "Python", left-middle ────────────
    box_px = map_box(M, page["questions"]["5"]["answerBoxes"]["1"])
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
        f_note = hand_font(int(28))    # ~14px page → ~28-30px pixel
        # Use transformed positions for empty zones below all answer regions.
        marks_map = [
            (60, 880, "stack = LIFO"),
            (450, 890, "Q2 -> B"),
            (60, 1020, "review later *"),
            (300, 1020, "study chapter 4"),
        ]
        for px, py, txt in marks_map:
            xp, yp = map_pt(M, px, py)
            d.text((xp, yp), txt, font=f_note, fill=(50, 50, 50))
        # Arrow + star also in transformed space
        ax, ay = map_pt(M, 600, 1025)
        d.line((ax, ay, ax + 30, ay), fill=(50, 50, 50), width=2)
        d.line((ax + 30, ay, ax + 22, ay - 5), fill=(50, 50, 50), width=2)
        d.line((ax + 30, ay, ax + 22, ay + 5), fill=(50, 50, 50), width=2)
        scx, scy = map_pt(M, 700, 1025)
        for ang in [0, 72, 144, 216, 288]:
            x2 = scx + math.cos(math.radians(ang - 90)) * 9
            y2 = scy + math.sin(math.radians(ang - 90)) * 9
            d.line((scx, scy, x2, y2), fill=(50, 50, 50), width=2)

    if edge_case == "scribble_blot":
        # Heavy pencil scribble in an empty zone below Q4/Q5.
        bcx, bcy = map_pt(M, 540, 900)
        rng = random.Random(909)
        for _ in range(220):
            ang = rng.uniform(0, 2 * math.pi)
            r1 = rng.uniform(0, 50)
            r2 = rng.uniform(0, 50)
            x1 = bcx + math.cos(ang) * r1
            y1 = bcy + math.sin(ang) * r1
            x2 = bcx + math.cos(ang + 0.7) * r2
            y2 = bcy + math.sin(ang + 0.7) * r2
            d.line((x1, y1, x2, y2), fill=(20, 20, 20), width=4)

    if edge_case in ("anchors_blackened", "anchors_blackened_and_tilted"):
        # Paint solid black discs over the DETECTED bullseye positions.
        # Sized comfortably larger than the bullseye itself.
        detected = detect_bullseye_centers(blank_img)
        # Approximate bullseye radius from one of the corner quadrants
        gray = np.array(blank_img.convert("L"))
        h, w = gray.shape
        quad = int(min(w, h) * 0.06)
        bullseye_r_estimate = quad * 0.16   # ≈ 22px at 200 DPI
        for edge, (cx, cy) in detected.items():
            r = bullseye_r_estimate * 1.6
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill="black")

    return img


def apply_rotation(img: Image.Image, degrees: float) -> Image.Image:
    """PIL rotates CCW for positive degrees; pass negative for CW."""
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
    M = build_transform(blank_img, exam_map)
    print(f"Blank: {blank_img.size}")
    print(f"Transform (map → pixel):")
    print(M)

    # Verify by transforming a known anchor and printing
    a_tl = exam_map["pages"][0]["anchors"]["TL"]["center"]
    tx, ty = map_pt(M, a_tl["x"], a_tl["y"])
    detected = detect_bullseye_centers(blank_img)
    print(f"  TL anchor: transformed=({tx:.1f},{ty:.1f}) "
          f"detected=({detected['TL'][0]:.1f},{detected['TL'][1]:.1f})  "
          f"(should match)")

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
        ("2200000006", "anchors_blackened_and_tilted"),
    ]

    pages = []
    for idx, (sn, case) in enumerate(plan):
        page = render_student(blank_img, exam_map, M, sn, case, qr_bbox)
        if case in ("tilted_2deg_cw", "anchors_blackened_and_tilted"):
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
            "Page rotated -2° (clockwise / sağa). All 4 corner anchors are "
            "intact; alignment should resolve via homography and absorb "
            "the tilt — all answers should score normally.",
        "scribbled_qr":
            "QR code obliterated by pencil scribbles. QR detection fails; "
            "splitter falls back to sequential page grouping. Answers "
            "themselves are clean.",
        "margin_writing":
            "Random pencil notes / arrow / star scattered below all "
            "answer regions and in the bottom margin. None of the marks "
            "intersect any answer box.",
        "scribble_blot":
            "One heavy pencil scribble blot in the empty zone below "
            "Q4/Q5 (centred around page-coord 540, 900). The blot sits "
            "well outside every answer box.",
        "anchors_blackened":
            "All four corner anchors inked into solid black discs — no "
            "bullseye pattern. The bullseye detector will fail; alignment "
            "may fall back to <2-anchor pure-resize mode and answer "
            "crops could shift a few pixels.",
        "anchors_blackened_and_tilted":
            "All four corner anchors inked into solid black discs AND "
            "the page rotated -2° clockwise. Combined edge case: alignment "
            "loses its anchors AND has tilt to correct. Worst case in this "
            "set; we want to see if pure-resize fallback survives the tilt.",
    }

    expected = {
        "examId": EXAM_ID,
        "totalStudents": 6,
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
