"""
Preprocessing — produce clean, read-ready images from aligned page crops.

Pipeline per region:
  1. Crop by map box coordinates (with expand + border inset)
  2. Grayscale
  3. Denoise (median blur)
  4. Contrast enhance (CLAHE for handwriting)
  5. Tight content crop (remove empty margins)
  6. (optional) Resize to model-native size

Also provides:
  - is_blank(region) → bool
  - encode_jpeg_b64(region) → str  (backward compat with review UI)
  - encode_webp_b64(region) → str  (for future storage/UI)

Every reader module should take preprocessed ndarrays from here.
"""

import base64
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

import config


# ── Raw cropping ──────────────────────────────────────────────────

def crop_region(image: np.ndarray, box: Dict) -> np.ndarray:
    """
    Extract a rectangular region from an aligned image.
    Safe-clips to image bounds; returns 1×1 black if box is degenerate.
    """
    x = max(0, int(round(box.get("x", 0))))
    y = max(0, int(round(box.get("y", 0))))
    w = int(round(box.get("w", 0)))
    h = int(round(box.get("h", 0)))

    img_h, img_w = image.shape[:2]
    x = min(x, img_w - 1)
    y = min(y, img_h - 1)
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    if w <= 0 or h <= 0:
        return np.zeros((1, 1, 3), dtype=np.uint8) if image.ndim == 3 else np.zeros((1, 1), dtype=np.uint8)
    return image[y:y + h, x:x + w].copy()


def expand_box(box: Dict, expand: Optional[int] = None) -> Dict:
    """Expand a box by `expand` pixels on each side (default: config.CROP_EXPAND_PX)."""
    e = config.CROP_EXPAND_PX if expand is None else expand
    return {
        "x": box.get("x", 0) - e,
        "y": box.get("y", 0) - e,
        "w": box.get("w", 0) + e * 2,
        "h": box.get("h", 0) + e * 2,
    }


def crop_inside_border(image: np.ndarray, box: Dict, inset: Optional[int] = None) -> np.ndarray:
    """Crop inside a box, skipping the border pixels."""
    i = config.BORDER_INSET_PX if inset is None else inset
    bx = int(round(box.get("x", 0)))
    by = int(round(box.get("y", 0)))
    bw = int(round(box.get("w", 0)))
    bh = int(round(box.get("h", 0)))
    img_h, img_w = image.shape[:2]

    x1 = max(0, min(bx + i, img_w - 1))
    y1 = max(0, min(by + i, img_h - 1))
    x2 = min(bx + bw - i, img_w)
    y2 = min(by + bh - i, img_h)

    if x2 <= x1 or y2 <= y1:
        return np.full((20, 20, 3), 255, dtype=np.uint8)
    return image[y1:y2, x1:x2].copy()


# ── Enhancement ──────────────────────────────────────────────────

def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale if not already."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Light median blur — kills dust specks without damaging strokes."""
    return cv2.medianBlur(gray, 3)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE — adaptive contrast, good for faint handwriting."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def tight_content_crop(gray: np.ndarray, pad: int = 4) -> np.ndarray:
    """
    Crop to the bounding box of dark content + small padding.
    Falls back to original if nothing dark found.
    """
    _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return gray
    x, y, w, h = cv2.boundingRect(coords)
    H, W = gray.shape
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return gray
    return gray[y0:y1, x0:x1]


def pad_white(image: np.ndarray, pad: int = 12) -> np.ndarray:
    """Add white padding around an image (helps OCR detect boundaries)."""
    value = 255 if image.ndim == 2 else (255, 255, 255)
    return cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=value)


# ── Combined presets ──────────────────────────────────────────────

def crop_raw(aligned: np.ndarray, box: Dict) -> np.ndarray:
    """
    Crop region without enhancement — used for the Review UI (color crop).
    Box is expanded slightly to catch overflow handwriting.
    """
    return crop_region(aligned, expand_box(box))


def crop_for_reading(aligned: np.ndarray, box: Dict, pad: int = 10) -> np.ndarray:
    """
    Minimal OCR-ready crop: inside-border crop + white padding.

    Reverted from the earlier denoise + CLAHE pipeline — those steps were
    actually hurting OCR quality on thin handwriting strokes (median blur
    bleeding the strokes, CLAHE amplifying paper texture into false ink).
    OCR engines (TrOCR, Tesseract) do their own internal preprocessing and
    perform better on the raw cropped pixels.

    Matches the OLD char_reader.read_answer_box_*  approach (inset + pad).
    """
    inner = crop_inside_border(aligned, box)
    return pad_white(inner, pad=pad)


def crop_for_digit(aligned: np.ndarray, box: Dict) -> np.ndarray:
    """
    Crop for digit CNN (28x28 MNIST-style).
    Expanded box + inside-border crop. DigitCNN handles its own preprocessing.
    """
    expanded = expand_box(box)
    inner = crop_inside_border(aligned, expanded)
    return inner


def crop_for_letter(aligned: np.ndarray, box: Dict) -> np.ndarray:
    """
    Letter crop for matching questions — same as crop_for_reading.

    Earlier versions did Otsu binarization + tight-bbox + center-on-canvas
    + 2-4x upsample, but those transforms occasionally destroyed thin
    handwritten strokes ("B" turning into noise, "fill" trimmed away).
    Plain inside-border crop + white padding is more reliable.
    """
    return crop_for_reading(aligned, box, pad=10)


def crop_for_letter_cnn_simple(aligned: np.ndarray, box: Dict, inset: int = 2) -> np.ndarray:
    """
    Minimal preprocessing for the letter CNN — preserves the FULL letter
    shape. Avoids the aggressive largest-connected-component selection used
    by `crop_for_letter_cnn`, which can mangle multi-stroke letters (e.g.
    drop one bowl of a "B" if Otsu disconnects it from the spine, or pick
    up a printed border sliver as the largest CC and warp the shape).

    Pipeline:
      1. Crop inside the printed box border by `inset` px (small inset to
         keep cropping minimal — user-tuned default is 2).
      2. Grayscale + Otsu binarize (white text on black bg, EMNIST shape).
      3. Light edge-zero (2 px) — kills 1-2 px border slivers, doesn't
         eat letter strokes.
      4. INTER_AREA resize to 28x28.

    Trade-off vs. the aggressive variant: keeps stray noise specks if any
    survived edge-zero, but for clean printed exam scans those are rare,
    and dropping a real stroke is far worse than including a speck.
    """
    inner = crop_inside_border(aligned, box, inset=inset)
    if inner.size == 0 or inner.shape[0] < 4 or inner.shape[1] < 4:
        return np.zeros((28, 28), dtype=np.uint8)

    gray = to_gray(inner)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    edge = 2
    th[:edge, :] = 0
    th[-edge:, :] = 0
    th[:, :edge] = 0
    th[:, -edge:] = 0

    return cv2.resize(th, (28, 28), interpolation=cv2.INTER_AREA)


def crop_for_letter_cnn(aligned: np.ndarray, box: Dict, inset: int = 6) -> np.ndarray:
    """
    EMNIST-style 28x28 crop for the letter CNN classifier.

    The model expects the same input distribution as the EMNIST 'letters'
    dataset: 28x28 grayscale, white character strokes on black background,
    centered, with minimal padding (~10-15% margin).

    `inset` controls how far inside the printed box border we crop before
    extracting the letter. Different insets (e.g. 6 vs 8) trade off slightly
    different border-sliver risk vs. clipping-the-stroke risk; the caller
    can ensemble multiple insets and vote.

    Pipeline:
      1. Crop generously inside the printed box border by `inset` px — the
         printed line is several pixels thick on a 200dpi scan, a small
         inset leaves slivers that Otsu picks up as content and inflates
         the bounding box.
      2. Grayscale + Otsu binarize (white text on black bg).
      3. Aggressive edge-zero (3 px) — removes any remaining border slivers.
      4. Largest-connected-component as the letter (drops random specks).
      5. Tight bbox of that component.
      6. Center on a square black canvas with minimal padding.
      7. INTER_AREA resize to 28x28.

    Returns: uint8 (28, 28) array, white strokes (255) on black bg (0).
    """
    inner = crop_inside_border(aligned, box, inset=inset)
    if inner.size == 0 or inner.shape[0] < 4 or inner.shape[1] < 4:
        return np.zeros((28, 28), dtype=np.uint8)

    gray = to_gray(inner)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Aggressive edge-zero — kills border slivers up to 3 px wide
    edge = 3
    th[:edge, :] = 0
    th[-edge:, :] = 0
    th[:, :edge] = 0
    th[:, -edge:] = 0

    # Largest connected component (drops random specks and remaining border bits)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if n_labels <= 1:
        return np.zeros((28, 28), dtype=np.uint8)
    # stats: [x, y, w, h, area]; index 0 is background, skip
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return np.zeros((28, 28), dtype=np.uint8)
    largest_idx = 1 + int(areas.argmax())
    letter_mask = (labels == largest_idx).astype(np.uint8) * 255

    # Reject tiny noise blobs
    coords = cv2.findNonZero(letter_mask)
    if coords is None:
        return np.zeros((28, 28), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(coords)
    if w < 4 or h < 4 or w * h < 30:
        return np.zeros((28, 28), dtype=np.uint8)
    letter = letter_mask[y:y + h, x:x + w]

    # Center on square black canvas with minimal padding (matches EMNIST density)
    side = max(letter.shape) + 4
    canvas = np.zeros((side, side), dtype=np.uint8)
    cy0 = (side - letter.shape[0]) // 2
    cx0 = (side - letter.shape[1]) // 2
    canvas[cy0:cy0 + letter.shape[0], cx0:cx0 + letter.shape[1]] = letter

    return cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)


# ══════════════════════════════════════════════════════════════════
#  Multi-line / multi-variant helpers (lazy cascade for hard reads)
# ══════════════════════════════════════════════════════════════════

def split_text_lines(image_gray: np.ndarray, min_band_height: int = 10) -> list:
    """
    Split a multi-line handwriting region into individual line crops.

    Uses horizontal projection of a binarized + horizontally-closed image
    so words on the same line stay merged. Returns a list of grayscale
    ndarrays, ordered top-to-bottom.

    Falls back to [original_image] if no clear bands found.
    """
    if image_gray.ndim != 2:
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)

    H, W = image_gray.shape
    if H < 20 or W < 20:
        return [image_gray]

    blur = cv2.GaussianBlur(image_gray, (3, 3), 0)
    _, inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Connect characters horizontally so a word stays as one band
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    linked = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)

    # Horizontal projection
    proj = linked.sum(axis=1) / 255.0
    threshold = max(4.0, W * 0.015)

    bands = []
    inside = False
    start = 0
    for y, val in enumerate(proj):
        if val >= threshold and not inside:
            start = y
            inside = True
        elif val < threshold and inside:
            if y - start >= min_band_height:
                bands.append((start, y))
            inside = False
    if inside and len(proj) - start >= min_band_height:
        bands.append((start, len(proj)))

    if not bands:
        return [image_gray]

    # Merge bands separated by a thin gap (descenders/ascenders)
    merged = []
    for s, e in bands:
        if merged and s - merged[-1][1] < 8:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # Crop each line with small vertical padding
    lines = []
    for s, e in merged:
        s = max(0, s - 6)
        e = min(H, e + 6)
        line = image_gray[s:e, :].copy()
        if line.shape[0] >= 8 and line.shape[1] >= 8:
            lines.append(line)

    return lines if lines else [image_gray]


def generate_ocr_variants(
    image_gray: np.ndarray,
    max_variants: Optional[int] = None,
    include_inverted: Optional[bool] = None,
) -> list:
    """
    Produce a small set of preprocessed variants for OCR robustness.

    Variant strategies:
      - Otsu threshold on denoised image (clean strokes)
      - CLAHE + adaptive threshold (preserves faint marks, robust to gradients)

    If `include_inverted=True`, each variant is also returned in inverted form
    (some scanners produce dark backgrounds; TrOCR sometimes prefers the inverse).

    Final list size: min(max_variants, len(strategies)) * (2 if inverted else 1).
    Defaults pulled from config.
    """
    if image_gray.ndim != 2:
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)

    n = config.MULTILINE_MAX_VARIANTS if max_variants is None else max_variants
    inv_flag = config.MULTILINE_TRY_INVERTED if include_inverted is None else include_inverted

    if image_gray.size == 0:
        return [image_gray]

    # Denoise once (shared)
    denoised = cv2.fastNlMeansDenoising(image_gray, None, 12, 7, 21)

    variants = []

    # Strategy 1: Otsu on denoised
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    # Strategy 2: CLAHE + adaptive
    if n >= 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
        adaptive = cv2.adaptiveThreshold(
            clahe, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            31, 11,
        )
        variants.append(adaptive)

    variants = variants[:max(1, n)]

    if inv_flag:
        variants = variants + [255 - v for v in variants]

    return variants


# ── Blank detection ──────────────────────────────────────────────

def is_blank(region: np.ndarray, threshold: Optional[float] = None) -> bool:
    """
    True if the region contains almost no dark pixels.
    Threshold defaults to config.BLANK_DARK_RATIO.
    """
    t = config.BLANK_DARK_RATIO if threshold is None else threshold
    if region.size == 0:
        return True
    gray = to_gray(region)
    _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    ratio = float(np.sum(binary > 0) / binary.size)
    return ratio < t


# ── Encoding (for storage / review UI) ───────────────────────────

def encode_jpeg_b64(image: np.ndarray, quality: Optional[int] = None) -> str:
    """Encode as JPEG base64 (backward-compatible with review UI)."""
    if image.size == 0:
        return ""
    q = config.JPEG_QUALITY if quality is None else quality
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return ""
    return base64.b64encode(buf).decode("utf-8")


def encode_webp_b64(image: np.ndarray, quality: Optional[int] = None) -> str:
    """Encode as WebP base64 (smaller, for future use)."""
    if image.size == 0:
        return ""
    q = config.WEBP_QUALITY if quality is None else quality
    ok, buf = cv2.imencode(".webp", image, [cv2.IMWRITE_WEBP_QUALITY, q])
    if not ok:
        return ""
    return base64.b64encode(buf).decode("utf-8")


def decode_b64(image_b64: str) -> Optional[np.ndarray]:
    """Decode a base64 image (jpeg/webp/png auto) back to BGR ndarray."""
    if not image_b64:
        return None
    raw = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
    try:
        data = base64.b64decode(raw)
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None
