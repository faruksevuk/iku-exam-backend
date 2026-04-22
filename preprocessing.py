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
    Crop + enhance pipeline for OCR/CNN input.
    Returns grayscale, denoised, contrast-enhanced, padded image.
    """
    inner = crop_inside_border(aligned, box)
    gray = to_gray(inner)
    gray = denoise(gray)
    gray = enhance_contrast(gray)
    gray = pad_white(gray, pad=pad)
    return gray


def crop_for_digit(aligned: np.ndarray, box: Dict) -> np.ndarray:
    """
    Crop for digit CNN (28×28 MNIST-style).
    Expanded box + inside-border crop, grayscale, no CLAHE (matches training).
    """
    expanded = expand_box(box)
    inner = crop_inside_border(aligned, expanded)
    return inner  # DigitCNN handles its own grayscale + resize


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
