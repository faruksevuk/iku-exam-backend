"""
Character Reader — MNIST CNN for digits, TrOCR for letters/text.

Key improvements:
1. Student number digits: MNIST CNN (98.5% accuracy, <1ms per digit)
2. Expanded extraction: boxes are expanded by a margin to catch overflow handwriting
3. Border removal: crop inside the box border before recognition
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List
import base64

# How much to expand extraction boxes (px) to catch overflow handwriting
BOX_EXPAND = 3


def _expand_box(box: Dict, expand: int = BOX_EXPAND) -> Dict:
    """Expand a box by `expand` pixels on each side."""
    return {
        "x": box.get("x", 0) - expand,
        "y": box.get("y", 0) - expand,
        "w": box.get("w", 0) + expand * 2,
        "h": box.get("h", 0) + expand * 2,
    }


def _crop_inside_border(image: np.ndarray, box: Dict, inset: int = 4) -> np.ndarray:
    """Crop inside a box, skipping the border pixels."""
    bx = int(round(box.get("x", 0)))
    by = int(round(box.get("y", 0)))
    bw = int(round(box.get("w", 0)))
    bh = int(round(box.get("h", 0)))
    img_h, img_w = image.shape[:2]

    x1 = max(0, min(bx + inset, img_w - 1))
    y1 = max(0, min(by + inset, img_h - 1))
    x2 = min(bx + bw - inset, img_w)
    y2 = min(by + bh - inset, img_h)

    if x2 <= x1 or y2 <= y1:
        return np.full((20, 20, 3), 255, dtype=np.uint8)
    return image[y1:y2, x1:x2].copy()


def _is_blank(region: np.ndarray, threshold: float = 0.01) -> bool:
    """Check if a region is blank."""
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
    _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    return (np.sum(binary > 0) / binary.size) < threshold


def _pad_white(image: np.ndarray, pad: int = 12) -> np.ndarray:
    """Add white padding."""
    return cv2.copyMakeBorder(image, pad, pad, pad, pad,
                               cv2.BORDER_CONSTANT, value=(255, 255, 255))


def _image_to_base64(image: np.ndarray) -> str:
    """Convert to base64 JPEG."""
    _, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')


def read_digit_box(image: np.ndarray, box: Dict) -> Tuple[str, float]:
    """
    Read a single digit from a structured box using MNIST CNN.
    Expands the box slightly, crops inside border, classifies with CNN.
    """
    from digit_model import classify_digit

    expanded = _expand_box(box, BOX_EXPAND)
    inner = _crop_inside_border(image, expanded, inset=4)

    if _is_blank(inner):
        return ("?", 0.5)

    digit, conf = classify_digit(inner)
    return (str(digit), round(conf, 3))


def read_student_number(
    aligned_image: np.ndarray,
    digit_boxes: List[Dict],
) -> Tuple[str, float]:
    """
    Read student number by classifying each digit box with MNIST CNN.
    """
    digits = []
    confidences = []

    for box in digit_boxes:
        d, c = read_digit_box(aligned_image, box)
        digits.append(d)
        confidences.append(c)

    number = ''.join(digits)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return (number, round(avg_conf, 3))


def read_answer_box_letter(
    aligned_image: np.ndarray,
    box: Dict,
) -> Tuple[str, float]:
    """Read a single letter using TrOCR. Border removed, white padded."""
    from ocr import read_handwriting

    inner = _crop_inside_border(aligned_image, box, inset=3)
    padded = _pad_white(inner, pad=10)
    text, conf = read_handwriting(padded)

    if not text.strip():
        return ("", conf)

    letters = ''.join(c for c in text.upper() if c.isalpha())
    if letters:
        return (letters[0], conf)

    digits = ''.join(c for c in text if c.isdigit())
    if digits:
        return (digits[0], conf * 0.7)

    return (text.strip()[0].upper(), conf * 0.5)


def read_answer_box_text(
    aligned_image: np.ndarray,
    box: Dict,
) -> Tuple[str, float]:
    """Read longer text from an answer box using TrOCR."""
    from ocr import read_handwriting

    inner = _crop_inside_border(aligned_image, box, inset=3)
    padded = _pad_white(inner, pad=8)
    return read_handwriting(padded)


def crop_answer_region(
    aligned_image: np.ndarray,
    box: Dict,
) -> str:
    """Crop a region and return as base64 JPEG."""
    from scanner import extract_region
    region = extract_region(aligned_image, box)
    if region.size == 0:
        return ""
    return _image_to_base64(region)
