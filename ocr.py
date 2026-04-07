"""
OCR Module — TrOCR for handwritten text recognition.

Uses Microsoft's TrOCR (Transformer-based OCR) for reading handwritten answers
from cropped answer box images.

Provides:
- read_handwriting(image) → (text, confidence)
- read_student_number(image, num_boxes, box_coords) → (number_string, confidence)
- fuzzy_match(predicted, expected) → (is_correct, similarity_ratio)
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image

# Lazy-load TrOCR to avoid slow startup
_processor = None
_model = None


def _load_trocr():
    """Lazy-load TrOCR model on first use."""
    global _processor, _model
    if _processor is not None:
        return

    print("[OCR] Loading TrOCR model (first time may download ~300MB)...")
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    model_name = "microsoft/trocr-base-handwritten"
    _processor = TrOCRProcessor.from_pretrained(model_name)
    _model = VisionEncoderDecoderModel.from_pretrained(model_name)
    _model.eval()
    print("[OCR] TrOCR loaded.")


def read_handwriting(image: np.ndarray) -> Tuple[str, float]:
    """
    Read handwritten text from a cropped image region.
    Returns (text, confidence) where confidence is 0.0-1.0.
    """
    _load_trocr()

    if image.size == 0 or image.shape[0] < 5 or image.shape[1] < 5:
        return ("", 0.0)

    # Convert BGR to RGB PIL Image
    if len(image.shape) == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    pil_img = Image.fromarray(rgb)

    # Check if image is mostly blank (white)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    dark_ratio = np.sum(binary > 0) / binary.size
    if dark_ratio < 0.005:  # less than 0.5% dark pixels = blank
        return ("", 1.0)  # confident it's blank

    import torch
    pixel_values = _processor(images=pil_img, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = _model.generate(
            pixel_values,
            max_new_tokens=50,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = outputs.sequences
    text = _processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # Compute confidence from token probabilities
    if outputs.scores:
        import torch.nn.functional as F
        probs = [F.softmax(score, dim=-1).max().item() for score in outputs.scores]
        confidence = sum(probs) / len(probs) if probs else 0.5
    else:
        confidence = 0.5

    return (text, round(min(confidence, 1.0), 4))


def read_digit(image: np.ndarray) -> Tuple[str, float]:
    """Read a single digit from a small box image."""
    text, conf = read_handwriting(image)
    # Extract just digits
    digits = ''.join(c for c in text if c.isdigit())
    if len(digits) == 0:
        return ("", conf)
    return (digits[0], conf)


def read_student_number(
    aligned_image: np.ndarray,
    box_coords: List[Dict],
) -> Tuple[str, float]:
    """
    Read student number from individual digit boxes.
    box_coords: list of {"x", "y", "w", "h"} for each digit box.
    Returns (10-digit string, average confidence).
    """
    from scanner import extract_region

    digits = []
    confidences = []

    for box in box_coords:
        region = extract_region(aligned_image, box)
        digit, conf = read_digit(region)
        digits.append(digit if digit else "?")
        confidences.append(conf)

    number = "".join(digits)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return (number, round(avg_conf, 4))


def fuzzy_match(predicted: str, expected: str, case_sensitive: bool = False) -> Tuple[bool, float]:
    """
    Fuzzy string comparison using Levenshtein ratio.
    Returns (is_match, similarity_ratio).
    is_match = True if similarity > 0.80
    """
    if not case_sensitive:
        predicted = predicted.lower().strip()
        expected = expected.lower().strip()
    else:
        predicted = predicted.strip()
        expected = expected.strip()

    if not predicted and not expected:
        return (True, 1.0)
    if not predicted or not expected:
        return (False, 0.0)

    from Levenshtein import ratio
    sim = ratio(predicted, expected)

    return (sim >= 0.80, round(sim, 4))
