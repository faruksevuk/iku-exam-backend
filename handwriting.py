"""
Handwriting — all single-box reading lives here.

Three readers:
  1. DigitCNN  — MNIST-trained CNN for single digits (student number)
  2. Tesseract — single-character OCR with A-Z whitelist (matching)
                 single-word OCR (fill_blanks fallback)
  3. TrOCR     — transformer handwriting OCR (fill_blanks primary,
                 AI evaluation fallback)

Confidence cascade for fill_blanks:
  - TrOCR reads first. If confidence ≥ HIGH_CONF_THRESHOLD → accept.
  - Else run Tesseract as fallback.
    - If readings agree (fuzzy ≥ FUZZY_AGREE_THRESHOLD) → accept higher-conf.
    - Else flag needsReview and return the higher-conf reading.

All readers return a ReaderResult with (text, confidence, source,
needs_review, fallback_used).
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
import preprocessing


# ── Standard result shape ────────────────────────────────────────

@dataclass
class ReaderResult:
    text: str
    confidence: float
    source: str                       # "digit_cnn" | "tesseract" | "trocr" | ...
    needs_review: bool = False
    fallback_used: bool = False
    raw_debug: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════
#  1) DigitCNN — MNIST single-digit classifier
# ══════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MiniDigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


_digit_model: Optional[_MiniDigitCNN] = None


def _load_digit_cnn() -> _MiniDigitCNN:
    global _digit_model
    if _digit_model is not None:
        return _digit_model
    model = _MiniDigitCNN()
    model.load_state_dict(torch.load(config.DIGIT_CNN_PATH, map_location="cpu", weights_only=True))
    model.eval()
    _digit_model = model
    print("[Handwriting] DigitCNN loaded")
    return model


def _classify_digit(image: np.ndarray) -> Tuple[int, float]:
    """Classify a 0-9 digit image. Returns (digit, confidence 0-1)."""
    model = _load_digit_cnn()

    gray = preprocessing.to_gray(image)
    # MNIST convention: bright digit on dark background
    if gray.mean() > 127:
        gray = 255 - gray

    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0) / 255.0
    tensor = (tensor - 0.1307) / 0.3081  # MNIST normalization

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)
        conf, pred = probs.max(1)
    return (int(pred.item()), round(float(conf.item()), 4))


def read_student_number(
    aligned_image: np.ndarray,
    digit_boxes: List[dict],
) -> ReaderResult:
    """
    Read a student number by classifying each digit box with DigitCNN.
    Returns ReaderResult where text is the concatenated digit string.
    """
    digits: List[str] = []
    confidences: List[float] = []

    for box in digit_boxes:
        crop = preprocessing.crop_for_digit(aligned_image, box)
        if preprocessing.is_blank(crop):
            digits.append("?")
            confidences.append(0.5)
            continue
        digit, conf = _classify_digit(crop)
        digits.append(str(digit))
        confidences.append(conf)

    number = "".join(digits)
    avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    needs_review = avg_conf < config.HIGH_CONF_THRESHOLD or "?" in number

    return ReaderResult(
        text=number,
        confidence=avg_conf,
        source="digit_cnn",
        needs_review=needs_review,
        raw_debug={"per_digit_confidences": confidences},
    )


# ══════════════════════════════════════════════════════════════════
#  2) Tesseract — single-character and single-word OCR
# ══════════════════════════════════════════════════════════════════

_tesseract_available: Optional[bool] = None


def _ensure_tesseract() -> bool:
    """Configure pytesseract and verify the binary runs. Cached."""
    global _tesseract_available
    if _tesseract_available is not None:
        return _tesseract_available
    try:
        import pytesseract
        if os.path.exists(config.TESSERACT_CMD):
            pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
        # smoke: request version
        pytesseract.get_tesseract_version()
        _tesseract_available = True
        print("[Handwriting] Tesseract available")
    except Exception as e:
        print(f"[Handwriting] WARN: Tesseract unavailable ({e}) - matching/fill_blanks fallback disabled")
        _tesseract_available = False
    return _tesseract_available


def _tesseract_read(image: np.ndarray, cfg: str) -> Tuple[str, float]:
    """Run Tesseract with a given config; return (text, confidence 0-1)."""
    if not _ensure_tesseract():
        return ("", 0.0)
    try:
        import pytesseract
        # image_to_data gives us per-word confidence (0-100)
        data = pytesseract.image_to_data(
            image, config=cfg, output_type=pytesseract.Output.DICT
        )
        texts = [t for t in data.get("text", []) if t and t.strip()]
        confs_raw = data.get("conf", [])
        confs: List[int] = []
        for c in confs_raw:
            try:
                v = int(float(c))
                if v >= 0:
                    confs.append(v)
            except (TypeError, ValueError):
                continue

        text = " ".join(texts).strip()
        conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
        return (text, round(conf, 4))
    except Exception as e:
        print(f"[Handwriting] Tesseract error: {e}")
        return ("", 0.0)


def _read_letter_tesseract(crop_for_ocr: np.ndarray) -> Tuple[str, float]:
    """Single-character (A-Z) Tesseract read. Whitelist prevents hallucination."""
    text, conf = _tesseract_read(crop_for_ocr, config.TESSERACT_MATCHING_CONFIG)
    # Extract first alpha char (whitelist should guarantee this already)
    letters = "".join(c for c in text.upper() if c.isalpha())
    return (letters[0] if letters else "", conf)


def _read_word_tesseract(crop_for_ocr: np.ndarray) -> Tuple[str, float]:
    """Single-word Tesseract read for fill_blanks fallback."""
    text, conf = _tesseract_read(crop_for_ocr, config.TESSERACT_FILL_CONFIG)
    return (text.strip(), conf)


# ══════════════════════════════════════════════════════════════════
#  3) TrOCR — transformer handwriting OCR
# ══════════════════════════════════════════════════════════════════

_trocr_processor = None
_trocr_model = None


def _load_trocr():
    global _trocr_processor, _trocr_model
    if _trocr_processor is not None:
        return
    print("[Handwriting] Loading TrOCR (first time may download ~300MB)...")
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    model_name = "microsoft/trocr-base-handwritten"
    _trocr_processor = TrOCRProcessor.from_pretrained(model_name)
    _trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
    _trocr_model.eval()
    print("[Handwriting] TrOCR loaded")


def _read_trocr(image: np.ndarray) -> Tuple[str, float]:
    """Run TrOCR handwriting OCR. Returns (text, confidence 0-1)."""
    if image.size == 0 or image.shape[0] < 5 or image.shape[1] < 5:
        return ("", 0.0)

    # Short-circuit: if the crop is almost white, don't wake TrOCR
    if preprocessing.is_blank(image):
        return ("", 1.0)

    _load_trocr()

    from PIL import Image as PILImage
    if image.ndim == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    pil_img = PILImage.fromarray(rgb)

    pixel_values = _trocr_processor(images=pil_img, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = _trocr_model.generate(
            pixel_values,
            max_new_tokens=50,
            return_dict_in_generate=True,
            output_scores=True,
        )

    text = _trocr_processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()

    # Confidence = mean max-prob over generated tokens
    if outputs.scores:
        probs = [F.softmax(score, dim=-1).max().item() for score in outputs.scores]
        conf = sum(probs) / len(probs) if probs else 0.5
    else:
        conf = 0.5

    return (text, round(min(float(conf), 1.0), 4))


# ══════════════════════════════════════════════════════════════════
#  Public API — per-box readers with confidence cascade
# ══════════════════════════════════════════════════════════════════

def read_letter_box(aligned_image: np.ndarray, box: dict) -> ReaderResult:
    """
    Read a single handwritten letter (matching questions).
    Primary: Tesseract psm=10 + A-Z whitelist (hallucination-proof).
    No fallback needed — whitelist guarantees output ∈ {A..Z} or empty.
    """
    crop = preprocessing.crop_for_reading(aligned_image, box, pad=10)

    if preprocessing.is_blank(crop):
        return ReaderResult(text="", confidence=1.0, source="blank", needs_review=False)

    letter, conf = _read_letter_tesseract(crop)

    # No letter came out → low confidence, needs review
    if not letter:
        return ReaderResult(
            text="",
            confidence=max(conf, 0.3),
            source="tesseract",
            needs_review=True,
            raw_debug={"reason": "tesseract returned no letter"},
        )

    needs_review = conf < config.HIGH_CONF_THRESHOLD
    return ReaderResult(
        text=letter,
        confidence=conf,
        source="tesseract",
        needs_review=needs_review,
    )


def _fuzzy_agree(a: str, b: str) -> float:
    """Similarity ratio between two strings (0-1)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    try:
        from Levenshtein import ratio
        return float(ratio(a.lower().strip(), b.lower().strip()))
    except ImportError:
        # Fallback: exact match only
        return 1.0 if a.lower().strip() == b.lower().strip() else 0.0


def read_text_box(aligned_image: np.ndarray, box: dict) -> ReaderResult:
    """
    Read a handwritten word/phrase (fill_blanks questions).
    Cascade:
      1. TrOCR primary. If conf ≥ HIGH → accept.
      2. Else Tesseract fallback.
         - If agree (fuzzy ≥ FUZZY_AGREE) → pick higher-conf, small conf boost.
         - Else → pick higher-conf, flag needs_review.
    """
    crop_ocr = preprocessing.crop_for_reading(aligned_image, box, pad=8)

    if preprocessing.is_blank(crop_ocr):
        return ReaderResult(text="", confidence=1.0, source="blank", needs_review=False)

    trocr_text, trocr_conf = _read_trocr(crop_ocr)

    if trocr_conf >= config.HIGH_CONF_THRESHOLD:
        return ReaderResult(
            text=trocr_text,
            confidence=trocr_conf,
            source="trocr",
            needs_review=False,
        )

    # Fallback: run Tesseract too
    tess_text, tess_conf = _read_word_tesseract(crop_ocr)

    if not tess_text and not trocr_text:
        return ReaderResult(
            text="",
            confidence=max(trocr_conf, tess_conf, 0.3),
            source="trocr+tesseract",
            needs_review=True,
            fallback_used=True,
            raw_debug={"trocr": trocr_text, "tesseract": tess_text},
        )

    agreement = _fuzzy_agree(trocr_text, tess_text)
    picked_text, picked_conf, picked_source = (
        (trocr_text, trocr_conf, "trocr")
        if trocr_conf >= tess_conf
        else (tess_text, tess_conf, "tesseract")
    )

    if agreement >= config.FUZZY_AGREE_THRESHOLD:
        # Readers agree → boost confidence, don't force review
        boosted = min(1.0, picked_conf + 0.10)
        needs_review = boosted < config.HIGH_CONF_THRESHOLD
        return ReaderResult(
            text=picked_text,
            confidence=round(boosted, 4),
            source=f"{picked_source}(+fallback)",
            needs_review=needs_review,
            fallback_used=True,
            raw_debug={
                "trocr": (trocr_text, trocr_conf),
                "tesseract": (tess_text, tess_conf),
                "agreement": agreement,
            },
        )

    # Readers disagree → needs human review
    return ReaderResult(
        text=picked_text,
        confidence=picked_conf,
        source=f"{picked_source}(disagree)",
        needs_review=True,
        fallback_used=True,
        raw_debug={
            "trocr": (trocr_text, trocr_conf),
            "tesseract": (tess_text, tess_conf),
            "agreement": agreement,
        },
    )


def read_handwriting_image(image: np.ndarray) -> ReaderResult:
    """
    Read any handwriting from a pre-cropped image (used by AI evaluation fallback).
    No preprocessing — caller passes the image it wants OCR'd.
    """
    text, conf = _read_trocr(image)
    return ReaderResult(
        text=text,
        confidence=conf,
        source="trocr",
        needs_review=conf < config.HIGH_CONF_THRESHOLD,
    )


# ── Fuzzy match helper (used by grading) ─────────────────────────

def fuzzy_match(predicted: str, expected: str, case_sensitive: bool = False) -> Tuple[bool, float]:
    """Fuzzy string match via Levenshtein ratio. Returns (is_match, similarity)."""
    if case_sensitive:
        a, b = predicted.strip(), expected.strip()
    else:
        a, b = predicted.lower().strip(), expected.lower().strip()

    if not a and not b:
        return (True, 1.0)
    if not a or not b:
        return (False, 0.0)

    try:
        from Levenshtein import ratio
        sim = ratio(a, b)
    except ImportError:
        sim = 1.0 if a == b else 0.0

    return (sim >= config.FUZZY_MATCH_THRESHOLD, round(float(sim), 4))
