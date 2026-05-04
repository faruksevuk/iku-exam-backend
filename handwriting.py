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

# `os` is imported above and used here for LETTER_CNN_PATH existence check.


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


class _MiniLetterCNN(nn.Module):
    """
    EMNIST-letters classifier (26 classes, A-Z).

    Architecture inferred from the existing models/letter_cnn.pt state dict:
      - Same conv backbone as digit CNN (1->16->32 channels, two MaxPool)
      - Wider classifier head: Linear(1568, 128) -> ReLU -> Dropout -> Linear(128, 26)
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 26),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


_digit_model: Optional[_MiniDigitCNN] = None
_letter_model: Optional[_MiniLetterCNN] = None
_letter_cnn_available: Optional[bool] = None


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


def _load_letter_cnn() -> Optional[_MiniLetterCNN]:
    """Load the EMNIST-trained letter CNN. Returns None if weights missing/incompatible."""
    global _letter_model, _letter_cnn_available
    if _letter_cnn_available is False:
        return None
    if _letter_model is not None:
        return _letter_model
    try:
        if not os.path.exists(config.LETTER_CNN_PATH):
            print(f"[Handwriting] LetterCNN: weights not found at {config.LETTER_CNN_PATH}")
            _letter_cnn_available = False
            return None
        model = _MiniLetterCNN()
        state = torch.load(config.LETTER_CNN_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _letter_model = model
        _letter_cnn_available = True
        print("[Handwriting] LetterCNN loaded (EMNIST 26-class A-Z)")
        return model
    except Exception as e:
        print(f"[Handwriting] LetterCNN failed to load: {e} -> disabled")
        _letter_cnn_available = False
        _letter_model = None
        return None


def _preprocess_digit_for_cnn(image: np.ndarray) -> np.ndarray:
    """
    Convert a digit crop into the standard MNIST input distribution.

    MNIST convention (which our CNN was trained against):
      - 28x28 uint8, white digit (~255) on black background (~0)
      - tight digit, aspect-preserved scale to fit a 20x20 inner box
      - centered on the 28x28 canvas with 4px margin around the digit

    Naive resize-without-centering (the previous behavior) feeds the CNN a
    tiny digit afloat in whitespace plus possible border slivers — a
    distribution it has never seen. Hence the catastrophic misreads.

    Returns a 28x28 black canvas if no digit content is found.
    """
    gray = preprocessing.to_gray(image)
    if gray.size == 0 or gray.shape[0] < 4 or gray.shape[1] < 4:
        return np.zeros((28, 28), dtype=np.uint8)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Edge-zero — kill any leftover printed border slivers up to 2 px wide
    edge = 2
    binary[:edge, :] = 0
    binary[-edge:, :] = 0
    binary[:, :edge] = 0
    binary[:, -edge:] = 0

    # Largest connected component (drops dust, retains the digit body)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return np.zeros((28, 28), dtype=np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return np.zeros((28, 28), dtype=np.uint8)
    largest_idx = 1 + int(areas.argmax())
    digit_mask = (labels == largest_idx).astype(np.uint8) * 255

    coords = cv2.findNonZero(digit_mask)
    if coords is None:
        return np.zeros((28, 28), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(coords)
    if w < 2 or h < 4 or w * h < 12:
        return np.zeros((28, 28), dtype=np.uint8)
    digit = digit_mask[y:y + h, x:x + w]

    # Aspect-preserving scale to fit a 20x20 inner box
    target = 20
    if w >= h:
        new_w = target
        new_h = max(1, int(round(h * target / w)))
    else:
        new_h = target
        new_w = max(1, int(round(w * target / h)))
    resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center on the 28x28 canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y0 = (28 - new_h) // 2
    x0 = (28 - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _classify_digit(image: np.ndarray) -> Tuple[int, float]:
    """Classify a 0-9 digit image. Returns (digit, confidence 0-1)."""
    model = _load_digit_cnn()
    canvas = _preprocess_digit_for_cnn(image)

    tensor = torch.FloatTensor(canvas).unsqueeze(0).unsqueeze(0) / 255.0
    tensor = (tensor - 0.1307) / 0.3081  # MNIST normalization

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)
        conf, pred = probs.max(1)
    return (int(pred.item()), round(float(conf.item()), 4))


def _classify_letter(
    image_28x28: np.ndarray,
    allowed: Optional[set] = None,
) -> Optional[Tuple[str, float]]:
    """
    Classify a 28x28 EMNIST-style image as A-Z.
    Returns (letter, confidence 0-1) or None if model unavailable.

    Input expectations (matches preprocessing.crop_for_letter_cnn output):
      - shape: (28, 28) uint8
      - white strokes (≈255) on black background (≈0)

    HONEST `allowed` semantics (NOT renormalized softmax):
      The earlier implementation masked logits outside `allowed` to -inf
      before softmax, which renormalized confidence to sum to 1 over the
      allowed letters. That produced catastrophic dishonesty: when the
      CNN's real top-1 was OUTSIDE allowed (e.g. it saw "X" with 0.67
      probability and the real A/B mass was 0.003), the renormalized
      result still reported "A 0.87" — a fake 87% certainty on a letter
      the CNN does not believe in. Two independent insets repeating that
      lie boosted to 100% in the consensus check.

      The honest semantics:
        * Compute unrestricted softmax over all 26 classes once.
        * If the unrestricted top-1 is INSIDE `allowed`: return it with
          its raw probability — confidence is unchanged from the natural
          read.
        * Otherwise: pick whichever allowed letter has the highest raw
          probability and report that RAW probability (no renormalization).
          When the CNN doesn't really see any allowed letter, this honestly
          reports a low confidence (e.g. 0.002), so downstream review
          gating fires.
    """
    model = _load_letter_cnn()
    if model is None:
        return None

    if image_28x28.shape != (28, 28):
        image_28x28 = cv2.resize(image_28x28, (28, 28), interpolation=cv2.INTER_AREA)

    # Defensive: ensure white-on-black (training convention)
    if image_28x28.mean() > 127:
        image_28x28 = 255 - image_28x28

    tensor = torch.FloatTensor(image_28x28).unsqueeze(0).unsqueeze(0) / 255.0
    tensor = (tensor - 0.1307) / 0.3081  # EMNIST shares MNIST normalization

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]  # shape (26,)
        top_idx = int(torch.argmax(probs).item())
    top_letter = chr(ord("A") + top_idx)
    top_conf = float(probs[top_idx].item())

    allowed_norm: Optional[set] = None
    if allowed:
        allowed_norm = {
            L.upper() for L in allowed
            if isinstance(L, str) and len(L) == 1
            and "A" <= L.upper() <= "Z"
        } or None

    if not allowed_norm:
        return (top_letter, round(top_conf, 4))

    if top_letter in allowed_norm:
        # CNN's natural top is allowed — confidence is honest as-is.
        return (top_letter, round(top_conf, 4))

    # CNN's natural top is outside allowed. Pick best within allowed by
    # RAW probability and report that raw value — NO renormalization.
    best_letter = top_letter
    best_conf = 0.0
    for L in allowed_norm:
        idx = ord(L) - ord("A")
        p = float(probs[idx].item())
        if p > best_conf:
            best_conf = p
            best_letter = L
    return (best_letter, round(best_conf, 4))


def read_student_number(
    aligned_image: np.ndarray,
    digit_boxes: List[dict],
) -> ReaderResult:
    """
    Read a student number by combining two independent readers:

      1. PER-DIGIT DigitCNN — one classification per box. Specialized,
         hallucination-proof (always 0-9), no length/alignment ambiguity.
      2. SEQUENCE-LEVEL TrOCR over the whole strip with digit-only constraint.
         Sees spacing/context, can recover from a single CNN misread.

    Reconciliation (only when TrOCR length matches the box count):
      - Per-position: agree → keep + boost conf; disagree → trust the
        higher-confidence reader at that position.
      - Mismatched length → fall back to CNN-only and flag for review.
    """
    cnn_digits: List[str] = []
    cnn_confs: List[float] = []

    for box in digit_boxes:
        crop = preprocessing.crop_for_digit(aligned_image, box)
        if preprocessing.is_blank(crop):
            cnn_digits.append("?")
            cnn_confs.append(0.5)
            continue
        digit, conf = _classify_digit(crop)
        cnn_digits.append(str(digit))
        cnn_confs.append(conf)

    cnn_number = "".join(cnn_digits)
    cnn_avg = sum(cnn_confs) / len(cnn_confs) if cnn_confs else 0.0

    # Cross-check gating: skip the 3-5 s TrOCR strip pass only when the
    # per-digit CNN is uniformly very confident AND has no blanks. The
    # threshold is intentionally high — anything below it (a single 0.94 in
    # an otherwise 0.99 sequence) still fires the cross-check so we never
    # silently accept a low-confidence digit just because the average is
    # high. Reading quality is preserved exactly for any uncertain student.
    SN_TROCR_SKIP_MIN_CONF = 0.97
    skip_trocr = (
        bool(cnn_confs)
        and "?" not in cnn_digits
        and min(cnn_confs) >= SN_TROCR_SKIP_MIN_CONF
    )

    # Sequence-level cross-check: crop the whole SN strip and run digit-only TrOCR.
    trocr_text = ""
    trocr_conf = 0.0
    try:
        if not skip_trocr and digit_boxes and aligned_image.size > 0:
            pad = 4
            min_x = max(0, int(min(b["x"] for b in digit_boxes)) - pad)
            min_y = max(0, int(min(b["y"] for b in digit_boxes)) - pad)
            max_x = min(aligned_image.shape[1],
                        int(max(b["x"] + b["w"] for b in digit_boxes)) + pad)
            max_y = min(aligned_image.shape[0],
                        int(max(b["y"] + b["h"] for b in digit_boxes)) + pad)
            if max_x > min_x and max_y > min_y:
                strip = aligned_image[min_y:max_y, min_x:max_x]
                trocr_text, trocr_conf = _read_trocr_digit_strip(strip)
    except Exception as e:
        if config.VERBOSE:
            print(f"[Handwriting] SN TrOCR cross-check failed: {e}")

    # Reconcile per-position when TrOCR returned a usable same-length sequence.
    final_digits = list(cnn_digits)
    final_confs = list(cnn_confs)
    agreements = 0
    disagreements = 0
    overrides = 0
    used_trocr = False
    if trocr_text and len(trocr_text) == len(cnn_digits):
        used_trocr = True
        for i, (cd, td) in enumerate(zip(cnn_digits, trocr_text)):
            if cd == td:
                agreements += 1
                # Boost confidence — two independent readers agree
                final_confs[i] = min(1.0, max(cnn_confs[i], trocr_conf) * 1.05)
            else:
                disagreements += 1
                # Disagreement: take the higher-confidence reader for this position.
                # CNN is a per-digit specialist, so we tilt toward it unless its
                # confidence is clearly lower than TrOCR's overall sequence conf.
                if cd == "?" or cnn_confs[i] < trocr_conf - 0.05:
                    final_digits[i] = td
                    final_confs[i] = trocr_conf
                    overrides += 1
                else:
                    # Keep CNN but penalize — one reader thinks it's wrong
                    final_confs[i] = cnn_confs[i] * 0.85

    final_number = "".join(final_digits)
    final_avg = round(sum(final_confs) / len(final_confs), 4) if final_confs else 0.0
    needs_review = (
        final_avg < config.HIGH_CONF_THRESHOLD
        or "?" in final_number
        or (used_trocr and disagreements > 1)
    )

    if used_trocr and overrides > 0:
        source = "digit_cnn+trocr"
    elif used_trocr:
        source = "digit_cnn(trocr_confirmed)"
    else:
        source = "digit_cnn"

    return ReaderResult(
        text=final_number,
        confidence=final_avg,
        source=source,
        needs_review=needs_review,
        raw_debug={
            "cnn_text": cnn_number,
            "cnn_avg_conf": round(cnn_avg, 4),
            "cnn_per_digit": [round(c, 4) for c in cnn_confs],
            "trocr_text": trocr_text,
            "trocr_conf": round(trocr_conf, 4),
            "agreements": agreements,
            "disagreements": disagreements,
            "overrides": overrides,
        },
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
    """
    Single-word Tesseract for fill_blanks fallback.
    Tries psm=8 first; if it returns empty, falls through psm=7 and psm=13.
    Each PSM is tried only when the previous returned no text.
    """
    text, conf = _tesseract_read(crop_for_ocr, config.TESSERACT_FILL_CONFIG)
    if text.strip():
        return (text.strip(), conf)

    # Fallback PSMs — only if primary returned nothing
    for psm_cfg in config.TESSERACT_FILL_FALLBACK_PSMS:
        t, c = _tesseract_read(crop_for_ocr, psm_cfg)
        if t.strip():
            return (t.strip(), c)

    return ("", conf)


# ══════════════════════════════════════════════════════════════════
#  3) TrOCR — transformer handwriting OCR
# ══════════════════════════════════════════════════════════════════

_trocr_processor = None
_trocr_model = None
_trocr_bad_words_ids: Optional[list] = None
_trocr_digit_bad_words_ids: Optional[list] = None


def _load_trocr():
    """
    Load the TrOCR processor + model lazily on first use.
    Tries config.TROCR_MODEL_NAME first (large by default for quality).
    If that fails (network down on first run, missing weights, etc.),
    falls back to config.TROCR_FALLBACK_MODEL (base).
    """
    global _trocr_processor, _trocr_model
    if _trocr_processor is not None:
        return

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    candidates = []
    primary = config.TROCR_MODEL_NAME
    candidates.append(primary)
    fallback = getattr(config, "TROCR_FALLBACK_MODEL", None)
    if fallback and fallback != primary:
        candidates.append(fallback)

    last_err: Optional[Exception] = None
    for model_name in candidates:
        try:
            print(f"[Handwriting] Loading TrOCR: {model_name} (first run downloads weights)...")
            _trocr_processor = TrOCRProcessor.from_pretrained(model_name)
            _trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
            _trocr_model.eval()
            print(f"[Handwriting] TrOCR loaded: {model_name}")
            return
        except Exception as e:
            last_err = e
            print(f"[Handwriting] WARN: failed to load {model_name}: {e}")
            _trocr_processor = None
            _trocr_model = None
            continue

    raise RuntimeError(f"Could not load any TrOCR model. Last error: {last_err}")


def _ensure_trocr_bad_words():
    """
    Compute bad_words_ids for punctuation/symbol tokens once, cached.
    Used to block hallucinated punctuation at generation time (cleaner than
    post-stripping the output text).
    """
    global _trocr_bad_words_ids
    if _trocr_bad_words_ids is not None:
        return _trocr_bad_words_ids

    if not config.TROCR_BLOCK_PUNCTUATION:
        _trocr_bad_words_ids = []
        return _trocr_bad_words_ids

    _load_trocr()
    if _trocr_processor is None:
        _trocr_bad_words_ids = []
        return _trocr_bad_words_ids

    tokenizer = _trocr_processor.tokenizer
    seen_ids = set()
    bad_ids: list = []

    # Try each blocked char in several contexts (standalone, with leading
    # space, with trailing space) to catch all token IDs the BPE produces.
    for char in config.TROCR_BLOCKED_CHARS:
        for variant in (char, " " + char, char + " "):
            try:
                ids = tokenizer.encode(variant, add_special_tokens=False)
                for tid in ids:
                    if tid not in seen_ids:
                        seen_ids.add(tid)
                        bad_ids.append([tid])
            except Exception:
                continue

    # Never block EOS / BOS / PAD — generation would deadlock
    special_ids = {
        getattr(tokenizer, attr, None)
        for attr in ("eos_token_id", "bos_token_id", "pad_token_id", "unk_token_id")
    }
    bad_ids = [b for b in bad_ids if b[0] not in special_ids]

    _trocr_bad_words_ids = bad_ids
    if config.VERBOSE:
        print(f"[Handwriting] TrOCR bad_words_ids: {len(bad_ids)} tokens blocked")
    return _trocr_bad_words_ids


def _ensure_trocr_digit_bad_words():
    """
    Compute bad_words_ids that block every non-digit token, cached.

    Used to constrain TrOCR generation to digits only — for the student-number
    strip read, where the output is known to be a sequence of digits.
    """
    global _trocr_digit_bad_words_ids
    if _trocr_digit_bad_words_ids is not None:
        return _trocr_digit_bad_words_ids

    _load_trocr()
    if _trocr_processor is None:
        _trocr_digit_bad_words_ids = []
        return _trocr_digit_bad_words_ids

    tokenizer = _trocr_processor.tokenizer
    special_ids = {
        getattr(tokenizer, attr, None)
        for attr in ("eos_token_id", "bos_token_id", "pad_token_id", "unk_token_id")
    }
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)

    bad_ids: list = []
    for tid in range(vocab_size):
        if tid in special_ids:
            continue
        try:
            text = tokenizer.decode([tid])
        except Exception:
            continue
        non_ws = "".join(c for c in text if not c.isspace())
        if non_ws and not non_ws.isdigit():
            bad_ids.append([tid])

    _trocr_digit_bad_words_ids = bad_ids
    if config.VERBOSE:
        print(f"[Handwriting] TrOCR digit-only bad_words_ids: {len(bad_ids)} tokens blocked")
    return _trocr_digit_bad_words_ids


def _read_trocr(image: np.ndarray) -> Tuple[str, float]:
    """
    Single-pass TrOCR handwriting OCR with beam search + no-repeat n-grams.
    Returns (text, confidence 0-1) — both calibrated for honest reporting.

    Beam search (num_beams=4) + no_repeat_ngram_size=2 reduce literal-repeat
    hallucinations like "the the the". On top of that, this function adds:

    1. Trailing-punctuation strip (TrOCR often appends " ." to short words —
       e.g., "treat ." for handwritten "test"). Strip these before scoring.
    2. Per-token min-probability penalty. If even one token in the generated
       sequence had very low max prob, the whole reading is likely
       hallucinated on a noisy image -> discount confidence accordingly.
    3. Calibration multiplier (config.TROCR_CONF_CALIBRATION). TrOCR is
       trained on full lines and tends to be over-confident on small
       word/char crops; a small constant discount keeps downstream cascade
       triggers honest.
    """
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
    bad_words = _ensure_trocr_bad_words() or None
    with torch.no_grad():
        outputs = _trocr_model.generate(
            pixel_values,
            max_new_tokens=config.TROCR_MAX_NEW_TOKENS,
            num_beams=config.TROCR_BEAM_SIZE,
            no_repeat_ngram_size=config.TROCR_NO_REPEAT_NGRAM,
            bad_words_ids=bad_words,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

    text = _trocr_processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()

    # ── Strip trailing hallucinated punctuation ──
    text = text.rstrip(config.TROCR_TRAILING_STRIP)

    # ── Compute confidence with token-level penalty ──
    # With beam search, scores[i] has shape (num_beams, vocab); .max() over the
    # entire tensor is the best-token prob at that step.
    conf = 0.5
    if outputs.scores:
        probs = [F.softmax(score, dim=-1).max().item() for score in outputs.scores]
        if probs:
            avg_conf = sum(probs) / len(probs)
            min_prob = min(probs)
            # Penalize if any token was very low confidence
            if min_prob < config.TROCR_MIN_TOKEN_PROB_BAD:
                conf = avg_conf * 0.50
            elif min_prob < config.TROCR_MIN_TOKEN_PROB_WARN:
                conf = avg_conf * 0.70
            else:
                conf = avg_conf

    # ── Calibrate (TrOCR over-confident on small handwritten crops) ──
    conf *= config.TROCR_CONF_CALIBRATION

    return (text, round(min(float(conf), 1.0), 4))


def _read_trocr_digit_strip(image: np.ndarray) -> Tuple[str, float]:
    """
    Single-pass TrOCR with non-digit characters blocked at generation time.

    Returns (digits_string, confidence). Used as a sequence-level cross-check
    against the per-digit CNN for student numbers — TrOCR sees the whole strip
    as one line, so it can recover from a single CNN misread by leveraging
    spacing/context cues. Output is post-filtered to digits only as defense
    in depth.
    """
    if image.size == 0 or image.shape[0] < 5 or image.shape[1] < 5:
        return ("", 0.0)
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
    bad_words = _ensure_trocr_digit_bad_words() or None

    with torch.no_grad():
        outputs = _trocr_model.generate(
            pixel_values,
            max_new_tokens=config.TROCR_MAX_NEW_TOKENS,
            num_beams=config.TROCR_BEAM_SIZE,
            no_repeat_ngram_size=config.TROCR_NO_REPEAT_NGRAM,
            bad_words_ids=bad_words,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

    text = _trocr_processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
    digits = "".join(c for c in text if c.isdigit())

    conf = 0.5
    if outputs.scores:
        probs = [F.softmax(score, dim=-1).max().item() for score in outputs.scores]
        if probs:
            avg_conf = sum(probs) / len(probs)
            min_prob = min(probs)
            if min_prob < config.TROCR_MIN_TOKEN_PROB_BAD:
                conf = avg_conf * 0.50
            elif min_prob < config.TROCR_MIN_TOKEN_PROB_WARN:
                conf = avg_conf * 0.70
            else:
                conf = avg_conf
    conf *= config.TROCR_CONF_CALIBRATION

    return (digits, round(min(float(conf), 1.0), 4))


def _read_trocr_smart(image: np.ndarray) -> Tuple[str, float]:
    """
    Cascade reader for hard handwriting:

      Pass 1: single-shot _read_trocr.
              If conf >= MULTILINE_TRIGGER_CONF -> done.
      Pass 2: split image into text lines (horizontal projection).
              For each line, try N preprocessing variants (Otsu, CLAHE-adaptive)
              and optionally their inverses; pick the best variant per line.
              Combine line texts; return averaged confidence.

    The cascade is "lazy" — high-confidence single-pass reads skip the heavy work.
    For typical clean fill_blanks (single word, dark ink) Pass 1 succeeds and
    cost is the same as before. Hard reads pay 2-4x but recover quality.

    Used by open-ended (multi-line answers) and as fallback for fill_blanks.
    """
    if preprocessing.is_blank(image):
        return ("", 1.0)

    # Pass 1
    text, conf = _read_trocr(image)
    if conf >= config.MULTILINE_TRIGGER_CONF and text.strip():
        return (text, conf)

    # Pass 2: line split + variants
    gray = preprocessing.to_gray(image)
    lines = preprocessing.split_text_lines(gray)

    line_results: list = []
    for li, line_img in enumerate(lines):
        candidates: list = []

        # If single-line case, the original pass result is already a candidate
        if len(lines) == 1:
            candidates.append((text, conf))

        variants = preprocessing.generate_ocr_variants(line_img)
        for v in variants:
            t, c = _read_trocr(v)
            candidates.append((t, c))

        # Pick best: prefer non-empty, then higher confidence, then longer text
        best = max(
            candidates,
            key=lambda r: (1 if r[0].strip() else 0, r[1], len(r[0])),
        )
        if best[0].strip():
            line_results.append(best)

    if not line_results:
        # All variants empty — return original Pass 1 result
        return (text, conf)

    combined_text = " ".join(t for t, _ in line_results).strip()
    combined_conf = sum(c for _, c in line_results) / len(line_results)
    return (combined_text, round(float(combined_conf), 4))


# ══════════════════════════════════════════════════════════════════
#  Public API — per-box readers with confidence cascade
# ══════════════════════════════════════════════════════════════════

def read_letter_box(
    aligned_image: np.ndarray,
    box: dict,
    expected_set: Optional[set] = None,
) -> ReaderResult:
    """
    Read a single handwritten letter (matching questions).

    Reader cascade — letter CNN is PRIMARY because it's specifically trained
    on EMNIST single-character data; Tesseract and TrOCR are fallbacks.

      Pass 1 (PRIMARY): EMNIST LetterCNN (26-class A-Z classifier).
                        ~3ms inference on CPU, hallucination-proof
                        (always outputs one of A-Z), trained for this task.
      Pass 2: Tesseract psm=10 + A-Z whitelist with multiple paddings.
              psm=10 is sensitive to input padding so we try (10, 8, 12, 6)
              and collect any letter that comes out.
      Pass 3: TrOCR (large) + first alphabetic char of its read.
              0.85x conf discount (line model on a single char).

    If `expected_set` is given (the union of valid letters for this matching
    question, e.g. {"A","B","C"}), every reader's output is restricted to
    that set:
      - CNN logits outside the set are masked to -inf before softmax,
      - Tesseract / TrOCR letters outside the set are dropped,
    so a 26-class problem collapses to ≤6 classes and ambiguity drops sharply.

    All candidates are pooled, highest-confidence wins. Whitelists on
    Tesseract and CNN make hallucination structurally impossible — combining
    multiple readers is safe (worst case adds compute, never adds wrong
    letters from thin air).
    """
    # Normalize expected_set to upper-case single-letter members
    allowed: Optional[set] = None
    if expected_set:
        allowed = {
            L.upper() for L in expected_set
            if isinstance(L, str) and len(L) == 1 and L.isalpha()
        }
        if not allowed:
            allowed = None

    # Blank-detection probe — use a light-inset crop for the threshold check.
    blank_probe = preprocessing.crop_for_reading(aligned_image, box, pad=10)
    if preprocessing.is_blank(blank_probe):
        return ReaderResult(text="", confidence=1.0, source="blank", needs_review=False)

    candidates: list = []  # (letter, conf, source_label)

    # Pass 1: EMNIST LetterCNN — ensemble over (0, 2, 4) using SIMPLE
    # preprocessing (no largest-connected-component selection).
    #
    # Why simple preprocessing: the aggressive variant (`crop_for_letter_cnn`)
    # picks the largest connected component of the binarized crop. On
    # multi-stroke letters where Otsu disconnects parts (e.g. a hand-drawn
    # "B" whose two bowls don't quite touch the spine), it drops real
    # strokes — and on small insets where the printed answer-box border
    # bleeds in, it picks the BORDER as the largest CC and feeds the CNN a
    # warped shape. Both regimes were producing confident "A" reads on
    # actual "B" handwriting (verified against student 0012346900 Q4).
    #
    # Why insets (0, 2, 4): per user direction — keep cropping minimal.
    # Mild edge-zero (2 px) inside `crop_for_letter_cnn_simple` handles
    # thin printed-border slivers without eating letter strokes.
    #
    # NOISE GATE: when the within-allowed pick has confidence below
    # CNN_NOISE_FLOOR, the unrestricted top-1 is almost always far outside
    # `allowed` (CNN saw e.g. "D 0.77" while allowed is {A,B}). Letting
    # that ~0.04 noise into the pool actively misleads the max-by-conf
    # selection. Drop noise reads outright.
    CNN_NOISE_FLOOR = 0.05

    # Pass 1: EMNIST LetterCNN at insets (0, 2) — minimal cropping per user
    # direction. Simple preprocessing (no largest-CC) preserves multi-stroke
    # letters that Otsu would otherwise disconnect.
    for inset in (0, 2):
        cnn_crop = preprocessing.crop_for_letter_cnn_simple(aligned_image, box, inset=inset)
        if cnn_crop.sum() <= 100:
            continue
        cnn_result = _classify_letter(cnn_crop, allowed=allowed)
        if cnn_result is None:
            continue
        letter, conf = cnn_result
        if allowed is not None and conf < CNN_NOISE_FLOOR:
            continue
        candidates.append((letter, conf, f"letter_cnn(inset={inset})"))

    # Pass 2: Tesseract psm=10 + A-Z whitelist at insets (0, 2) with a
    # generous white pad. Tesseract's layout analyzer needs enough white
    # margin to lock onto the character — too little (pad<10) and inset=0
    # crops where the printed border touches the bbox edge confuse it
    # (tested: pad=8 dropped Box 2's clean "A" from 0.96 to 0.0 conf).
    # Pad=12 keeps the letter well-isolated whether or not the printed
    # border is included.
    for inset in (0, 2):
        inner = preprocessing.crop_inside_border(aligned_image, box, inset=inset)
        if inner.size == 0:
            continue
        crop = preprocessing.pad_white(inner, pad=12)
        if preprocessing.is_blank(crop):
            continue
        letter, conf = _read_letter_tesseract(crop)
        if letter and (allowed is None or letter in allowed):
            candidates.append((letter, conf, f"tesseract(inset={inset})"))

    # Pass 3: TrOCR on the RAW box image (no inside-border crop) — per
    # user direction. Calligraphic handwriting (e.g. a "B" with a bottom
    # flourish) often has strokes that extend close to the printed border;
    # `crop_inside_border` truncates those descenders. Feeding TrOCR the
    # full bounding box with a generous white margin lets the line-trained
    # transformer see the letter in full context.
    trocr_raw = preprocessing.crop_region(aligned_image, box)
    if trocr_raw.size > 0:
        trocr_input = preprocessing.pad_white(trocr_raw, pad=14)
        if not preprocessing.is_blank(trocr_input):
            trocr_text, trocr_conf = _read_trocr(trocr_input)
            letters_only = "".join(c for c in (trocr_text or "").upper() if c.isalpha())
            if allowed is not None:
                letters_only = "".join(c for c in letters_only if c in allowed)
            if letters_only:
                candidates.append((letters_only[0], trocr_conf * 0.85, "trocr_raw"))

    if not candidates:
        return ReaderResult(
            text="",
            confidence=0.3,
            source="all_failed",
            needs_review=True,
            raw_debug={"reason": "no reader produced a letter"},
        )

    # Pick highest-confidence candidate and report its own calibrated
    # confidence directly. No solo penalty, no agreement boost.
    #
    # Why direct: each reader's confidence is already honestly calibrated:
    #   - LetterCNN now uses non-renormalized softmax (the masked-softmax
    #     dishonesty has been removed), so a high CNN number means real
    #     CNN belief, not a renormalization artifact.
    #   - TrOCR carries its single-char discount (×0.85) and token-prob
    #     penalty internally, plus TROCR_CONF_CALIBRATION × 0.90.
    #   - Tesseract reports its native per-word confidence.
    # The earlier ×0.65 solo penalty unfairly buried correct lone TrOCR
    # reads (e.g. a B that CNN cannot see) into review with misleading
    # 49%-style scores. The ×1.10 boost was a small reward when readers
    # happened to agree but never load-bearing.
    best = max(candidates, key=lambda c: c[1])

    # Vote tally — the share of "real" readers (above the 0.10 noise floor)
    # that agree with the top-confidence pick. Key signal for ambiguous
    # handwriting: when the user's "B" looks "A"-like to most readers,
    # `best` may be high-conf "A" while a substantial minority votes "B".
    # We can't auto-pick the right answer in that case, but we can refuse
    # to claim certainty and route the cell to manual review.
    real_votes = [c for c in candidates if c[1] >= 0.10]
    if real_votes:
        from collections import Counter as _Counter
        tally = _Counter(c[0] for c in real_votes)
        top_letter, top_count = tally.most_common(1)[0]
        agreeing = top_count
        vote_share = top_count / len(real_votes)
        # If `best`'s letter isn't even the majority pick, prefer the
        # majority letter at its highest-conf reader. Catches the failure
        # mode where a single 0.85 reader outvotes 4 readers at 0.50.
        if top_letter != best[0] and top_count > sum(1 for c in real_votes if c[0] == best[0]):
            best = max((c for c in candidates if c[0] == top_letter), key=lambda c: c[1])
    else:
        agreeing = 1
        vote_share = 1.0

    final_conf = float(best[1])
    # Disagreement-aware review flag. With the simpler 4-reader pool
    # (2 CNN + 1-2 Tesseract + 1 TrOCR), individual reader confidences
    # naturally land lower than the old 9-reader pool — so the standard
    # `conf < HIGH_CONF_THRESHOLD` rule alone would over-flag clean
    # consensus reads. Trigger review on REAL disagreement instead:
    #   1. Vote share < 0.66 — a substantial minority disputes the pick.
    #   2. Strong dissent — any reader at conf ≥ 0.30 picked a different
    #      letter than the winner (TrOCR's "B 0.36" on the user's
    #      calligraphic B is exactly this case).
    #   3. Solo low-confidence — only one reader supports the winner AND
    #      that reader's conf is below 0.60. Catches the case where
    #      most readers were filtered as noise.
    DISAGREEMENT_THRESHOLD = 0.66
    STRONG_DISSENT_CONF = 0.30
    SOLO_LOW_CONF = 0.60

    strong_dissent = any(
        c[0] != best[0] and c[1] >= STRONG_DISSENT_CONF
        for c in candidates
    )
    best_supporters = sum(1 for c in real_votes if c[0] == best[0])
    solo_low = best_supporters <= 1 and final_conf < SOLO_LOW_CONF

    if vote_share < DISAGREEMENT_THRESHOLD:
        final_conf = min(final_conf, vote_share)
        needs_review = True
    elif strong_dissent:
        final_conf = min(final_conf, 0.65)
        needs_review = True
    elif solo_low:
        needs_review = True
    else:
        needs_review = False
    return ReaderResult(
        text=best[0],
        confidence=round(min(final_conf, 1.0), 4),
        source=best[2],
        needs_review=needs_review,
        raw_debug={
            "all_candidates": candidates,
            "agreeing_sources": agreeing,
            "vote_share": round(vote_share, 3),
        } if config.VERBOSE else {
            "agreeing_sources": agreeing,
            "vote_share": round(vote_share, 3),
        },
    )


def _fuzzy_agree(a: str, b: str) -> float:
    """Similarity ratio between two strings (0-1). Uses rapidfuzz."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    try:
        from rapidfuzz import fuzz
        return float(fuzz.ratio(a.lower().strip(), b.lower().strip())) / 100.0
    except ImportError:
        try:
            from Levenshtein import ratio
            return float(ratio(a.lower().strip(), b.lower().strip()))
        except ImportError:
            return 1.0 if a.lower().strip() == b.lower().strip() else 0.0


def read_text_box(aligned_image: np.ndarray, box: dict, expected: str = "") -> ReaderResult:
    """
    Read a handwritten word/phrase (fill_blanks questions).

    Cascade with expected-aware validation:
      1. TrOCR primary (single-pass with beam search).
         If conf >= HIGH AND (no expected OR fuzzy_match(text, expected)) -> done.
         IMPORTANT: high-confidence TrOCR misreads (e.g., "test" -> "treat" with
         91% conf) used to slip through. Now if expected is provided AND the
         TrOCR output doesn't fuzzy-match it, we ALWAYS run the fallback
         readers — TrOCR's confidence isn't trusted by itself.
      2. TrOCR smart cascade (line split + variants on hard reads).
      3. Tesseract fallback (psm=8 -> 7 -> 13).
      4. If expected is provided, the candidate that fuzzy-matches it wins
         (resilient to TrOCR misreads that look confident).
      5. Otherwise: agreement-based merge of TrOCR + Tesseract.
    """
    crop_ocr = preprocessing.crop_for_reading(aligned_image, box, pad=8)

    if preprocessing.is_blank(crop_ocr):
        return ReaderResult(text="", confidence=1.0, source="blank", needs_review=False)

    # Pass 1: single-pass TrOCR
    trocr_text, trocr_conf = _read_trocr(crop_ocr)

    # Validation: high TrOCR conf is only "trusted" when no expected is set OR
    # the reading fuzzy-matches the expected answer. This catches misreads
    # like "test" -> "treat" that beam search makes look confident.
    if trocr_conf >= config.HIGH_CONF_THRESHOLD and trocr_text.strip():
        if not expected:
            return ReaderResult(text=trocr_text, confidence=trocr_conf,
                                source="trocr", needs_review=False)
        is_match, _ = fuzzy_match(trocr_text, expected)
        if is_match:
            return ReaderResult(text=trocr_text, confidence=trocr_conf,
                                source="trocr", needs_review=False)
        # else: high conf but no fuzzy match — suspicious, force fallback below

    # Pass 2: TrOCR smart cascade — variants try to recover faint/blurry reads
    smart_text, smart_conf = _read_trocr_smart(crop_ocr)
    if smart_conf > trocr_conf and smart_text.strip():
        trocr_text, trocr_conf = smart_text, smart_conf

    if trocr_conf >= config.HIGH_CONF_THRESHOLD and trocr_text.strip():
        if not expected:
            return ReaderResult(text=trocr_text, confidence=trocr_conf,
                                source="trocr", needs_review=False)
        is_match, _ = fuzzy_match(trocr_text, expected)
        if is_match:
            return ReaderResult(text=trocr_text, confidence=trocr_conf,
                                source="trocr", needs_review=False)
        # else: still suspicious — fall through to Tesseract

    # Pass 3: Tesseract fallback (multi-PSM)
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

    # Direct logic (matches read_letter_box): highest-confidence reader wins,
    # reports its own calibrated confidence. No agreement boost, no
    # disagreement penalty. Each reader is already calibrated:
    #   - TrOCR has TROCR_CONF_CALIBRATION × token-prob penalties baked in.
    #   - Tesseract reports native per-word confidence.
    # needs_review is driven purely by the winning reader's confidence vs.
    # HIGH_CONF_THRESHOLD; the agreement value stays in raw_debug for
    # diagnostics.
    agreement = _fuzzy_agree(trocr_text, tess_text)
    picked_text, picked_conf, picked_source = (
        (trocr_text, trocr_conf, "trocr")
        if trocr_conf >= tess_conf
        else (tess_text, tess_conf, "tesseract")
    )

    needs_review = picked_conf < config.HIGH_CONF_THRESHOLD
    return ReaderResult(
        text=picked_text,
        confidence=round(float(picked_conf), 4),
        source=picked_source,
        needs_review=needs_review,
        fallback_used=True,
        raw_debug={
            "trocr": (trocr_text, trocr_conf),
            "tesseract": (tess_text, tess_conf),
            "agreement": agreement,
        },
    )


def read_handwriting_image(image: np.ndarray) -> ReaderResult:
    """
    Read any handwriting from a pre-cropped image (used by AI evaluation fallback
    for open-ended questions). Uses the smart cascade — line split + variants —
    because open-ended answers are typically multi-line and the rare ones are
    where TrOCR struggles most.
    """
    text, conf = _read_trocr_smart(image)
    return ReaderResult(
        text=text,
        confidence=conf,
        source="trocr",
        needs_review=conf < config.HIGH_CONF_THRESHOLD,
    )


# ── Fuzzy match helper (used by grading) ─────────────────────────

def fuzzy_match(predicted: str, expected: str, case_sensitive: bool = False) -> Tuple[bool, float]:
    """
    Fuzzy string match via rapidfuzz. Returns (is_match, similarity).

    Combines two complementary scores:
      - ratio: standard edit-distance similarity ("polymorphism" vs "polymorpism")
      - token_sort_ratio: word-order independent ("Java VM" vs "VM Java")

    Final similarity = max of the two (only if FUZZY_USE_TOKEN_SORT=True).
    We deliberately exclude `partial_ratio` to avoid over-credit for substrings
    (e.g., student writes "morph", expected "polymorphism" -> partial would say 1.0).
    """
    if case_sensitive:
        a, b = predicted.strip(), expected.strip()
    else:
        a, b = predicted.lower().strip(), expected.lower().strip()

    if not a and not b:
        return (True, 1.0)
    if not a or not b:
        return (False, 0.0)

    try:
        from rapidfuzz import fuzz
        scores = [fuzz.ratio(a, b) / 100.0]
        if config.FUZZY_USE_TOKEN_SORT:
            scores.append(fuzz.token_sort_ratio(a, b) / 100.0)
        sim = max(scores)
    except ImportError:
        # Last-resort fallback to python-Levenshtein
        try:
            from Levenshtein import ratio
            sim = float(ratio(a, b))
        except ImportError:
            sim = 1.0 if a == b else 0.0

    return (sim >= config.FUZZY_MATCH_THRESHOLD, round(float(sim), 4))
