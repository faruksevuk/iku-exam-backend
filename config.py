"""
Central config — all tunable constants in one place.

Grouped by pipeline stage. Change values here; modules read them.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_BASE_DIR, "output")
MODELS_DIR = os.path.join(_BASE_DIR, "models")
DIGIT_CNN_PATH = os.path.join(MODELS_DIR, "digit_cnn.pt")
LETTER_CNN_PATH = os.path.join(MODELS_DIR, "letter_cnn.pt")

# ── Alignment (scanner) ────────────────────────────────────────────
SCAN_DPI = 200
PAGE_WIDTH_PX = 756           # 200mm @ 96dpi (matches generator render)
PAGE_HEIGHT_PX = 1086          # 287mm @ 96dpi
BULLSEYE_CORNER_RATIO = 0.07   # per-corner search region (7% of page side, fallback path)

# ROI-based anchor search (precise, runs first per corner)
ANCHOR_ROI_RADIUS_RATIO = 0.05      # search radius as fraction of min(img_w, img_h)
ANCHOR_ROI_RADIUS_MIN = 40
ANCHOR_ROI_RADIUS_MAX = 200

# Three-tier transform thresholds
HOMOGRAPHY_RANSAC_THRESHOLD = 3.0   # RANSAC reprojection threshold (px)
# 4 anchors -> RANSAC homography
# 2-3 anchors -> affine median offset
# <2 anchors -> pure resize (no drift correction; warning logged)

# ── OMR (MC / MS bubbles) ──────────────────────────────────────────
# We measure fill ratio INSIDE the printed border (inset by OMR_INNER_INSET
# pixels), so the bubble's own outline doesn't dominate the count. With the
# inset, an empty bubble reads ~0-5% (clean interior) and a filled bubble
# reads ~80-95% — there is a huge, unambiguous gap in between.
OMR_INNER_INSET = 3            # px inset INSIDE the box border before measuring fill
OMR_BINARY_THRESHOLD = 140     # cv2 threshold for "dark" pixel
OMR_BLANK_THRESHOLD = 0.10     # below = no mark at all
OMR_EMPTY_BORDER = 0.40        # below = empty (no mark) — only a faint stray dot
OMR_FILLED_THRESHOLD = 0.50    # above = student filled this bubble

# ── Preprocessing ──────────────────────────────────────────────────
BLANK_DARK_RATIO = 0.01        # <1% dark pixels = region is blank
CROP_EXPAND_PX = 3             # expand box by this many px to catch overflow
BORDER_INSET_PX = 4            # crop this far inside the box border
WEBP_QUALITY = 80
JPEG_QUALITY = 85              # backward compat for review UI

# ── Handwriting cascade ────────────────────────────────────────────
HIGH_CONF_THRESHOLD = 0.80     # above = accept as-is
LOW_CONF_THRESHOLD = 0.50      # below = force manual review
FUZZY_AGREE_THRESHOLD = 0.90   # primary + fallback agree if sim >= this
FUZZY_MATCH_THRESHOLD = 0.80   # fuzzy match vs. expected answer

# ── Tesseract ──────────────────────────────────────────────────────
# Windows default install path; pytesseract uses this if PATH unset.
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Matching: single-character mode + A-Z whitelist + English language hint.
# `-l eng --oem 3` makes the LSTM engine use English language priors even on
# a single character — slightly biases ambiguous reads toward letters that
# commonly appear in English handwriting samples.
TESSERACT_MATCHING_CONFIG = (
    "-l eng --psm 10 --oem 3 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)

# Fill_blanks (primary): single-word mode + English + alphanumeric whitelist.
# IMPORTANT: dict-word penalty is set to 0.0 (disabled). Tesseract's default
# behavior is to bias outputs toward valid English dictionary words, but for
# exam grading that's a HAZARD — it would auto-correct a student's
# "congradulations" misspelling to the dict word "congratulations", causing
# the student to get full credit despite an actual error. Setting the
# penalty to 0 preserves whatever the student actually wrote, so fuzzy_match
# in grading.py can correctly award partial credit for typos.
TESSERACT_FILL_CONFIG = (
    "-l eng --psm 8 --oem 3 "
    "-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    "-c language_model_penalty_non_dict_word=0.0"
)

# Fill_blanks fallback PSMs (tried only when primary returns empty):
# psm 7 = single line, psm 13 = raw line / no layout assumptions.
_FILL_FALLBACK_BASE = (
    "-l eng --oem 3 "
    "-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    "-c language_model_penalty_non_dict_word=0.0"
)
TESSERACT_FILL_FALLBACK_PSMS = (
    f"--psm 7 {_FILL_FALLBACK_BASE}",
    f"--psm 13 {_FILL_FALLBACK_BASE}",
)

# ── TrOCR model + generation ──────────────────────────────────────
# Primary: microsoft/trocr-base-handwritten (~300MB, ~334M params).
# Fallback: microsoft/trocr-large-handwritten (~1.3GB, ~558M params).
#
# Why base as primary: our workload is dominated by short reads —
# single-letter matching cells, single/short-word fill_blanks, and
# digit-only student-number strips. Quality gap between base and large
# on these short inputs is ~1-3% accuracy; on multi-line cursive it
# would be larger but we have very little of that. CPU inference for
# base is ~2x faster, which matters because every matching cell and
# every fill_blank pays this cost. Large stays as a safety net.
TROCR_MODEL_NAME = "microsoft/trocr-base-handwritten"
TROCR_FALLBACK_MODEL = "microsoft/trocr-large-handwritten"

TROCR_BEAM_SIZE = 4               # beam search instead of greedy
TROCR_NO_REPEAT_NGRAM = 2         # prevents "the the the" hallucinations
TROCR_MAX_NEW_TOKENS = 64

# TrOCR is line-trained; on small word/char crops it tends to be over-confident.
# Multiply the raw average-prob confidence by this factor before reporting,
# so downstream cascades trigger fallback at a realistic threshold.
TROCR_CONF_CALIBRATION = 0.90

# When TrOCR sequence-internal min token probability is very low, the whole
# reading is likely a hallucination on a noisy image. Penalize confidence:
TROCR_MIN_TOKEN_PROB_WARN = 0.40   # if min < this -> conf *= 0.70
TROCR_MIN_TOKEN_PROB_BAD = 0.20    # if min < this -> conf *= 0.50

# Trailing characters that TrOCR commonly hallucinates on isolated words
# (e.g., "treat ." instead of "test"). Strip from final output.
TROCR_TRAILING_STRIP = " .,;:!?_-'\"`*"

# Block these characters from appearing in TrOCR output via bad_words_ids.
# Cleaner than post-stripping — prevents hallucination at generation time.
# We keep alphanumeric chars; everything punctuation/symbol-ish gets blocked.
# NOTE: "_" and "-" are deliberately excluded — in BPE/SentencePiece-style
# tokenizers, the underscore-prefix marker (▁) is special and can collide
# with letter-prefix tokens. Hyphens appear in compound words.
TROCR_BLOCKED_CHARS = ".,;:!?'\"()[]{}*&^%$#@<>|\\/=+~`"
TROCR_BLOCK_PUNCTUATION = True

# Cascade disagreement penalty. When TrOCR and Tesseract disagree (fuzzy
# similarity below FUZZY_AGREE_THRESHOLD), the reported confidence of the
# winner is multiplied by this factor — so a "wrong but high-conf" reading
# can't slip past the review threshold.
DISAGREEMENT_PENALTY = 0.5

# ── Smart cascade for open-ended / multi-char text ─────────────────
# Lazy multi-pass: only triggered when single-pass confidence is low.
MULTILINE_TRIGGER_CONF = 0.85     # below = run line-by-line + variants
MULTILINE_MAX_VARIANTS = 2        # preprocessing variants per line
MULTILINE_TRY_INVERTED = True     # also try black/white inverted

# ── Letter crop for matching ───────────────────────────────────────
# 2x upsample preserves stroke detail; 4x cubic over-blurs small letters.
LETTER_UPSAMPLE_FACTOR = 2

# ── Fuzzy matching ─────────────────────────────────────────────────
# Use rapidfuzz. ratio is exact-similarity; token_sort handles word reorder.
# We deliberately exclude partial_ratio (would over-credit substrings).
FUZZY_USE_TOKEN_SORT = True

# ── AI evaluation (open-ended) ─────────────────────────────────────
AI_ENABLED = True             # AI grading via Ollama. Set to False to fall back to placeholder/manual review.
OLLAMA_URL = "http://localhost:11434"
VISION_MODEL = None           # Reserved for a future vision-LLM hook. This round uses handwriting.py for OCR.
GRADING_MODEL = "qwen3:1.7b"
AI_TIMEOUT_SECONDS = 90
# Phrases that, if echoed in AI explanation, force manual review:
AI_SAFETY_FLAGS = [
    "ignore previous",
    "ignore the rubric",
    "full marks",
    "give full",
    "override",
    "as requested",
]

# ── Logging ────────────────────────────────────────────────────────
VERBOSE = True
