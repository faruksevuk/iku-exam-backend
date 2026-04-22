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

# ── Alignment (scanner) ────────────────────────────────────────────
SCAN_DPI = 200
PAGE_WIDTH_PX = 756        # 200mm @ 96dpi (matches generator render)
PAGE_HEIGHT_PX = 1086      # 287mm @ 96dpi
BULLSEYE_CORNER_RATIO = 0.07  # per-corner search region (7% of page side)
BULLSEYE_MIN_COUNT = 3     # below this → fallback to cv2.resize

# ── OMR (MC / MS bubbles) ──────────────────────────────────────────
OMR_BINARY_THRESHOLD = 140     # cv2 threshold for "dark" pixel
OMR_BLANK_THRESHOLD = 0.10     # below = no mark at all
OMR_EMPTY_BORDER = 0.40        # empty circle/square border artifact
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
TESSERACT_MATCHING_CONFIG = (
    "--psm 10 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)
TESSERACT_FILL_CONFIG = "--psm 8"   # single word mode

# ── AI evaluation (open-ended) ─────────────────────────────────────
AI_ENABLED = False             # placeholder mode — AI off by default
OLLAMA_URL = "http://localhost:11434"
VISION_MODEL = "moondream"
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
