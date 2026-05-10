"""Generate Figure 4 for Weekly Report v3 — new matching cascade.

Three panels showing what each reader sees on the same hand-printed B:
  - CNN inset=0 (simple preprocessing, no largest-CC)
  - CNN inset=2 (simple preprocessing)
  - TrOCR input: raw bbox + white pad (no inside-border crop)
"""
import sys
sys.path.insert(0, "D:/repos/exam-backend")

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

import preprocessing
import handwriting
from handwriting import _classify_letter

ALIGNED = "D:/repos/exam-backend/output/0012346900_page1.jpg"
MAP = "C:/Users/faruk/AppData/Roaming/iku-exam-generator/exams/morj6pbmpfepg.map.json"
OUT = "C:/Users/faruk/Downloads/_report_assets_v3/figure4_lettercnn_insets.png"

img = cv2.imread(ALIGNED)
m = json.load(open(MAP, "r", encoding="utf-8"))

# Q4 box 1 = the calligraphic B that started this whole investigation.
q4 = None
for page in m.get("pages", []):
    if "4" in page.get("questions", {}):
        q4 = page["questions"]["4"]
        break
box = q4["answerBoxes"]["1"]
allowed = {"A", "B"}

# Panel 1: CNN simple, inset=0
cnn0 = preprocessing.crop_for_letter_cnn_simple(img, box, inset=0)
r0 = _classify_letter(cnn0, allowed=allowed)

# Panel 2: CNN simple, inset=2
cnn2 = preprocessing.crop_for_letter_cnn_simple(img, box, inset=2)
r2 = _classify_letter(cnn2, allowed=allowed)

# Panel 3: TrOCR input — raw bbox + pad_white(14)
raw = preprocessing.crop_region(img, box)
trocr_input = preprocessing.pad_white(raw, pad=14)
trocr_text, trocr_conf = handwriting._read_trocr(trocr_input)

# Convert TrOCR input (color) to grayscale for visual consistency, but keep
# the actual aspect ratio so it reads as the "real" pad+bbox the network sees.
trocr_gray = cv2.cvtColor(trocr_input, cv2.COLOR_BGR2GRAY) if trocr_input.ndim == 3 else trocr_input

fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
fig.suptitle("Letter reader cascade — what each engine sees on a hand-printed B",
             fontsize=13, fontweight="bold", y=0.99)

DISPLAY = 200  # px square — equal panel size for visual parity

# Letter 1 (CNN) — the 28x28 trained input, scaled up for readability.
axes[0].imshow(cv2.resize(cnn0, (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST), cmap="gray")
axes[0].set_title(f"CNN inset = 0 (simple)\nreads: {r0[0]} ({int(r0[1] * 100)}%)",
                  fontsize=11)
axes[0].axis("off")

axes[1].imshow(cv2.resize(cnn2, (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST), cmap="gray")
axes[1].set_title(f"CNN inset = 2 (simple)\nreads: {r2[0]} ({int(r2[1] * 100)}%)",
                  fontsize=11)
axes[1].axis("off")

# TrOCR panel: pad bbox to a square so it occupies the same on-page area
# as the CNN panels — visual parity matters more than native aspect here.
def _square_pad(img, side):
    h, w = img.shape[:2]
    scale = side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.full((side, side), 255, dtype=np.uint8) if resized.ndim == 2 \
          else np.full((side, side, 3), 255, dtype=np.uint8)
    y0 = (side - new_h) // 2
    x0 = (side - new_w) // 2
    if resized.ndim == 2:
        out[y0:y0+new_h, x0:x0+new_w] = resized
    else:
        out[y0:y0+new_h, x0:x0+new_w] = resized
    return out

axes[2].imshow(_square_pad(trocr_gray, DISPLAY), cmap="gray", vmin=0, vmax=255)
trocr_letters = "".join(c for c in (trocr_text or "").upper() if c.isalpha() and c in allowed)
trocr_letter = trocr_letters[0] if trocr_letters else "—"
# Show production confidence (after the 0.85 single-char penalty applied
# in read_letter_box when TrOCR's letter joins the candidate pool).
production_conf = trocr_conf * 0.85
axes[2].set_title(f"TrOCR — raw bbox + pad\nreads: {trocr_letter} "
                  f"({int(production_conf * 100)}%)",
                  fontsize=11)
axes[2].axis("off")

plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
print(f"  CNN inset=0: {r0}")
print(f"  CNN inset=2: {r2}")
print(f"  TrOCR raw:   {trocr_text!r} conf={trocr_conf:.4f}")
