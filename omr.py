"""
OMR — Optical Mark Recognition for MC (multiple choice) and MS (multi-select).

Key idea:
  - We crop INSIDE the printed border (inset by config.OMR_INNER_INSET) so
    the bubble outline doesn't contaminate the fill ratio.
  - With the inset, an empty bubble reads ~0-5% (clean interior).
  - A student-filled bubble reads ~80-95%.
  - The gap is unambiguous — config.OMR_FILLED_THRESHOLD (0.50) cleanly
    separates them in any realistic scan.

Thresholds live in config.py.
"""

from typing import Dict, List, Optional

import cv2
import numpy as np

import config


def get_fill_ratio(image: np.ndarray, box: Dict) -> float:
    """Dark-pixel ratio inside the bubble's interior (border excluded). 0.0–1.0."""
    inset = config.OMR_INNER_INSET
    x = int(round(box.get("x", 0))) + inset
    y = int(round(box.get("y", 0))) + inset
    w = int(round(box.get("w", 0))) - 2 * inset
    h = int(round(box.get("h", 0))) - 2 * inset

    # Clamp to image bounds, and fall back to no-inset crop if inset would
    # leave nothing (can happen on tiny degenerate boxes).
    img_h, img_w = image.shape[:2]
    if w <= 0 or h <= 0:
        x = max(0, int(round(box.get("x", 0))))
        y = max(0, int(round(box.get("y", 0))))
        w = int(round(box.get("w", 0)))
        h = int(round(box.get("h", 0)))
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    if w <= 0 or h <= 0:
        return 0.0

    region = image[y:y + h, x:x + w]
    if region.size == 0:
        return 0.0

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
    _, binary = cv2.threshold(gray, config.OMR_BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    return float(np.sum(binary > 0) / binary.size)


def evaluate_mc(
    image: np.ndarray,
    options: Dict[str, Dict],
    correct_option: Optional[str] = None,
) -> Dict:
    """
    Multiple choice (exactly one answer).
    Returns dict with selected, isBlank, isCorrect, confidence, fillRatios,
    explanation, needsReview.
    """
    if not options:
        return {"selected": None, "isBlank": True, "confidence": 0.0,
                "explanation": "No options in map."}

    ratios = {k: round(get_fill_ratio(image, box), 4) for k, box in options.items()}
    max_letter = max(ratios, key=ratios.get)
    max_ratio = ratios[max_letter]

    # No mark anywhere
    if max_ratio < config.OMR_EMPTY_BORDER:
        return {
            "selected": None,
            "isBlank": True,
            "isCorrect": None,
            "confidence": 0.95,
            "fillRatios": ratios,
            "explanation": "No answer marked.",
        }

    # Ambiguous: something dark, but not above filled threshold
    if max_ratio < config.OMR_FILLED_THRESHOLD:
        return {
            "selected": None,
            "isBlank": False,
            "isCorrect": None,
            "confidence": 0.40,
            "fillRatios": ratios,
            "explanation": f"Ambiguous mark. Highest fill: {max_letter}={max_ratio:.2f}",
            "needsReview": True,
        }

    selected = max_letter

    # Confidence via gap between top two ratios
    sorted_ratios = sorted(ratios.values(), reverse=True)
    gap = sorted_ratios[0] - sorted_ratios[1] if len(sorted_ratios) >= 2 else 0.5
    if gap > 0.25:
        confidence = 0.98
    elif gap > 0.15:
        confidence = 0.92
    elif gap > 0.08:
        confidence = 0.80
    else:
        confidence = 0.55

    is_correct = (selected == correct_option) if correct_option else None

    if correct_option:
        explanation = (
            f"Selected {selected} — Correct."
            if is_correct
            else f"Selected {selected}, correct answer is {correct_option}."
        )
    else:
        explanation = f"Selected {selected}."

    return {
        "selected": selected,
        "isBlank": False,
        "isCorrect": is_correct,
        "confidence": round(confidence, 3),
        "fillRatios": ratios,
        "explanation": explanation,
    }


def evaluate_ms(
    image: np.ndarray,
    options: Dict[str, Dict],
    correct_options: Optional[List[str]] = None,
) -> Dict:
    """
    Multi-select. Per-item detection and scoring.
    """
    if not options:
        return {"selected": [], "isBlank": True, "confidence": 0.0,
                "explanation": "No options in map."}

    ratios = {k: round(get_fill_ratio(image, box), 4) for k, box in options.items()}
    selected = [k for k, v in ratios.items() if v >= config.OMR_FILLED_THRESHOLD]
    is_blank = all(v < config.OMR_EMPTY_BORDER for v in ratios.values())

    if is_blank:
        return {
            "selected": [],
            "isBlank": True,
            "isCorrect": None,
            "confidence": 0.95,
            "fillRatios": ratios,
            "explanation": "No answers marked.",
        }

    # Confidence: gap between min-selected and max-unselected
    selected_ratios = [ratios[k] for k in selected] if selected else [0.0]
    unselected_ratios = [ratios[k] for k in ratios if k not in selected] if selected else list(ratios.values())
    min_sel = min(selected_ratios) if selected_ratios else 0.0
    max_unsel = max(unselected_ratios) if unselected_ratios else 0.0
    gap = min_sel - max_unsel
    if gap > 0.20:
        confidence = 0.95
    elif gap > 0.10:
        confidence = 0.85
    elif gap > 0.05:
        confidence = 0.70
    else:
        confidence = 0.50

    is_correct = (set(selected) == set(correct_options)) if correct_options else None

    item_results: Dict[str, Dict] = {}
    correct_count = 0
    wrong_count = 0

    if correct_options:
        for letter in sorted(ratios.keys()):
            should_select = letter in correct_options
            was_selected = letter in selected
            is_item_correct = was_selected == should_select
            if is_item_correct:
                correct_count += 1
            else:
                wrong_count += 1
            item_results[letter] = {
                "selected": was_selected,
                "shouldBeSelected": should_select,
                "isCorrect": is_item_correct,
            }

    if correct_options:
        sel_str = ", ".join(selected) if selected else "none"
        cor_str = ", ".join(correct_options)
        if is_correct:
            explanation = f"Selected [{sel_str}] — All correct."
        else:
            wrong_items = [k for k, v in item_results.items() if not v["isCorrect"]]
            explanation = (
                f"Selected [{sel_str}], correct is [{cor_str}]. "
                f"{wrong_count} mistake(s): {', '.join(wrong_items)}."
            )
    else:
        explanation = f"Selected [{', '.join(selected)}]."

    return {
        "selected": selected,
        "isBlank": False,
        "isCorrect": is_correct,
        "confidence": round(confidence, 3),
        "fillRatios": ratios,
        "itemResults": item_results,
        "correctCount": correct_count,
        "wrongCount": wrong_count,
        "explanation": explanation,
    }
