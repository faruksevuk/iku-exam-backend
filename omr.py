"""
OMR Engine — Optical Mark Recognition for MC and MS questions.

Key insight: empty MC circles/MS squares have ~30-35% dark pixels from their border.
A filled bubble has 55%+. Blank (no mark at all) would be below 10%.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional

# Thresholds calibrated for 12x12 bubble images where empty circle border ≈ 30-35%
BLANK_THRESHOLD = 0.10     # below = definitely no mark
EMPTY_BORDER = 0.40        # empty circle/square border artifact
FILLED_THRESHOLD = 0.50    # above = student filled this bubble


def get_fill_ratio(image: np.ndarray, box: Dict) -> float:
    """Calculate dark pixel ratio in a bubble region."""
    x = max(0, int(round(box.get("x", 0))))
    y = max(0, int(round(box.get("y", 0))))
    w = int(round(box.get("w", 0)))
    h = int(round(box.get("h", 0)))
    if w <= 0 or h <= 0:
        return 0.0
    x = min(x, image.shape[1] - 1)
    y = min(y, image.shape[0] - 1)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    region = image[y:y+h, x:x+w]
    if region.size == 0:
        return 0.0
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
    _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    return float(np.sum(binary > 0) / binary.size)


def evaluate_mc(
    image: np.ndarray,
    options: Dict[str, Dict],
    correct_option: Optional[str] = None,
) -> Dict:
    """Evaluate multiple-choice. Returns selected, correctness, explanation."""
    if not options:
        return {"selected": None, "isBlank": True, "confidence": 0.0}

    ratios = {k: round(get_fill_ratio(image, box), 4) for k, box in options.items()}

    # Find options that are clearly filled (above empty border threshold)
    filled = {k: v for k, v in ratios.items() if v >= FILLED_THRESHOLD}
    max_letter = max(ratios, key=ratios.get)
    max_ratio = ratios[max_letter]

    # Blank: nothing above empty border level
    if max_ratio < EMPTY_BORDER:
        return {
            "selected": None,
            "isBlank": True,
            "isCorrect": None,
            "confidence": 0.95,
            "fillRatios": ratios,
            "explanation": "No answer marked.",
        }

    # Select the most filled option (must be above FILLED_THRESHOLD)
    selected = max_letter if max_ratio >= FILLED_THRESHOLD else None

    if selected is None:
        # Ambiguous — some mark but not clearly filled
        return {
            "selected": None,
            "isBlank": False,
            "isCorrect": None,
            "confidence": 0.40,
            "fillRatios": ratios,
            "explanation": f"Ambiguous mark detected. Highest fill: {max_letter}={max_ratio:.2f}",
            "needsReview": True,
        }

    # Confidence based on gap between selected and next highest
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

    is_correct = (selected == correct_option) if correct_option and selected else None

    explanation = f"Selected {selected}."
    if correct_option:
        if is_correct:
            explanation = f"Selected {selected} — Correct."
        else:
            explanation = f"Selected {selected}, correct answer is {correct_option}."

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
    """Evaluate multi-select. Per-item detection and scoring."""
    if not options:
        return {"selected": [], "isBlank": True, "confidence": 0.0}

    ratios = {k: round(get_fill_ratio(image, box), 4) for k, box in options.items()}
    selected = [k for k, v in ratios.items() if v >= FILLED_THRESHOLD]
    is_blank = all(v < EMPTY_BORDER for v in ratios.values())

    if is_blank:
        return {
            "selected": [],
            "isBlank": True,
            "isCorrect": None,
            "confidence": 0.95,
            "fillRatios": ratios,
            "explanation": "No answers marked.",
        }

    # Confidence
    selected_ratios = [ratios[k] for k in selected] if selected else [0]
    unselected_ratios = [ratios[k] for k in ratios if k not in selected] if selected else list(ratios.values())
    min_sel = min(selected_ratios) if selected_ratios else 0
    max_unsel = max(unselected_ratios) if unselected_ratios else 0
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

    # Per-item results
    item_results = {}
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

    # Explanation
    if correct_options:
        selected_str = ", ".join(selected) if selected else "none"
        correct_str = ", ".join(correct_options)
        if is_correct:
            explanation = f"Selected [{selected_str}] — All correct."
        else:
            wrong_items = [k for k, v in item_results.items() if not v["isCorrect"]]
            explanation = f"Selected [{selected_str}], correct is [{correct_str}]. {wrong_count} mistake(s): {', '.join(wrong_items)}."
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
