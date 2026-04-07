"""
Scoring Engine — Computes scores with detailed explanations.

MC: full points if correct, -penalty if wrong, 0 if blank
MS: points divided by number of CORRECT options. +share per correct, -penalty per wrong. 0 if blank.
Match/Fill: points divided by number of items. +share per correct, -penalty per wrong. 0 if blank.
Open: stub (AI later), 0 for now.
"""

from typing import Dict, Any, Optional


def score_mc(result: Dict, scoring: Dict, expected: Dict) -> Dict[str, Any]:
    """Score MC question."""
    points = scoring.get("points", 0)
    penalty = scoring.get("penaltyPerItem", 0) or 0

    if result.get("isBlank"):
        return {"score": 0, "maxPoints": points, "status": "blank",
                "explanation": "Blank — no answer marked. Score: 0."}

    selected = result.get("selected")
    correct = expected.get("correctOption")

    if selected == correct:
        return {"score": points, "maxPoints": points, "status": "correct",
                "explanation": f"Correct ({selected}). Score: {points}/{points}."}
    else:
        score = max(0, -penalty)
        expl = f"Wrong — chose {selected}, answer is {correct}."
        if penalty > 0:
            expl += f" Penalty: -{penalty}."
        expl += f" Score: {score}/{points}."
        return {"score": round(score, 2), "maxPoints": points, "status": "wrong",
                "explanation": expl}


def score_ms(result: Dict, scoring: Dict, expected: Dict) -> Dict[str, Any]:
    """
    Score MS question — per correct answer basis.
    Points are divided by the number of CORRECT options.
    Student gets +share for each correctly identified, -penalty for each wrong selection.
    """
    points = scoring.get("points", 0)
    penalty = scoring.get("penaltyPerItem", 0) or 0
    correct_options = expected.get("correctOptions", [])

    if result.get("isBlank"):
        return {"score": 0, "maxPoints": points, "status": "blank",
                "explanation": "Blank — no answers marked. Score: 0."}

    num_correct_expected = len(correct_options) if correct_options else 1
    points_per_correct = points / num_correct_expected

    selected = set(result.get("selected", []))
    expected_set = set(correct_options)

    correct_selections = selected & expected_set       # correctly selected
    wrong_selections = selected - expected_set          # selected but shouldn't be
    missed = expected_set - selected                    # should have selected but didn't

    score = len(correct_selections) * points_per_correct
    score -= len(wrong_selections) * penalty
    score = max(0, round(score, 2))

    parts = []
    if correct_selections:
        parts.append(f"Correct: {', '.join(sorted(correct_selections))}")
    if wrong_selections:
        parts.append(f"Wrong: {', '.join(sorted(wrong_selections))}")
    if missed:
        parts.append(f"Missed: {', '.join(sorted(missed))}")

    status = "correct" if not wrong_selections and not missed else "partial" if correct_selections else "wrong"
    explanation = f"Selected [{', '.join(sorted(selected))}], answer [{', '.join(sorted(expected_set))}]. "
    explanation += ". ".join(parts) + "."
    explanation += f" Score: {score}/{points}."

    return {
        "score": score,
        "maxPoints": points,
        "status": status,
        "correctSelections": sorted(correct_selections),
        "wrongSelections": sorted(wrong_selections),
        "missed": sorted(missed),
        "explanation": explanation,
    }


def score_match(
    ocr_answers: Dict[str, str],
    expected: Dict,
    scoring: Dict,
    confidences: Dict[str, float],
) -> Dict[str, Any]:
    """Score matching with fuzzy comparison and explanations."""
    from ocr import fuzzy_match

    points = scoring.get("points", 0)
    penalty = scoring.get("penaltyPerItem", 0) or 0
    correct_matches = expected.get("correctMatches", {})
    item_count = len(correct_matches) or 1
    points_per_item = points / item_count

    score = 0
    correct_count = 0
    wrong_count = 0
    blank_count = 0
    items = {}
    explanations = []

    for idx in sorted(correct_matches.keys(), key=lambda x: int(x)):
        expected_val = correct_matches[idx]
        predicted = ocr_answers.get(idx, "").strip()
        conf = confidences.get(idx, 0.0)

        if not predicted:
            blank_count += 1
            items[idx] = {"predicted": "", "expected": expected_val, "status": "blank", "confidence": conf}
            explanations.append(f"#{idx}: blank (expected '{expected_val}')")
            continue

        is_match, sim = fuzzy_match(predicted, expected_val)
        if is_match:
            correct_count += 1
            score += points_per_item
            items[idx] = {"predicted": predicted, "expected": expected_val, "status": "correct", "similarity": sim, "confidence": conf}
            explanations.append(f"#{idx}: '{predicted}' = '{expected_val}' [OK]")
        else:
            wrong_count += 1
            score -= penalty
            items[idx] = {"predicted": predicted, "expected": expected_val, "status": "wrong", "similarity": sim, "confidence": conf}
            explanations.append(f"#{idx}: '{predicted}' != '{expected_val}' [X]")

    score = max(0, round(score, 2))
    total = correct_count + wrong_count + blank_count
    status = "correct" if wrong_count == 0 and blank_count == 0 else "partial" if correct_count > 0 else "blank" if blank_count == total else "wrong"

    explanation = f"{correct_count}/{total} correct. " + " | ".join(explanations) + f" Score: {score}/{points}."

    return {"score": score, "maxPoints": points, "status": status, "items": items, "explanation": explanation,
            "confidence": min(confidences.values()) if confidences else 0.0}


def score_fill(
    ocr_answers: Dict[str, str],
    expected: Dict,
    scoring: Dict,
    confidences: Dict[str, float],
) -> Dict[str, Any]:
    """Score fill-in-blank with fuzzy comparison."""
    from ocr import fuzzy_match

    points = scoring.get("points", 0)
    penalty = scoring.get("penaltyPerItem", 0) or 0
    correct_blanks = expected.get("correctBlanks", {})
    item_count = len(correct_blanks) or 1
    points_per_item = points / item_count

    score = 0
    correct_count = 0
    wrong_count = 0
    blank_count = 0
    items = {}
    explanations = []

    for idx in sorted(correct_blanks.keys(), key=lambda x: int(x)):
        expected_val = correct_blanks[idx]
        predicted = ocr_answers.get(idx, "").strip()
        conf = confidences.get(idx, 0.0)

        if not predicted:
            blank_count += 1
            items[idx] = {"predicted": "", "expected": expected_val, "status": "blank", "confidence": conf}
            explanations.append(f"Blank #{idx}")
            continue

        is_match, sim = fuzzy_match(predicted, expected_val)
        if is_match:
            correct_count += 1
            score += points_per_item
            items[idx] = {"predicted": predicted, "expected": expected_val, "status": "correct", "similarity": sim, "confidence": conf}
            explanations.append(f"#{idx}: '{predicted}' [OK]")
        else:
            wrong_count += 1
            score -= penalty
            items[idx] = {"predicted": predicted, "expected": expected_val, "status": "wrong", "similarity": sim, "confidence": conf}
            explanations.append(f"#{idx}: '{predicted}' != '{expected_val}'")

    score = max(0, round(score, 2))
    total = correct_count + wrong_count + blank_count
    status = "correct" if wrong_count == 0 and blank_count == 0 else "partial" if correct_count > 0 else "blank" if blank_count == total else "wrong"

    explanation = f"{correct_count}/{total} correct. " + " | ".join(explanations) + f" Score: {score}/{points}."

    return {"score": score, "maxPoints": points, "status": status, "items": items, "explanation": explanation,
            "confidence": min(confidences.values()) if confidences else 0.0}


def score_open(scoring: Dict) -> Dict[str, Any]:
    """Stub for open-ended — AI evaluation later."""
    points = scoring.get("points", 0)
    return {
        "score": 0, "maxPoints": points, "status": "pending_ai",
        "needsReview": True, "confidence": 0.0,
        "explanation": "Open-ended — awaiting AI evaluation.",
    }
