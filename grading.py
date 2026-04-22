"""
Grading — pure-math scoring rules per question type.

No I/O, no model calls. Takes reader outputs + expected answers from the map
and returns a QuestionResult dict ready for the frontend + Excel.

Scoring rules:
  MC     : full points if correct; -penalty if wrong; 0 if blank.
  MS     : points split across correct options; +share per correct, -penalty per wrong.
  Match  : points split across items; +share per correct (fuzzy), -penalty per wrong.
  Fill   : same as match (fuzzy on expected text).
  Open   : delegates to ai_evaluation result — usually pending_review.
"""

from typing import Any, Dict, Optional

import config
from handwriting import fuzzy_match


# ── MC ────────────────────────────────────────────────────────────

def score_mc(omr_result: Dict, scoring: Dict, expected: Dict) -> Dict[str, Any]:
    """Score a multiple-choice question."""
    points = float(scoring.get("points", 0) or 0)
    penalty = float(scoring.get("penaltyPerItem", 0) or 0)

    if omr_result.get("isBlank"):
        return {
            "score": 0.0,
            "maxPoints": points,
            "status": "blank",
            "explanation": "Blank — no answer marked. Score: 0.",
            "confidence": omr_result.get("confidence", 0.9),
            "needsReview": False,
        }

    selected = omr_result.get("selected")
    correct = expected.get("correctOption")

    if selected is None:
        return {
            "score": 0.0,
            "maxPoints": points,
            "status": "pending_review",
            "explanation": omr_result.get("explanation", "Ambiguous mark."),
            "confidence": omr_result.get("confidence", 0.3),
            "needsReview": True,
        }

    if selected == correct:
        return {
            "score": points,
            "maxPoints": points,
            "status": "correct",
            "explanation": f"Correct ({selected}). Score: {points}/{points}.",
            "confidence": omr_result.get("confidence", 0.95),
            "needsReview": False,
        }

    score = max(0.0, -penalty)
    expl = f"Wrong — chose {selected}, answer is {correct}."
    if penalty > 0:
        expl += f" Penalty: -{penalty}."
    expl += f" Score: {score}/{points}."
    return {
        "score": round(score, 2),
        "maxPoints": points,
        "status": "wrong",
        "explanation": expl,
        "confidence": omr_result.get("confidence", 0.9),
        "needsReview": False,
    }


# ── MS ────────────────────────────────────────────────────────────

def score_ms(omr_result: Dict, scoring: Dict, expected: Dict) -> Dict[str, Any]:
    """Score a multi-select question — per correct answer basis."""
    points = float(scoring.get("points", 0) or 0)
    penalty = float(scoring.get("penaltyPerItem", 0) or 0)
    correct_options = expected.get("correctOptions") or []

    if omr_result.get("isBlank"):
        return {
            "score": 0.0,
            "maxPoints": points,
            "status": "blank",
            "explanation": "Blank — no answers marked. Score: 0.",
            "confidence": omr_result.get("confidence", 0.9),
            "needsReview": False,
        }

    num_correct = len(correct_options) if correct_options else 1
    points_per_correct = points / num_correct

    selected = set(omr_result.get("selected", []))
    expected_set = set(correct_options)

    correct_selections = selected & expected_set
    wrong_selections = selected - expected_set
    missed = expected_set - selected

    score = len(correct_selections) * points_per_correct - len(wrong_selections) * penalty
    score = max(0.0, round(score, 2))

    parts = []
    if correct_selections:
        parts.append(f"Correct: {', '.join(sorted(correct_selections))}")
    if wrong_selections:
        parts.append(f"Wrong: {', '.join(sorted(wrong_selections))}")
    if missed:
        parts.append(f"Missed: {', '.join(sorted(missed))}")

    status = (
        "correct" if not wrong_selections and not missed
        else "partial" if correct_selections
        else "wrong"
    )
    explanation = (
        f"Selected [{', '.join(sorted(selected))}], "
        f"answer [{', '.join(sorted(expected_set))}]. "
        + ". ".join(parts)
        + f". Score: {score}/{points}."
    )

    return {
        "score": score,
        "maxPoints": points,
        "status": status,
        "explanation": explanation,
        "confidence": omr_result.get("confidence", 0.9),
        "needsReview": False,
        "correctSelections": sorted(correct_selections),
        "wrongSelections": sorted(wrong_selections),
        "missed": sorted(missed),
    }


# ── Matching / Fill (shared logic) ────────────────────────────────

def _score_answer_boxes(
    ocr_answers: Dict[str, str],
    expected_map: Dict[str, str],
    scoring: Dict,
    confidences: Dict[str, float],
    needs_review_flags: Dict[str, bool],
    label: str,  # "matching" | "fill_blanks"
) -> Dict[str, Any]:
    """Generic scorer for matching & fill_blanks."""
    points = float(scoring.get("points", 0) or 0)
    penalty = float(scoring.get("penaltyPerItem", 0) or 0)
    item_count = len(expected_map) or 1
    points_per_item = points / item_count

    score = 0.0
    correct_count = 0
    wrong_count = 0
    blank_count = 0
    items: Dict[str, Dict] = {}
    explanations = []
    any_needs_review = False

    for idx in sorted(expected_map.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
        expected_val = expected_map[idx]
        predicted = (ocr_answers.get(idx, "") or "").strip()
        conf = confidences.get(idx, 0.0)
        flag = needs_review_flags.get(idx, False)
        if flag:
            any_needs_review = True

        if not predicted:
            blank_count += 1
            items[idx] = {
                "predicted": "", "expected": expected_val,
                "status": "blank", "confidence": conf,
            }
            explanations.append(f"#{idx}: blank (expected '{expected_val}')")
            continue

        is_match, sim = fuzzy_match(predicted, expected_val)
        if is_match:
            correct_count += 1
            score += points_per_item
            items[idx] = {
                "predicted": predicted, "expected": expected_val,
                "status": "correct", "similarity": sim, "confidence": conf,
            }
            explanations.append(f"#{idx}: '{predicted}' = '{expected_val}' [OK]")
        else:
            wrong_count += 1
            score -= penalty
            items[idx] = {
                "predicted": predicted, "expected": expected_val,
                "status": "wrong", "similarity": sim, "confidence": conf,
            }
            explanations.append(f"#{idx}: '{predicted}' != '{expected_val}' [X]")

    score = max(0.0, round(score, 2))
    total = correct_count + wrong_count + blank_count
    if wrong_count == 0 and blank_count == 0:
        status = "correct"
    elif blank_count == total:
        status = "blank"
    elif correct_count > 0:
        status = "partial"
    else:
        status = "wrong"

    explanation = (
        f"{correct_count}/{total} correct. "
        + " | ".join(explanations)
        + f" Score: {score}/{points}."
    )

    min_conf = min(confidences.values()) if confidences else 0.0
    needs_review = any_needs_review or min_conf < config.HIGH_CONF_THRESHOLD

    return {
        "score": score,
        "maxPoints": points,
        "status": status,
        "explanation": explanation,
        "confidence": round(min_conf, 4),
        "needsReview": needs_review,
        "items": items,
        "_label": label,
    }


def score_match(
    ocr_answers: Dict[str, str],
    expected: Dict,
    scoring: Dict,
    confidences: Dict[str, float],
    needs_review_flags: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """Score a matching question (letter-per-box)."""
    return _score_answer_boxes(
        ocr_answers,
        expected.get("correctMatches", {}) or {},
        scoring,
        confidences,
        needs_review_flags or {},
        label="matching",
    )


def score_fill(
    ocr_answers: Dict[str, str],
    expected: Dict,
    scoring: Dict,
    confidences: Dict[str, float],
    needs_review_flags: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """Score a fill-in-blank question (word/phrase-per-blank)."""
    return _score_answer_boxes(
        ocr_answers,
        expected.get("correctBlanks", {}) or {},
        scoring,
        confidences,
        needs_review_flags or {},
        label="fill_blanks",
    )


# ── Open-ended ────────────────────────────────────────────────────

def score_open(scoring: Dict, ai_result: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Wrap an ai_evaluation result (or lack thereof) into the standard shape.
    """
    points = float(scoring.get("points", 0) or 0)

    if not ai_result:
        return {
            "score": 0.0,
            "maxPoints": points,
            "status": "pending_review",
            "confidence": 0.0,
            "explanation": "Open-ended — awaiting manual review.",
            "needsReview": True,
        }

    return {
        "score": float(ai_result.get("score", 0.0)),
        "maxPoints": points,
        "status": ai_result.get("status", "pending_review"),
        "confidence": float(ai_result.get("confidence", 0.0)),
        "explanation": ai_result.get("explanation", ""),
        "needsReview": bool(ai_result.get("needsReview", True)),
        "studentReadText": ai_result.get("studentReadText", ""),
    }
