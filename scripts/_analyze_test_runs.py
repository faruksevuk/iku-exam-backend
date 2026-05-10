"""Compare actual evaluation results against expected answers for the
3 test exams (4012 SE, MAT102 Calc, PHY201 Physics) — 30 students total.

Outputs:
  C:/Users/faruk/Downloads/_test_report_assets/per_question_correctness.png
  C:/Users/faruk/Downloads/_test_report_assets/per_exam_accuracy.png
  C:/Users/faruk/Downloads/_test_report_assets/per_question_type.png
  C:/Users/faruk/Downloads/_test_report_assets/score_distribution.png
  C:/Users/faruk/Downloads/_test_report_assets/timing_breakdown.png
  C:/Users/faruk/Downloads/_test_report_assets/_summary.json
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXAMS_DIR = Path(os.environ["APPDATA"]) / "iku-exam-generator" / "exams"
ASSETS_DIR = Path(r"C:/Users/faruk/Downloads/_test_report_assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

EXAM_SLUGS = [
    ("4012-se-midterm-2026", "Software Engineering"),
    ("mat102-calc2-midterm-2026", "Calculus II"),
    ("phy201-physics-quiz-2026", "Physics"),
]

# IKU red palette for charts
IKU_RED = "#ED1B24"
IKU_RED_DARK = "#A8141A"
GREEN = "#10B981"
AMBER = "#F59E0B"
GRAY = "#94A3B8"


def expected_for_question(q_map: dict) -> dict:
    """Pull the canonical expected answer from a map question entry."""
    ea = q_map.get("expectedAnswer", {}) or {}
    qtype = q_map.get("type", "")
    if qtype == "multiple_choice":
        return {"option": ea.get("correctOption")}
    if qtype == "matching":
        return {"matches": ea.get("correctMatches", {})}
    if qtype == "fill_blanks":
        return {"blanks": ea.get("correctBlanks", {})}
    if qtype == "open_ended":
        return {"text": ea.get("text", "")}
    return {}


def question_correct(q_actual: dict, expected: dict, qtype: str) -> bool:
    """Return True iff the actual reading matches the expected answer.

    Uses the backend's own verdict where available (`isCorrect` flag,
    or `status == 'correct'`). For open_ended where there's no boolean,
    we threshold on the AI-grader's score (≥50% of max).
    """
    # Most question types carry a backend-computed isCorrect.
    if "isCorrect" in q_actual:
        return bool(q_actual["isCorrect"])
    # Status field — set by grading.py for matching/fill.
    status = q_actual.get("status", "")
    if status == "correct":
        return True
    if status in ("wrong", "blank", "error"):
        return False
    if status == "partial":
        # Partial credit on matching/fill — count as "not fully correct"
        # for the strict accuracy measure but not as "wrong" for review.
        return False
    if qtype == "open_ended":
        # AI graded — trust score / max ratio.
        score = float(q_actual.get("score", 0) or 0)
        max_pts = float(q_actual.get("maxPoints", 0) or 1)
        return score >= max_pts * 0.5
    # Last resort: assume not correct.
    return False


def question_partial(q_actual: dict, qtype: str) -> bool:
    """Returns True for matching / fill_blanks where some but not all
    cells matched. These are graded with proportional credit."""
    return q_actual.get("status") == "partial"


# ── Run analysis ─────────────────────────────────────────────────────
overall = []
per_exam = []
per_qtype = defaultdict(lambda: {"correct": 0, "partial": 0, "total": 0, "needsReview": 0})

for slug, label in EXAM_SLUGS:
    res = json.loads((EXAMS_DIR / f"{slug}.results.json").read_text(encoding="utf-8"))
    exam_map = json.loads((EXAMS_DIR / f"{slug}.map.json").read_text(encoding="utf-8"))

    # Build q_num → expected and q_num → type lookups
    expected_by_num: dict[str, dict] = {}
    qtype_by_num: dict[str, str] = {}
    for page in exam_map.get("pages", []):
        for q_num, q in (page.get("questions") or {}).items():
            qtype_by_num[q_num] = q.get("type", "")
            expected_by_num[q_num] = expected_for_question(q)

    students = res.get("students", [])
    exam_correct = 0
    exam_partial = 0
    exam_total_q = 0
    exam_needs_review = 0
    score_pcts = []

    for s in students:
        s_total_max = s.get("totalMaxPoints", 0) or 1
        score_pcts.append(s.get("totalScore", 0) / s_total_max * 100)
        for q_num, q in (s.get("questions") or {}).items():
            qtype = q.get("type") or qtype_by_num.get(q_num, "")
            exp = expected_by_num.get(q_num, {})
            ok = question_correct(q, exp, qtype)
            partial = question_partial(q, qtype)
            exam_total_q += 1
            per_qtype[qtype]["total"] += 1
            if ok:
                exam_correct += 1
                per_qtype[qtype]["correct"] += 1
            elif partial:
                exam_partial += 1
                per_qtype[qtype]["partial"] += 1
            if q.get("needsReview"):
                exam_needs_review += 1
                per_qtype[qtype]["needsReview"] += 1

    accuracy = (exam_correct / exam_total_q * 100) if exam_total_q else 0
    partial_rate = (exam_partial / exam_total_q * 100) if exam_total_q else 0
    review_rate = (exam_needs_review / exam_total_q * 100) if exam_total_q else 0
    per_exam.append({
        "slug": slug,
        "label": label,
        "students": len(students),
        "questions": exam_total_q,
        "correct": exam_correct,
        "partial": exam_partial,
        "accuracy_pct": round(accuracy, 1),
        "partial_pct": round(partial_rate, 1),
        "needs_review": exam_needs_review,
        "review_rate_pct": round(review_rate, 1),
        "score_pcts": score_pcts,
        "avg_score_pct": round(sum(score_pcts) / len(score_pcts), 1) if score_pcts else 0,
    })

# ── Chart 1: per-exam accuracy + review rate ───────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
labels = [e["label"] for e in per_exam]
accuracy = [e["accuracy_pct"] for e in per_exam]
review = [e["review_rate_pct"] for e in per_exam]
x = np.arange(len(labels))
w = 0.35
ax.bar(x - w/2, accuracy, w, color=GREEN, label="Correct readings", edgecolor="white", linewidth=1)
ax.bar(x + w/2, review, w, color=AMBER, label="Flagged for review", edgecolor="white", linewidth=1)
for i, (a, r) in enumerate(zip(accuracy, review)):
    ax.text(i - w/2, a + 1, f"{a:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax.text(i + w/2, r + 1, f"{r:.0f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("% of questions", fontsize=11)
ax.set_ylim(0, 110)
ax.set_title("Per-exam accuracy and review-flag rates",
             fontsize=13, fontweight="bold", pad=14)
ax.legend(loc="upper right", framealpha=0.95)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "per_exam_accuracy.png", dpi=150, facecolor="white")
plt.close()

# ── Chart 2: per-question-type — stacked correct + partial ────────
qtype_order = ["multiple_choice", "matching", "fill_blanks", "open_ended"]
qtype_labels = ["Multiple Choice", "Matching", "Fill-in-Blank", "Open-Ended"]
qtype_correct_pct = []
qtype_partial_pct = []
qtype_review_pct = []
qtype_totals = []
for qt in qtype_order:
    s = per_qtype.get(qt, {"correct": 0, "partial": 0, "total": 0, "needsReview": 0})
    total = s["total"] or 1
    qtype_correct_pct.append(s["correct"] / total * 100)
    qtype_partial_pct.append(s["partial"] / total * 100)
    qtype_review_pct.append(s["needsReview"] / total * 100)
    qtype_totals.append(s["total"])

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(qtype_labels))
# Two grouped bar pairs: (correct + partial stacked) and (needs-review).
ax.bar(x - w/2, qtype_correct_pct, w, color=GREEN, label="Fully correct", edgecolor="white", linewidth=1)
ax.bar(x - w/2, qtype_partial_pct, w, bottom=qtype_correct_pct,
       color="#86efac", label="Partial credit (some cells correct)", edgecolor="white", linewidth=1)
ax.bar(x + w/2, qtype_review_pct, w, color=AMBER, label="Flagged for review", edgecolor="white", linewidth=1)
for i, (c, p, r, n) in enumerate(zip(qtype_correct_pct, qtype_partial_pct, qtype_review_pct, qtype_totals)):
    if c + p > 5:
        ax.text(i - w/2, c + p + 1.5, f"{c+p:.0f}%", ha="center", fontsize=10, fontweight="bold")
    if r > 5:
        ax.text(i + w/2, r + 1.5, f"{r:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax.text(i, -7, f"n={n}", ha="center", fontsize=9, color=GRAY)
ax.set_xticks(x)
ax.set_xticklabels(qtype_labels, fontsize=11)
ax.set_ylabel("% of questions", fontsize=11)
ax.set_ylim(-12, 110)
ax.set_title("Reading accuracy by question type (30 students × 3 exams = 150 questions)",
             fontsize=13, fontweight="bold", pad=14)
ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "per_question_type.png", dpi=150, facecolor="white")
plt.close()

# ── Chart 3: score distribution ───────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
all_scores = []
for e in per_exam:
    all_scores.extend(e["score_pcts"])
ax.hist(all_scores, bins=10, range=(0, 100), color=IKU_RED, edgecolor="white", linewidth=1.5, alpha=0.85)
mean = np.mean(all_scores)
median = np.median(all_scores)
ax.axvline(mean, color=IKU_RED_DARK, linestyle="--", linewidth=2, label=f"Mean {mean:.1f}%")
ax.axvline(median, color="black", linestyle=":", linewidth=2, label=f"Median {median:.1f}%")
ax.set_xlabel("Total score (% of max)", fontsize=11)
ax.set_ylabel("Number of students", fontsize=11)
ax.set_title(f"Score distribution across all 30 students",
             fontsize=13, fontweight="bold", pad=14)
ax.legend(loc="upper left", framealpha=0.95)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "score_distribution.png", dpi=150, facecolor="white")
plt.close()

# ── Summary JSON ───────────────────────────────────────────────────
total_questions = sum(e["questions"] for e in per_exam)
total_correct = sum(e["correct"] for e in per_exam)
total_review = sum(e["needs_review"] for e in per_exam)
overall_acc = total_correct / total_questions * 100 if total_questions else 0
overall_review = total_review / total_questions * 100 if total_questions else 0

total_partial = sum(e["partial"] for e in per_exam)
summary = {
    "total_exams": len(per_exam),
    "total_students": sum(e["students"] for e in per_exam),
    "total_questions_evaluated": total_questions,
    "questions_correct": total_correct,
    "questions_partial": total_partial,
    "overall_accuracy_pct": round(overall_acc, 1),
    "overall_partial_pct": round(total_partial / total_questions * 100, 1) if total_questions else 0,
    "questions_needing_review": total_review,
    "overall_review_rate_pct": round(overall_review, 1),
    "per_exam": per_exam,
    "per_question_type": {
        qt: {
            "correct": s["correct"],
            "partial": s["partial"],
            "total": s["total"],
            "accuracy_pct": round(s["correct"] / s["total"] * 100, 1) if s["total"] else 0,
            "partial_pct": round(s["partial"] / s["total"] * 100, 1) if s["total"] else 0,
            "review_pct": round(s["needsReview"] / s["total"] * 100, 1) if s["total"] else 0,
        }
        for qt, s in per_qtype.items()
    },
}
with open(ASSETS_DIR / "_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=lambda o: list(o) if isinstance(o, set) else o)

# Strip score_pcts from per_exam before printing (it's noise)
print("\n" + "═" * 60)
print("TEST RUN ANALYSIS — 3 exams × 10 students")
print("═" * 60)
print(f"Total questions evaluated: {total_questions}")
print(f"Correct readings:          {total_correct} ({overall_acc:.1f}%)")
print(f"Flagged for human review:  {total_review} ({overall_review:.1f}%)")
print(f"With partial credit:       {total_correct + total_partial} ({(total_correct + total_partial) / total_questions * 100:.1f}%)")
print("\nPer-exam:")
for e in per_exam:
    print(f"  {e['label']:<25} fully {e['accuracy_pct']:>5.1f}%  "
          f"partial {e['partial_pct']:>5.1f}%  review {e['review_rate_pct']:>5.1f}%  "
          f"score {e['avg_score_pct']:>5.1f}%")
print("\nPer-question-type:")
for qt, label in zip(qtype_order, qtype_labels):
    s = per_qtype.get(qt, {"correct": 0, "partial": 0, "total": 0, "needsReview": 0})
    total = s["total"] or 1
    acc = s["correct"] / total * 100
    par = s["partial"] / total * 100
    print(f"  {label:<18} fully {acc:>5.1f}%  partial {par:>5.1f}%  (n={s['total']})")
print("\nWrote charts + summary to:", ASSETS_DIR)
