"""
install_sample_exams.py — drop two production-shape .ikuexam files
into the Electron app's userData so they show up in Evaluations.

Schema matches what the app's QuestionEditor + Preview "Save" flow
writes (see demo-workspace.ikuexam for the canonical shape). The
matching `.map.json` is INTENTIONALLY not produced by this script —
the app's `utils/mapGenerator.ts` derives map coordinates from the
rendered DOM at Save time, which we can't replicate offline.

Workflow:
  1. Run this script — exams appear under userData/exams/.
  2. In the app, open each exam (Edit) → click through to Step 2
     (Preview) → click Save. The app generates and writes the
     `.map.json` for that exam at that moment.
  3. After both maps exist, drag a student PDF onto the card on
     the Evaluations tab to grade it.

Outputs:
  - cse301-ds-quiz-2026.ikuexam        (8 q, 50 pts)
  - mat102-calc2-midterm-2026.ikuexam  (10 q, 70 pts)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

USER_DATA = Path(os.environ.get("APPDATA", "")) / "iku-exam-generator" / "exams"

NOW = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def q(
    type_: str,
    text: str,
    *,
    options: list[str] | None = None,
    correctAnswer: int = 0,
    correctAnswers: list[int] | None = None,
    correctText: str = "",
    matchLeft: list[str] | None = None,
    matchRight: list[str] | None = None,
    matchCorrect: list[str] | None = None,
    fillText: str = "",
    fillAnswers: list[str] | None = None,
    points: int = 5,
    penaltyPerItem: int = 0,
) -> dict:
    """Build a question dict with every QuestionEditor field set, even
    if unused for this type. Matches demo-workspace.ikuexam."""
    return {
        "type": type_,
        "text": text,
        "imagePreview": None,
        "options": options or [],
        "correctAnswer": correctAnswer,
        "correctAnswers": correctAnswers or [],
        "correctText": correctText,
        "correctImagePreview": None,
        "spaceId": 0,
        "matchLeft": matchLeft or [],
        "matchRight": matchRight or [],
        "matchCorrect": matchCorrect or [],
        "fillText": fillText,
        "fillAnswers": fillAnswers or [],
        "points": points,
        "penaltyPerItem": penaltyPerItem,
    }


# ── Exam 1: Data Structures Quiz ─────────────────────────────────

CSE301 = {
    "version": 1,
    "exam": {
        "faculty": "Engineering",
        "department": "Computer Science",
        "course": "Data Structures",
        "courseCode": "CSE301",
        "examType": "Quiz",
        "date": "2026-05-12",
        "duration": "45",
        "instructions": "Answer every question. Show your reasoning where applicable.",
    },
    "questions": [
        q("mc",
          "Which data structure follows the LIFO (Last-In-First-Out) principle?",
          options=["Queue", "Stack", "Heap", "Tree"],
          correctAnswer=1, points=5),
        q("mc",
          "What is the average-case time complexity of binary search on a sorted array?",
          options=["O(n)", "O(log n)", "O(n log n)", "O(1)"],
          correctAnswer=1, points=5),
        q("mc",
          "Which of these is NOT a balanced binary search tree?",
          options=["AVL tree", "Red-black tree", "Linked list", "B-tree"],
          correctAnswer=2, points=5),
        q("ms",
          "Which of the following are linear data structures? (select all)",
          options=["Array", "Stack", "Binary tree", "Linked list"],
          correctAnswers=[0, 1, 3], points=8, penaltyPerItem=2),
        q("ms",
          "Which operations does a typical hash table support in O(1) average time?",
          options=["Insert", "Lookup", "Sort", "Delete"],
          correctAnswers=[0, 1, 3], points=8, penaltyPerItem=2),
        q("fill",
          "A queue follows the ___ principle, and the operation that removes "
          "the front element is called ___.",
          fillText="A queue follows the ___ principle, and the operation that removes the front element is called ___.",
          fillAnswers=["FIFO", "dequeue"],
          points=6, penaltyPerItem=0),
        q("match",
          "Match each data structure with its primary use case.",
          matchLeft=["Hash table", "Min-heap", "Trie"],
          matchRight=[
              "Prefix-based string lookup",
              "Constant-time average key lookup",
              "Repeated retrieval of the smallest element",
          ],
          matchCorrect=["B", "C", "A"],
          points=6, penaltyPerItem=0),
        q("open",
          "Briefly explain the difference between a stack and a queue (1-2 sentences).",
          correctText=(
              "A stack is LIFO — items are pushed and popped at the same end. "
              "A queue is FIFO — items are enqueued at one end and dequeued at the other."
          ),
          points=7, penaltyPerItem=0),
    ],
    "updatedAt": NOW,
}


# ── Exam 2: Calculus II Midterm ──────────────────────────────────

MAT102 = {
    "version": 1,
    "exam": {
        "faculty": "Engineering",
        "department": "Mathematics",
        "course": "Calculus II",
        "courseCode": "MAT102",
        "examType": "Midterm",
        "date": "2026-05-18",
        "duration": "90",
        "instructions": "Answer every question. You may use a non-graphing calculator.",
    },
    "questions": [
        q("mc",
          "What is the derivative of sin(x)?",
          options=["cos(x)", "-cos(x)", "sin(x)", "-sin(x)"],
          correctAnswer=0, points=5),
        q("mc",
          "Which integration technique is most appropriate for ∫ x · e^x dx?",
          options=["Substitution", "Integration by parts", "Partial fractions", "Trigonometric substitution"],
          correctAnswer=1, points=5),
        q("mc",
          "What is the value of the indefinite integral ∫ 1/x dx?",
          options=["1/(x²)", "ln|x| + C", "x · ln(x)", "-1/x²"],
          correctAnswer=1, points=5),
        q("mc",
          "A series Σ aₙ converges absolutely if:",
          options=[
              "Σ |aₙ| converges",
              "Σ aₙ converges and Σ |aₙ| diverges",
              "lim aₙ = 0",
              "aₙ is alternating",
          ],
          correctAnswer=0, points=5),
        q("ms",
          "Which of the following series converge? (select all)",
          options=[
              "Σ 1/n² (from n=1)",
              "Σ 1/n (from n=1)",
              "Σ (1/2)^n (from n=0)",
              "Σ (-1)^n / n (from n=1)",
          ],
          correctAnswers=[0, 2, 3], points=8, penaltyPerItem=2),
        q("ms",
          "Which functions have an antiderivative expressible in elementary terms?",
          options=["e^x", "e^(x²)", "1/x", "sin(x²)"],
          correctAnswers=[0, 2], points=8, penaltyPerItem=2),
        q("fill",
          "The derivative of e^(2x) is ___ , and the integral of cos(3x) dx is ___ + C.",
          fillText="The derivative of e^(2x) is ___ , and the integral of cos(3x) dx is ___ + C.",
          fillAnswers=["2e^(2x)", "(1/3)sin(3x)"],
          points=6, penaltyPerItem=0),
        q("fill",
          "Taylor series of sin(x) around x=0 is x - x³/___ + x⁵/___ - ...",
          fillText="Taylor series of sin(x) around x=0 is x - x³/___ + x⁵/___ - ...",
          fillAnswers=["6", "120"],
          points=6, penaltyPerItem=0),
        q("match",
          "Match each function with its derivative.",
          matchLeft=["x²", "ln(x)", "e^x"],
          matchRight=["1/x", "e^x", "2x"],
          matchCorrect=["C", "A", "B"],
          points=6, penaltyPerItem=0),
        q("open",
          "Use integration by parts to evaluate ∫ x · ln(x) dx. Show every step.",
          correctText=(
              "Let u = ln(x), dv = x dx → du = (1/x) dx, v = x²/2. "
              "∫ x ln(x) dx = (x²/2) ln(x) - ∫ (x²/2)(1/x) dx "
              "= (x²/2) ln(x) - x²/4 + C."
          ),
          points=16, penaltyPerItem=0),
    ],
    "updatedAt": NOW,
}


def install(slug: str, data: dict) -> None:
    USER_DATA.mkdir(parents=True, exist_ok=True)
    target = USER_DATA / f"{slug}.ikuexam"
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    n_q = len(data["questions"])
    pts = sum(q.get("points", 0) for q in data["questions"])
    print(f"[ok] wrote {target.name}  ({n_q} q, {pts} pts)")


def main() -> None:
    install("cse301-ds-quiz-2026", CSE301)
    install("mat102-calc2-midterm-2026", MAT102)
    print()
    print("Next steps in the app:")
    print("  1. Refresh the dashboard (Ctrl+R) — both exams appear on Evaluations.")
    print("  2. Click Edit on each card → step through to Preview → click Save.")
    print("     This invokes utils/mapGenerator.ts, writing the matching .map.json.")
    print("  3. Drag a student PDF onto the card to start grading.")


if __name__ == "__main__":
    main()
