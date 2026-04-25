"""
Import the 3 v6 sample exams into the Electron app's local store so the
user can evaluate them via the Dashboard UI.

For each map:
  - Generate a minimal .ikuexam file from the map's questions/answers.
  - Copy the map JSON as <id>.map.json so the Dashboard can load it.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime

EXAMS_DIR = os.path.expandvars(r"%APPDATA%\iku-exam-generator\exams")
SAMPLES_DIR = r"C:\Users\faruk\Downloads\v6\samples"

# (target_id, friendly_label, source_map_filename)
IMPORTS = [
    ("v6_multiple_choice", "Mobile Programming - MC test", "MULTIPLECHOICES4064-midterm-2026_map(3).json"),
    ("v6_open_ended",      "Mobile Programming - Open Ended test", "OPENENDED4064-midterm-2026_map(3).json"),
    ("v6_match_open_fill", "Mobile Programming - Mixed test",      "MATCHOPEN4064-midterm-2026_map(3).json"),
]

TYPE_BACKMAP = {
    "multiple_choice": "mc",
    "multi_select":    "ms",
    "open_ended":      "open",
    "matching":        "match",
    "fill_blanks":     "fill",
}

ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _empty_question_template() -> dict:
    """Default Question fields matching types/exam.ts shape."""
    return {
        "type": "open",
        "text": "",
        "imagePreview": None,
        "options": [],
        "correctAnswer": 0,
        "correctAnswers": [],
        "correctText": "",
        "correctImagePreview": None,
        "spaceId": 1,
        "matchLeft": [],
        "matchRight": [],
        "matchCorrect": [],
        "fillText": "",
        "fillAnswers": [],
        "points": 10,
        "penaltyPerItem": 0,
    }


def map_question_to_ikuexam_question(q_num: str, q_data: dict) -> dict:
    """Convert a map JSON question into an ikuexam Question object."""
    q = _empty_question_template()
    q_type = q_data.get("type", "open_ended")
    q["type"] = TYPE_BACKMAP.get(q_type, "open")
    q["text"] = q_data.get("questionText", f"Question {q_num}")
    q["points"] = int((q_data.get("scoring") or {}).get("points", 10))
    q["penaltyPerItem"] = int((q_data.get("scoring") or {}).get("penaltyPerItem") or 0)

    expected = q_data.get("expectedAnswer", {}) or {}

    if q_type == "multiple_choice":
        n = int((q_data.get("scoring") or {}).get("itemCount", len(q_data.get("options", {}))))
        q["options"] = [f"Option {ALPHA[i]}" for i in range(n)]
        opt = expected.get("correctOption") or "A"
        q["correctAnswer"] = ALPHA.index(opt) if opt in ALPHA else 0

    elif q_type == "multi_select":
        n = int((q_data.get("scoring") or {}).get("itemCount", len(q_data.get("options", {}))))
        q["options"] = [f"Option {ALPHA[i]}" for i in range(n)]
        ans = expected.get("correctOptions") or []
        q["correctAnswers"] = [ALPHA.index(a) for a in ans if a in ALPHA]

    elif q_type == "matching":
        matches = expected.get("correctMatches") or {}
        n = max([int(k) for k in matches.keys()] + [0])
        q["matchLeft"]  = [f"Item {i+1}" for i in range(n)]
        q["matchRight"] = [f"{ALPHA[i]}" for i in range(n)]
        q["matchCorrect"] = [matches.get(str(i + 1), "") for i in range(n)]

    elif q_type == "fill_blanks":
        blanks = expected.get("correctBlanks") or {}
        n = max([int(k) for k in blanks.keys()] + [0])
        q["fillText"] = " ".join(["___"] * n)
        q["fillAnswers"] = [blanks.get(str(i + 1), "") for i in range(n)]

    elif q_type == "open_ended":
        q["correctText"] = expected.get("text", "")

    return q


def build_ikuexam(map_data: dict) -> dict:
    """Construct a minimal but valid .ikuexam JSON from a map."""
    parts = (map_data.get("examId", "exam-2026")).split("-")
    course_code = parts[0] if parts else "0000"
    exam_type = parts[1] if len(parts) > 1 else "midterm"
    date = map_data.get("date") or "2026-04-23"

    questions = []
    for page in map_data.get("pages", []):
        for q_num in sorted(page.get("questions", {}).keys(), key=int):
            q_data = page["questions"][q_num]
            questions.append(map_question_to_ikuexam_question(q_num, q_data))

    return {
        "version": 1,
        "exam": {
            "faculty": "Engineering",
            "department": "Computer Engineering",
            "course": "Mobile Programming",
            "courseCode": course_code,
            "examType": exam_type,
            "date": date,
            "duration": "60",
            "instructions": "Imported from v6 sample for testing.",
        },
        "questions": questions,
        "updatedAt": datetime.utcnow().isoformat() + "Z",
    }


def main():
    if not os.path.isdir(EXAMS_DIR):
        os.makedirs(EXAMS_DIR, exist_ok=True)
    print(f"Target dir: {EXAMS_DIR}")
    for target_id, label, src_map_name in IMPORTS:
        src_map_path = os.path.join(SAMPLES_DIR, src_map_name)
        if not os.path.isfile(src_map_path):
            print(f"  SKIP {target_id}: missing {src_map_path}")
            continue

        with open(src_map_path, "r", encoding="utf-8") as f:
            map_data = json.load(f)

        ikuexam_data = build_ikuexam(map_data)
        # Override course label so user can recognize them in dashboard
        ikuexam_data["exam"]["course"] = label

        ikuexam_path = os.path.join(EXAMS_DIR, f"{target_id}.ikuexam")
        with open(ikuexam_path, "w", encoding="utf-8") as f:
            json.dump(ikuexam_data, f, indent=2, ensure_ascii=False)

        # Copy the map alongside
        dst_map_path = os.path.join(EXAMS_DIR, f"{target_id}.map.json")
        shutil.copyfile(src_map_path, dst_map_path)

        # Remove any stale results
        results_path = os.path.join(EXAMS_DIR, f"{target_id}.results.json")
        if os.path.exists(results_path):
            os.remove(results_path)

        print(f"  imported {target_id}: {len(ikuexam_data['questions'])} questions, "
              f"label='{label}'")

    print("done")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
