"""Stage 1 of the edge-case test: write a proper .ikuexam definition into
the app's exam store using the **exact** v1 schema the app's editor produces.

Question-type codes the renderer expects:
    mc    = single-answer multiple choice
    ms    = multi-select
    match = matching (4 left items <-> 4 letter answers)
    fill  = fill-in-the-blank (one or more underscored slots)
    open  = open-ended

After writing the file we run the Electron app with BULK_EXPORT_AND_QUIT=1.
That spawns a hidden BrowserWindow per exam, renders the printable preview,
generates the canonical map.json, and writes the blank PDF to
iku-exam-backend/samples/blanks/<id>.pdf.

Stage 2 (a separate script) reads back the generated blank + map and overlays
five students with mild edge cases.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXAM_ID = "bil101-edge-cases-2026"
APPDATA = Path(os.environ["APPDATA"]) / "iku-exam-generator"
EXAMS_DIR = APPDATA / "exams"
EXAMS_DIR.mkdir(parents=True, exist_ok=True)

IKU_EXAM = EXAMS_DIR / f"{EXAM_ID}.ikuexam"
MAP_PATH = EXAMS_DIR / f"{EXAM_ID}.map.json"

GENERATOR_DIR = Path(r"D:/repos/ExamGeneration/iku-exam-generator")
BLANK_OUTPUT_DIR = Path(r"D:/repos/exam-backend/samples/blanks")

# ─── The exam, in the exact .ikuexam v1 schema ─────────────────
EXAM_FILE = {
    "version": 1,
    "exam": {
        "faculty": "Faculty of Engineering",
        "department": "Computer Engineering",
        "course": "Introduction to Programming",
        "courseCode": "BIL101",
        "examType": "Quiz",
        "date": "2026-05-15",
        "duration": 30,
        "instructions": (
            "Answer all questions on this single page. Mark bubbles fully. "
            "For matching, write the letter (A/B/C/D) in each numbered box. "
            "Fill-in-the-blank answers must be a single word."
        ),
    },
    "questions": [
        # Q1 — Matching (4 cells)
        # Match algorithms to their average-case time complexity.
        # Left:  1) Binary Search   2) Bubble Sort   3) Linear Search   4) Merge Sort
        # Right: A) O(n)            B) O(log n)      C) O(n log n)      D) O(n^2)
        # Correct: 1→B, 2→D, 3→A, 4→C
        {
            "type": "match",
            "text": "Match each algorithm with its average-case time complexity.",
            "imagePreview": "",
            "options": ["", "", "", ""],
            "correctAnswer": 0,
            "correctAnswers": [],
            "correctText": "",
            "correctImagePreview": "",
            "spaceId": "",
            "matchLeft": ["Binary Search", "Bubble Sort", "Linear Search", "Merge Sort"],
            "matchRight": ["O(n)", "O(log n)", "O(n log n)", "O(n^2)"],
            "matchCorrect": ["B", "D", "A", "C"],
            "fillText": "",
            "fillAnswers": [],
            "points": 10,
            "penaltyPerItem": 0,
        },
        # Q2 — Multiple choice
        # "Which data structure follows LIFO?" Correct = Stack (index 1)
        {
            "type": "mc",
            "text": "Which data structure follows the Last-In-First-Out (LIFO) principle?",
            "imagePreview": "",
            "options": ["Queue", "Stack", "Linked List", "Binary Tree"],
            "correctAnswer": 1,
            "correctAnswers": [],
            "correctText": "",
            "correctImagePreview": "",
            "spaceId": "",
            "matchLeft": ["", ""],
            "matchRight": ["", ""],
            "matchCorrect": [],
            "fillText": "",
            "fillAnswers": [],
            "points": 10,
            "penaltyPerItem": 0,
        },
        # Q3 — Multi-select
        # "Select all primitive Python data types" — int (0) and float (2)
        {
            "type": "ms",
            "text": "Select all primitive Python data types (mark all that apply).",
            "imagePreview": "",
            "options": ["int", "list", "float", "dict"],
            "correctAnswer": 0,
            "correctAnswers": [0, 2],
            "correctText": "",
            "correctImagePreview": "",
            "spaceId": "",
            "matchLeft": ["", ""],
            "matchRight": ["", ""],
            "matchCorrect": [],
            "fillText": "",
            "fillAnswers": [],
            "points": 10,
            "penaltyPerItem": 0,
        },
        # Q4 — Matching (4 cells)
        # Match languages to paradigms.
        # Left:  1) Haskell    2) C        3) Java       4) Prolog
        # Right: A) OOP         B) Functional  C) Procedural  D) Logic
        # Correct: 1→B, 2→C, 3→A, 4→D
        {
            "type": "match",
            "text": "Match each programming language with its primary paradigm.",
            "imagePreview": "",
            "options": ["", "", "", ""],
            "correctAnswer": 0,
            "correctAnswers": [],
            "correctText": "",
            "correctImagePreview": "",
            "spaceId": "",
            "matchLeft": ["Haskell", "C", "Java", "Prolog"],
            "matchRight": ["Object-Oriented", "Functional", "Procedural", "Logic"],
            "matchCorrect": ["B", "C", "A", "D"],
            "fillText": "",
            "fillAnswers": [],
            "points": 10,
            "penaltyPerItem": 0,
        },
        # Q5 — Fill in the blank: "Python"
        {
            "type": "fill",
            "text": "",
            "imagePreview": "",
            "options": ["", "", "", ""],
            "correctAnswer": 0,
            "correctAnswers": [],
            "correctText": "",
            "correctImagePreview": "",
            "spaceId": "",
            "matchLeft": ["", ""],
            "matchRight": ["", ""],
            "matchCorrect": [],
            "fillText": (
                "The high-level interpreted programming language whose reference "
                "implementation is CPython is called ____."
            ),
            "fillAnswers": ["Python"],
            "points": 10,
            "penaltyPerItem": 0,
        },
    ],
    "updatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}


def write_exam() -> None:
    IKU_EXAM.write_text(
        json.dumps(EXAM_FILE, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {IKU_EXAM} ({IKU_EXAM.stat().st_size / 1024:.1f} KB)")
    # If a stale map.json exists, drop it so the renderer regenerates fresh.
    if MAP_PATH.exists():
        MAP_PATH.unlink()
        print(f"Removed stale {MAP_PATH.name}")


def run_bulk_export() -> None:
    """Spawn the built Electron app with BULK_EXPORT_AND_QUIT=1.

    The main process iterates every saved exam, opens a hidden window per
    exam to drive HeadlessExporter, captures the rendered preview as PDF,
    and quits. The renderer side writes <id>.map.json into the userData
    exams directory; main writes <id>.pdf into samples/blanks/.
    """
    BLANK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["BULK_EXPORT_AND_QUIT"] = "1"
    # Avoid Electron noise about default sandbox / GPU on this hardware
    env.setdefault("ELECTRON_DISABLE_SECURITY_WARNINGS", "1")

    print(f"Spawning Electron in {GENERATOR_DIR} with BULK_EXPORT_AND_QUIT=1")
    print("(this iterates every saved exam — expect roughly 10–30 s per exam)")

    proc = subprocess.run(
        ["npx", "--no-install", "electron", "."],
        cwd=str(GENERATOR_DIR),
        env=env,
        capture_output=True,
        text=True,
        shell=True,    # npx.cmd on Windows
        timeout=600,
    )

    # Echo only the most useful lines
    print("─── stdout tail ───")
    print("\n".join(proc.stdout.splitlines()[-40:]))
    if proc.returncode != 0:
        print("─── stderr tail ───")
        print("\n".join(proc.stderr.splitlines()[-30:]))
        print(f"\nElectron exited with code {proc.returncode}")
    else:
        print(f"\nElectron exited cleanly ({proc.returncode})")


if __name__ == "__main__":
    write_exam()
    run_bulk_export()

    # Sanity check the outputs
    print()
    print("─── Post-export state ───")
    blank_pdf = BLANK_OUTPUT_DIR / f"{EXAM_ID}.pdf"
    if blank_pdf.exists():
        print(f"OK   blank PDF: {blank_pdf}  ({blank_pdf.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"MISS blank PDF: {blank_pdf}")

    if MAP_PATH.exists():
        print(f"OK   map.json:  {MAP_PATH}  ({MAP_PATH.stat().st_size / 1024:.1f} KB)")
        # Show a quick structural sanity check
        m = json.loads(MAP_PATH.read_text(encoding="utf-8"))
        npages = len(m.get("pages", []))
        nq = sum(len(p.get("questions", {})) for p in m.get("pages", []))
        print(f"     pages={npages}, questions={nq}, totalPages={m.get('totalPages')}")
    else:
        print(f"MISS map.json:  {MAP_PATH}")
