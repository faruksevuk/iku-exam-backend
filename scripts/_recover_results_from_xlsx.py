"""One-shot recovery: reconstruct a results.json from a backend xlsx
plus the corresponding map.json. Used to rescue runs where the renderer
was killed before receiving the /evaluate response.

Usage:
    py scripts/_recover_results_from_xlsx.py 4012-se-midterm-2026

Reads:
    D:/repos/exam-backend/output/{slug}_results.xlsx
    %APPDATA%/iku-exam-generator/exams/{slug}.map.json

Writes:
    %APPDATA%/iku-exam-generator/exams/{slug}.results.json

Lossiness: the xlsx doesn't carry per-cell OCR base64 images or page
filenames. Recovered students will have score + needsReview + a coarse
explanation, which is enough for the dashboard, the students table, and
score-edit / approve flows. Full per-question OCR detail (image crops,
fine-grained ocrAnswers map) is reconstructable approximately from the
"Detail" column where present.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import openpyxl

USER_DATA = Path(os.environ["APPDATA"]) / "iku-exam-generator" / "exams"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def parse_q3_detail(detail: str) -> dict:
    """Extract OCR answers from Q3 (matching) detail string.
    Format: "4/4 correct. #1: 'A' = 'A' [OK] | #2: 'C' = 'C' [OK] | ..."
    """
    ocr: dict[str, str] = {}
    confs: dict[str, float] = {}
    for m in re.finditer(r"#(\d+):\s*'([^']*)'\s*=\s*'([^']*)'\s*\[(OK|X)\]", detail or ""):
        ocr[m.group(1)] = m.group(2)
        # No per-cell conf in detail; reuse the question conf below.
    return {"ocrAnswers": ocr, "ocrConfidences": confs}


def parse_q4_detail(detail: str) -> dict:
    """Q4 fill_blanks detail: "1/1 correct. #1: 'Aaile' = 'Agile' [OK] ..."
    """
    return parse_q3_detail(detail)


def recover(slug: str) -> None:
    xlsx_path = OUTPUT_DIR / f"{slug}_results.xlsx"
    map_path = USER_DATA / f"{slug}.map.json"
    out_path = USER_DATA / f"{slug}.results.json"

    if not xlsx_path.exists():
        sys.exit(f"No xlsx at {xlsx_path}")
    if not map_path.exists():
        sys.exit(f"No map at {map_path}")

    map_data = json.loads(map_path.read_text(encoding="utf-8"))
    # Build qNum → type lookup
    qtype_by_num: dict[str, str] = {}
    qmax_by_num: dict[str, float] = {}
    for page in map_data.get("pages", []):
        for q_num, q in (page.get("questions") or {}).items():
            qtype_by_num[q_num] = q.get("type", "unknown")
            qmax_by_num[q_num] = float(
                (q.get("scoring") or {}).get("points", 0) or 0,
            )

    wb = openpyxl.load_workbook(str(xlsx_path), data_only=True)
    sh = wb["Scores"]
    header = [c.value for c in sh[1]]

    # Identify column indices (1-based for openpyxl)
    def col(name: str) -> int | None:
        for i, h in enumerate(header, start=1):
            if h == name:
                return i
        return None

    students: list[dict] = []
    for row in sh.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        rec = dict(zip(header, row))
        sn = str(rec.get("Student Number") or "").strip()
        sn_conf = float(rec.get("SN Confidence") or 0) / 100.0

        # Build per-question dict.
        questions: dict = {}
        total_score = 0.0
        total_max = 0.0
        for q_num, q_type in qtype_by_num.items():
            score = rec.get(f"Q{q_num} Score")
            conf = rec.get(f"Q{q_num} Conf")
            detail = rec.get(f"Q{q_num} Detail") or ""
            if score is None:
                continue
            score_f = float(score)
            conf_f = float(conf) / 100.0 if conf is not None else 0.0
            max_pts = qmax_by_num.get(q_num, 0.0)
            total_score += score_f
            total_max += max_pts

            q: dict = {
                "type": q_type,
                "score": score_f,
                "maxPoints": max_pts,
                "confidence": conf_f,
                "explanation": detail,
                "status": "correct" if score_f >= max_pts else "wrong" if score_f == 0 else "partial",
                "needsReview": conf_f < 0.80,
            }
            # Type-specific fields parsed out of the detail string.
            if q_type == "matching":
                pq = parse_q3_detail(detail)
                if pq["ocrAnswers"]:
                    q["ocrAnswers"] = pq["ocrAnswers"]
            elif q_type == "fill_blanks":
                pq = parse_q4_detail(detail)
                if pq["ocrAnswers"]:
                    q["ocrAnswers"] = pq["ocrAnswers"]
            elif q_type == "multiple_choice":
                # Detail like "Correct (B)..." or "Wrong — chose A, answer is B"
                m = re.search(r"\b(?:chose|Correct \(|\(see )([A-D])", detail or "")
                if m:
                    q["selectedOption"] = m.group(1)
            elif q_type == "open_ended":
                q["status"] = "graded" if score_f > 0 else "wrong"
                q["needsReview"] = float(rec.get("Needs Review") == "YES")
            questions[q_num] = q

        students.append({
            "studentNumber": sn,
            "studentNumberConfidence": sn_conf,
            "studentNumberImage": "",
            "questions": questions,
            "totalScore": round(total_score, 2),
            "totalMaxPoints": total_max,
            # No `pages` array — backend image filenames are predictable
            # (`{sn}_page{N}.jpg`) but the count varies. Set later if a
            # full re-eval is run; the workspace falls back to schematic
            # view when pages is missing.
        })

    out = {
        "examId": slug,
        "generatedAt": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "students": students,
        "overrideHistory": [],
        "_recovered": True,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Recovered {len(students)} students -> {out_path}")
    avg_pct = sum(
        (s["totalScore"] / s["totalMaxPoints"] * 100) for s in students if s["totalMaxPoints"] > 0
    ) / len(students)
    print(f"Average: {avg_pct:.1f}%")


if __name__ == "__main__":
    slug = sys.argv[1] if len(sys.argv) > 1 else "4012-se-midterm-2026"
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    recover(slug)
