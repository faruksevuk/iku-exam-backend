"""
Pipeline — the end-to-end evaluation loop.

Flow:
  1. splitting.pdf_to_images  → list of page images
  2. splitting.split_by_students  → [StudentExam] (QR-based)
  3. For each student:
       a. alignment.align_page(page 1) → read student number (digit CNN)
       b. For each page: align → dispatch each question by type:
            multiple_choice → omr.evaluate_mc → grading.score_mc
            multi_select    → omr.evaluate_ms → grading.score_ms
            matching        → handwriting.read_letter_box → grading.score_match
            fill_blanks     → handwriting.read_text_box   → grading.score_fill
            open_ended      → ai_evaluation.evaluate_open_ended → grading.score_open
  4. export.export_results → Excel

Emits per-question image crops (base64 jpeg) for the frontend Review UI:
  - studentNumberImage
  - answerImage (full question bbox) OR solutionAreaImage (open-ended)

────────────────────────────────────────────────────────────────────
Structured progress log lines — read by the Electron renderer's
PipelineMonitor view via the existing backend:log-event IPC. KEEP THE
SHAPE STABLE — the renderer regex-matches these lines.

  [STAGE] sIdx=<i> sTotal=<n> sNum=<num> stage=<key>           — entering stage
  [STAGE] stage=batch state=start sTotal=<n>                   — run start
  [STAGE] stage=save  state=start                              — excel export
  [STAGE] stage=save  state=end   excel=<path>                 — fully done

`stage` ∈ {read, grade, done, save, batch}.
  read  = a student is being aligned + OMR/handwriting-OCR'd
  grade = a student has at least one open_ended question being scored
  done  = a student fully evaluated (questions populated, scores summed)
  save  = excel + json export at end of run (single, not per-student)
  batch = wraps the whole run; useful for the UI to size progress bars

Renderers must tolerate extra trailing fields and unknown stages.
────────────────────────────────────────────────────────────────────
"""

import asyncio
import os
import time
import traceback
from typing import Any, Dict, List

import cv2
import numpy as np

import config
import alignment
import preprocessing
import handwriting
import omr
import ai_evaluation
import grading
import export
from splitting import StudentExam, pdf_to_images, split_by_students


# ══════════════════════════════════════════════════════════════════
#  Question-level evaluation
# ══════════════════════════════════════════════════════════════════

async def evaluate_question(
    aligned: np.ndarray,
    q_data: Dict,
    q_num: str,
) -> Dict[str, Any]:
    """Dispatch one question to the right reader + grader by type."""
    q_type = q_data.get("type", "unknown")
    scoring = q_data.get("scoring", {})
    expected = q_data.get("expectedAnswer", {}) or {}

    result: Dict[str, Any] = {"type": q_type, "expected": expected}

    # Review UI crop — the full question bounding box
    bbox = q_data.get("boundingBox")
    if bbox:
        raw = preprocessing.crop_raw(aligned, bbox)
        result["answerImage"] = preprocessing.encode_jpeg_b64(raw)

    # ── MC ──
    if q_type == "multiple_choice":
        options = q_data.get("options", {}) or {}
        omr_res = omr.evaluate_mc(aligned, options, expected.get("correctOption"))
        score_res = grading.score_mc(omr_res, scoring, expected)
        result.update(omr_res)
        result.update(score_res)
        return result

    # ── MS ──
    if q_type == "multi_select":
        options = q_data.get("options", {}) or {}
        omr_res = omr.evaluate_ms(aligned, options, expected.get("correctOptions"))
        score_res = grading.score_ms(omr_res, scoring, expected)
        result.update(omr_res)
        result.update(score_res)
        return result

    # ── Matching ──
    if q_type == "matching":
        answer_boxes = q_data.get("answerBoxes", {}) or {}
        # The valid letter set is the union of correct matches — pass it to
        # the reader so the 26-class CNN collapses to N-class for this question.
        correct_matches = expected.get("correctMatches", {}) or {}
        expected_set = {
            str(v).upper() for v in correct_matches.values()
            if isinstance(v, str) and len(v) == 1 and v.isalpha()
        } or None
        ocr_answers: Dict[str, str] = {}
        confidences: Dict[str, float] = {}
        review_flags: Dict[str, bool] = {}
        for idx, box in answer_boxes.items():
            reader_res = handwriting.read_letter_box(aligned, box, expected_set=expected_set)
            ocr_answers[idx] = reader_res.text
            confidences[idx] = reader_res.confidence
            review_flags[idx] = reader_res.needs_review
        score_res = grading.score_match(ocr_answers, expected, scoring, confidences, review_flags)
        result.update(score_res)
        result["ocrAnswers"] = ocr_answers
        result["ocrConfidences"] = confidences
        return result

    # ── Fill blanks ──
    if q_type == "fill_blanks":
        answer_boxes = q_data.get("answerBoxes", {}) or {}
        correct_blanks = expected.get("correctBlanks", {}) or {}
        ocr_answers = {}
        confidences = {}
        review_flags = {}
        sources: Dict[str, str] = {}
        for idx, box in answer_boxes.items():
            # Pass per-blank expected so the reader can detect TrOCR misreads
            # that look confident but don't fuzzy-match the answer key.
            blank_expected = str(correct_blanks.get(idx, "") or "")
            reader_res = handwriting.read_text_box(aligned, box, expected=blank_expected)
            ocr_answers[idx] = reader_res.text
            confidences[idx] = reader_res.confidence
            review_flags[idx] = reader_res.needs_review
            sources[idx] = reader_res.source
        score_res = grading.score_fill(ocr_answers, expected, scoring, confidences, review_flags)
        result.update(score_res)
        result["ocrAnswers"] = ocr_answers
        result["ocrConfidences"] = confidences
        result["ocrSources"] = sources
        return result

    # ── Open-ended ──
    if q_type == "open_ended":
        solution_box = q_data.get("solutionArea")
        student_answer_crop = None
        if solution_box:
            raw = preprocessing.crop_raw(aligned, solution_box)
            result["solutionAreaImage"] = preprocessing.encode_jpeg_b64(raw)
            student_answer_crop = preprocessing.crop_for_reading(aligned, solution_box, pad=12)

        if expected.get("text"):
            result["expectedText"] = expected["text"]

        if student_answer_crop is not None and not preprocessing.is_blank(student_answer_crop):
            ai_res = await ai_evaluation.evaluate_open_ended(
                student_answer_image=student_answer_crop,
                question_text=q_data.get("questionText", f"Question {q_num}"),
                rubric_text=expected.get("text", ""),
                max_points=float(scoring.get("points", 0) or 0),
                reference_image=None,  # TODO: decode expected.images[0] when AI_ENABLED
            )
            score_res = grading.score_open(scoring, ai_res)
        else:
            score_res = {
                "score": 0.0,
                "maxPoints": float(scoring.get("points", 0) or 0),
                "status": "blank",
                "confidence": 1.0,
                "explanation": "Blank — no answer written.",
                "needsReview": False,
            }

        result.update(score_res)
        return result

    # ── Unknown ──
    result.update({
        "score": 0.0,
        "maxPoints": float(scoring.get("points", 0) or 0),
        "status": "error",
        "confidence": 0.0,
        "explanation": f"Unknown question type: {q_type}",
        "needsReview": True,
    })
    return result


async def evaluate_student_page(
    aligned: np.ndarray,
    page_data: Dict,
) -> Dict[str, Any]:
    """Evaluate every question on a single aligned page."""
    questions = page_data.get("questions", {}) or {}
    results: Dict[str, Any] = {}
    for q_num, q_data in questions.items():
        try:
            # Wall-clock time per question. Aggregated into stage
            # totals (ocr_ms / ai_ms) on the [STAGE] done event so the
            # renderer can show "OCR 32s · Grading 48s" per student.
            _t0 = time.time()
            results[q_num] = await evaluate_question(aligned, q_data, q_num)
            elapsed = int((time.time() - _t0) * 1000)
            results[q_num]["_time_ms"] = elapsed
            # Per-question progress event — emitted IMMEDIATELY after
            # each question finishes so the renderer can animate
            # progress between coarse student-level stage transitions.
            # Without this, between [STAGE] read and [STAGE] done
            # (often 60-120s for an open-ended-heavy exam) there are
            # zero updates and the UI looks frozen.
            q_type = q_data.get("type", "")
            print(
                f"[STAGE] stage=q qNum={q_num} qType={q_type} "
                f"elapsedMs={elapsed}"
            )
        except Exception as e:
            traceback.print_exc()
            results[q_num] = {
                "type": q_data.get("type", "unknown"),
                "maxPoints": float(q_data.get("scoring", {}).get("points", 0) or 0),
                "score": 0.0,
                "confidence": 0.0,
                "status": "error",
                "explanation": f"Evaluation error: {str(e)[:200]}",
                "needsReview": True,
            }
    return results


# ══════════════════════════════════════════════════════════════════
#  Per-student evaluation
# ══════════════════════════════════════════════════════════════════

async def evaluate_student(
    student: StudentExam,
    map_pages: List[Dict],
    student_index: int,
    student_total: int = 0,
) -> Dict[str, Any]:
    """Align every page, read the student number, evaluate every question.

    `student_total` is forwarded into the [STAGE] progress lines so the
    renderer can size progress bars correctly. Defaults to 0 when called
    from older callers; the renderer treats 0 as "unknown".
    """
    aligned_cache: Dict[int, np.ndarray] = {}
    # Mark the entry into the "read" stage for this student. The student
    # number isn't known yet; we'll re-emit with sNum after alignment.
    print(f"[STAGE] sIdx={student_index} sTotal={student_total} stage=read")

    # Align page 1 → read student number
    first_page = student.pages[0] if student.pages else None
    if first_page and map_pages:
        p1_map = map_pages[0]
        aligned_p1 = alignment.align_page(
            first_page["image"],
            p1_map.get("anchors", {}),
            page_width=int(p1_map.get("pageWidth", config.PAGE_WIDTH_PX)),
            page_height=int(p1_map.get("pageHeight", config.PAGE_HEIGHT_PX)),
        )
        aligned_cache[1] = aligned_p1

        sn_boxes = p1_map.get("studentNumberBoxes", [])
        if sn_boxes:
            sn_result = handwriting.read_student_number(aligned_p1, sn_boxes)
            student.student_number = sn_result.text or f"unknown_{student_index + 1}"
            student.student_number_confidence = sn_result.confidence
            # Crop the full SN region for the review UI
            min_x = min(b["x"] for b in sn_boxes) - 4
            min_y = min(b["y"] for b in sn_boxes) - 4
            max_x = max(b["x"] + b["w"] for b in sn_boxes) + 4
            max_y = max(b["y"] + b["h"] for b in sn_boxes) + 4
            sn_region_crop = preprocessing.crop_raw(
                aligned_p1,
                {"x": min_x, "y": min_y, "w": max_x - min_x, "h": max_y - min_y},
            )
            student.student_number_image = preprocessing.encode_jpeg_b64(sn_region_crop)
        else:
            student.student_number = f"unknown_{student_index + 1}"
            student.student_number_confidence = 0.0
            student.student_number_image = ""

    sn_label = student.student_number.replace("?", "X")
    print(f"[Pipeline] [{student_index + 1}] Student: "
          f"{student.student_number} (conf={student.student_number_confidence:.2f})")
    # Now that we know the student number, re-emit the read-stage entry
    # tagged with sNum so the log terminal can show "Reading 2400123456".
    print(f"[STAGE] sIdx={student_index} sTotal={student_total} "
          f"sNum={sn_label} stage=read")

    # Evaluate every page for this student
    all_q_results: Dict[str, Any] = {}
    student_pages: List[Dict[str, Any]] = []
    for page_info in student.pages:
        exam_page_num = page_info["examPageNum"]
        map_idx = exam_page_num - 1
        if map_idx < 0 or map_idx >= len(map_pages):
            continue

        if exam_page_num not in aligned_cache:
            pg_map = map_pages[map_idx]
            aligned = alignment.align_page(
                page_info["image"],
                pg_map.get("anchors", {}),
                page_width=int(pg_map.get("pageWidth", config.PAGE_WIDTH_PX)),
                page_height=int(pg_map.get("pageHeight", config.PAGE_HEIGHT_PX)),
            )
            aligned_cache[exam_page_num] = aligned
        else:
            aligned = aligned_cache[exam_page_num]

        # Save aligned page image for debugging + frontend canvas background.
        # The filename is recorded in `student_pages` so the frontend can
        # request it on demand from /aligned-page/{filename}.
        page_image_filename = f"{sn_label}_page{exam_page_num}.jpg"
        try:
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            cv2.imwrite(
                os.path.join(config.OUTPUT_DIR, page_image_filename),
                aligned,
            )
        except Exception:
            pass

        student_pages.append({
            "pageNum": exam_page_num,
            "imageFilename": page_image_filename,
        })

        page_results = await evaluate_student_page(aligned, map_pages[map_idx])
        all_q_results.update(page_results)

    total_score = sum(r.get("score", 0) for r in all_q_results.values())
    total_max = sum(r.get("maxPoints", 0) for r in all_q_results.values())

    # Detect whether this student had any AI-graded (open_ended) work so
    # the renderer's "Grading" stage card can be marked done meaningfully.
    had_ai = any(
        r.get("type") == "open_ended" for r in all_q_results.values()
    )

    # Aggregate per-question timings into stage buckets — emit on the
    # [STAGE] done event so the renderer can show per-student per-stage
    # durations ("OCR 32s · Grading 48s"). The `_time_ms` field is
    # popped here so the public results.json stays clean.
    ocr_ms = 0
    ai_ms = 0
    for r in all_q_results.values():
        ms = int(r.pop("_time_ms", 0) or 0)
        if r.get("type") == "open_ended":
            ai_ms += ms
        else:
            ocr_ms += ms

    print(f"[STAGE] sIdx={student_index} sTotal={student_total} "
          f"sNum={sn_label} stage=done hadAi={int(had_ai)} "
          f"ocrMs={ocr_ms} aiMs={ai_ms}")

    return {
        "studentNumber": student.student_number,
        "studentNumberConfidence": student.student_number_confidence,
        "studentNumberImage": student.student_number_image,
        "questions": all_q_results,
        "totalScore": round(total_score, 2),
        "totalMaxPoints": total_max,
        "pages": student_pages,
    }


# ══════════════════════════════════════════════════════════════════
#  Top-level entry point
# ══════════════════════════════════════════════════════════════════

async def evaluate_exam(pdf_bytes: bytes, exam_map: Dict) -> Dict[str, Any]:
    """
    Run the full exam evaluation pipeline.
    Returns a dict ready to ship to the frontend + drop into Excel.
    """
    exam_id = exam_map.get("examId", "unknown")
    map_pages = exam_map.get("pages", []) or []
    pages_per_exam = int(exam_map.get("totalPages", 1) or 1)
    question_nums: List[str] = []
    for page in map_pages:
        question_nums.extend(page.get("questions", {}).keys())

    print(f"[Pipeline] Exam: {exam_id}, {pages_per_exam} pages/student, {len(question_nums)} questions")

    # 1. PDF → images
    images = pdf_to_images(pdf_bytes)
    print(f"[Pipeline] PDF: {len(images)} pages")

    # 2. Split by students (QR or sequential fallback)
    students = split_by_students(images, exam_map, pages_per_exam)
    total_students = len(students)
    print(f"[Pipeline] Students: {total_students}")
    # Batch boundary — renderer uses sTotal to size the progress bars.
    print(f"[STAGE] stage=batch state=start sTotal={total_students}")
    del images

    # 3. Evaluate each student
    all_student_results: List[Dict[str, Any]] = []
    for i, student in enumerate(students):
        try:
            student_data = await evaluate_student(
                student, map_pages, i, student_total=total_students,
            )
        except Exception as e:
            traceback.print_exc()
            student_data = {
                "studentNumber": getattr(student, "student_number", f"error_{i + 1}"),
                "studentNumberConfidence": 0.0,
                "studentNumberImage": "",
                "questions": {},
                "totalScore": 0,
                "totalMaxPoints": 0,
                "error": str(e)[:300],
            }
            print(f"[STAGE] sIdx={i} sTotal={total_students} stage=error")
        all_student_results.append(student_data)
        await asyncio.sleep(0)  # yield between students

    # 4. Excel export
    print(f"[STAGE] stage=save state=start")
    excel_path = export.export_results(
        exam_id, question_nums, all_student_results, config.OUTPUT_DIR,
    )
    print(f"[STAGE] stage=save state=end excel={excel_path}")
    print(f"[STAGE] stage=batch state=end sTotal={total_students}")

    print(f"[Pipeline] Done. {total_students} students -> {excel_path}")

    return {
        "examId": exam_id,
        "totalStudents": total_students,
        "totalQuestions": len(question_nums),
        "aiEnabled": config.AI_ENABLED,
        "excelPath": excel_path,
        "students": all_student_results,
    }
