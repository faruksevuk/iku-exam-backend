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
"""

import asyncio
import os
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
        ocr_answers: Dict[str, str] = {}
        confidences: Dict[str, float] = {}
        review_flags: Dict[str, bool] = {}
        for idx, box in answer_boxes.items():
            reader_res = handwriting.read_letter_box(aligned, box)
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
        ocr_answers = {}
        confidences = {}
        review_flags = {}
        sources: Dict[str, str] = {}
        for idx, box in answer_boxes.items():
            reader_res = handwriting.read_text_box(aligned, box)
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
            results[q_num] = await evaluate_question(aligned, q_data, q_num)
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
) -> Dict[str, Any]:
    """Align every page, read the student number, evaluate every question."""
    aligned_cache: Dict[int, np.ndarray] = {}

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

    # Evaluate every page for this student
    all_q_results: Dict[str, Any] = {}
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

        # Save aligned page image for debugging
        try:
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            cv2.imwrite(
                os.path.join(config.OUTPUT_DIR, f"{sn_label}_page{exam_page_num}.jpg"),
                aligned,
            )
        except Exception:
            pass

        page_results = await evaluate_student_page(aligned, map_pages[map_idx])
        all_q_results.update(page_results)

    total_score = sum(r.get("score", 0) for r in all_q_results.values())
    total_max = sum(r.get("maxPoints", 0) for r in all_q_results.values())

    return {
        "studentNumber": student.student_number,
        "studentNumberConfidence": student.student_number_confidence,
        "studentNumberImage": student.student_number_image,
        "questions": all_q_results,
        "totalScore": round(total_score, 2),
        "totalMaxPoints": total_max,
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
    del images

    # 3. Evaluate each student
    all_student_results: List[Dict[str, Any]] = []
    for i, student in enumerate(students):
        try:
            student_data = await evaluate_student(student, map_pages, i)
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
        all_student_results.append(student_data)
        await asyncio.sleep(0)  # yield between students

    # 4. Excel export
    excel_path = export.export_results(
        exam_id, question_nums, all_student_results, config.OUTPUT_DIR,
    )

    print(f"[Pipeline] Done. {total_students} students -> {excel_path}")

    return {
        "examId": exam_id,
        "totalStudents": total_students,
        "totalQuestions": len(question_nums),
        "aiEnabled": config.AI_ENABLED,
        "excelPath": excel_path,
        "students": all_student_results,
    }
