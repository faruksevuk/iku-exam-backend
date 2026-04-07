"""
IKU Exam Evaluation Backend — FastAPI

Full pipeline:
  1. Receive scanned PDF (multi-student) + JSON map
  2. Split PDF by student (QR-based page grouping)
  3. Align each page (bullseye perspective correction)
  4. Read student numbers (TrOCR on digit boxes)
  5. Evaluate each question:
     - MC/MS: OMR bubble fill detection
     - Match/Fill: TrOCR + fuzzy matching
     - Open: crop solution area (AI evaluation later)
  6. Score all questions
  7. Export results as Excel
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import json
import os
from typing import Dict, Any, List

from scanner import align_page, extract_region
from omr import evaluate_mc, evaluate_ms
from ocr import read_handwriting, fuzzy_match
from char_reader import read_student_number, read_answer_box_letter, read_answer_box_text, crop_answer_region
from scoring import score_mc, score_ms, score_match, score_fill, score_open
from pdf_splitter import pdf_to_images, split_by_students
from excel_export import export_results

app = FastAPI(title="IKU Exam Evaluator", version="0.3.0")

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_student(
    student_pages: List[Dict],
    exam_map: Dict,
    student_label: str,
) -> Dict[str, Any]:
    """
    Evaluate one student's complete exam.
    student_pages: list of {examPageNum, image, ...}
    exam_map: the full JSON map with pages, questions, anchors
    """
    map_pages = exam_map.get("pages", [])
    all_question_results = {}

    for page_info in student_pages:
        exam_page_num = page_info["examPageNum"]
        image = page_info["image"]

        # Find the corresponding map page (0-indexed)
        map_page_idx = exam_page_num - 1
        if map_page_idx >= len(map_pages):
            print(f"[Eval] Student {student_idx}: page {exam_page_num} has no map data, skipping")
            continue

        page_data = map_pages[map_page_idx]
        anchors = page_data.get("anchors", {})
        page_w = int(page_data.get("pageWidth", 756))
        page_h = int(page_data.get("pageHeight", 1086))

        # Align the page
        aligned = align_page(image, anchors, page_width=page_w, page_height=page_h)

        # Save aligned image
        aligned_path = os.path.join(OUTPUT_DIR, f"{student_label}_page{exam_page_num}.jpg")
        import cv2
        cv2.imwrite(aligned_path, aligned)

        # Evaluate each question on this page
        questions = page_data.get("questions", {})
        for q_num, q_data in questions.items():
            q_type = q_data.get("type", "unknown")
            scoring = q_data.get("scoring", {})
            expected = q_data.get("expectedAnswer", {})
            options = q_data.get("options", {})

            result: Dict[str, Any] = {
                "type": q_type,
                "maxPoints": scoring.get("points", 0),
                "alignedImagePath": aligned_path,
            }

            # Crop the question's bounding box as base64 for review screen
            bbox = q_data.get("boundingBox")
            if bbox:
                result["answerImage"] = crop_answer_region(aligned, bbox)

            # Include expected answer for review
            result["expected"] = expected

            if q_type == "multiple_choice":
                omr_result = evaluate_mc(aligned, options, expected.get("correctOption"))
                score_result = score_mc(omr_result, scoring, expected)
                result.update(omr_result)
                result.update(score_result)

            elif q_type == "multi_select":
                omr_result = evaluate_ms(aligned, options, expected.get("correctOptions"))
                score_result = score_ms(omr_result, scoring, expected)
                result.update(omr_result)
                result.update(score_result)

            elif q_type == "matching":
                answer_boxes = q_data.get("answerBoxes", {})
                correct_matches = expected.get("correctMatches", {})
                ocr_answers = {}
                confidences = {}
                for idx, box in answer_boxes.items():
                    exp_val = correct_matches.get(idx, "")
                    if len(exp_val) <= 2:
                        text, conf = read_answer_box_letter(aligned, box)
                    else:
                        text, conf = read_answer_box_text(aligned, box)
                    ocr_answers[idx] = text
                    confidences[idx] = conf
                score_result = score_match(ocr_answers, expected, scoring, confidences)
                result.update(score_result)
                result["ocrAnswers"] = ocr_answers
                result["ocrConfidences"] = confidences

            elif q_type == "fill_blanks":
                answer_boxes = q_data.get("answerBoxes", {})
                correct_blanks = expected.get("correctBlanks", {})
                ocr_answers = {}
                confidences = {}
                for idx, box in answer_boxes.items():
                    exp_val = correct_blanks.get(idx, "")
                    if len(exp_val) <= 2:
                        text, conf = read_answer_box_letter(aligned, box)
                    else:
                        text, conf = read_answer_box_text(aligned, box)
                    ocr_answers[idx] = text
                    confidences[idx] = conf
                score_result = score_fill(ocr_answers, expected, scoring, confidences)
                result.update(score_result)
                result["ocrAnswers"] = ocr_answers
                result["ocrConfidences"] = confidences

            elif q_type == "open_ended":
                solution_box = q_data.get("solutionArea")
                if solution_box:
                    result["solutionAreaImage"] = crop_answer_region(aligned, solution_box)
                score_result = score_open(scoring)
                result.update(score_result)
                # Include expected answer text for review
                if expected.get("text"):
                    result["expectedText"] = expected["text"]

            # Flag low confidence for review
            conf = result.get("confidence", 1.0)
            if conf < 0.80 or result.get("needsReview"):
                result["needsReview"] = True

            all_question_results[q_num] = result

    # Compute totals
    total_score = sum(r.get("score", 0) for r in all_question_results.values())
    total_max = sum(r.get("maxPoints", 0) for r in all_question_results.values())

    return {
        "questions": all_question_results,
        "totalScore": round(total_score, 2),
        "totalMaxPoints": total_max,
    }


# ══════════════════════════════════════════════════════════
#  API Endpoints
# ══════════════════════════════════════════════════════════

@app.post("/evaluate")
async def evaluate_exam(
    pdf_file: UploadFile = File(...),
    map_file: UploadFile = File(...),
):
    """
    Full evaluation pipeline.
    Accepts a multi-student scanned PDF + JSON map.
    Returns per-student, per-question scores with confidence.
    Also generates an Excel file.
    """
    try:
        map_content = await map_file.read()
        exam_map = json.loads(map_content)
        pdf_content = await pdf_file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read files: {e}")

    exam_id = exam_map.get("examId", "unknown")
    pages_per_exam = exam_map.get("totalPages", 1)
    question_nums = []
    for page in exam_map.get("pages", []):
        question_nums.extend(page.get("questions", {}).keys())

    print(f"[Main] Exam: {exam_id}, {pages_per_exam} pages, {len(question_nums)} questions")

    # Step 1: Convert PDF to images
    images = pdf_to_images(pdf_content)
    print(f"[Main] Scanned PDF: {len(images)} pages")

    # Step 2: Split by students
    students = split_by_students(images, exam_map, pages_per_exam)
    print(f"[Main] Found {len(students)} students")

    # Step 3: Evaluate each student
    all_student_results = []

    for i, student in enumerate(students):
        print(f"\n[Main] === Evaluating student {i+1}/{len(students)} ===")

        # Read student number from first page (P1)
        first_page = student.pages[0] if student.pages else None
        if first_page:
            map_page = exam_map["pages"][0] if exam_map.get("pages") else {}
            anchors = map_page.get("anchors", {})
            pw = int(map_page.get("pageWidth", 756))
            ph = int(map_page.get("pageHeight", 1086))
            aligned_p1 = align_page(first_page["image"], anchors, page_width=pw, page_height=ph)

            # Read student number: per-digit-box with MNIST CNN
            sn_boxes = map_page.get("studentNumberBoxes", [])
            if sn_boxes:
                sn, sn_conf = read_student_number(aligned_p1, sn_boxes)
                # Also crop the student number region as base64 for review
                min_x = min(b["x"] for b in sn_boxes) - 4
                min_y = min(b["y"] for b in sn_boxes) - 4
                max_x = max(b["x"] + b["w"] for b in sn_boxes) + 4
                max_y = max(b["y"] + b["h"] for b in sn_boxes) + 4
                sn_region = {"x": min_x, "y": min_y, "w": max_x - min_x, "h": max_y - min_y}
                student.student_number_image = crop_answer_region(aligned_p1, sn_region)
            else:
                sn, sn_conf = None, 0.0
                student.student_number_image = ""

            if sn:
                student.student_number = sn
                student.student_number_confidence = sn_conf
            else:
                student.student_number = f"unknown_{i+1}"
                student.student_number_confidence = 0.0
            print(f"[Main] Student: {student.student_number} (conf={student.student_number_confidence:.2f})")

        # Use student number for file naming
        sn_label = student.student_number.replace("?", "X")

        # Evaluate all questions
        eval_result = evaluate_student(student.pages, exam_map, sn_label)

        student_data = {
            "studentNumber": student.student_number,
            "studentNumberConfidence": student.student_number_confidence,
            "studentNumberImage": student.student_number_image,
            **eval_result,
        }
        all_student_results.append(student_data)

    # Step 4: Export Excel
    excel_path = export_results(exam_id, question_nums, all_student_results, OUTPUT_DIR)

    return {
        "examId": exam_id,
        "totalStudents": len(all_student_results),
        "totalQuestions": len(question_nums),
        "excelPath": excel_path,
        "students": all_student_results,
    }


@app.post("/align")
async def align_only(
    pdf_file: UploadFile = File(...),
    map_file: UploadFile = File(...),
):
    """Debug: align pages only."""
    map_content = await map_file.read()
    exam_map = json.loads(map_content)
    pdf_content = await pdf_file.read()
    images = pdf_to_images(pdf_content)
    pages = exam_map.get("pages", [])
    saved = []

    import cv2
    for i, (img, page_data) in enumerate(zip(images, pages)):
        anchors = page_data.get("anchors", {})
        pw = int(page_data.get("pageWidth", 756))
        ph = int(page_data.get("pageHeight", 1086))
        aligned = align_page(img, anchors, page_width=pw, page_height=ph)
        path = os.path.join(OUTPUT_DIR, f"aligned_page_{i+1}.jpg")
        cv2.imwrite(path, aligned)
        saved.append(path)

    return {"alignedPages": saved}


@app.get("/results/{exam_id}/excel")
async def download_excel(exam_id: str):
    """Download the generated Excel file."""
    path = os.path.join(OUTPUT_DIR, f"{exam_id}_results.xlsx")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Excel not found. Run /evaluate first.")
    return FileResponse(path, filename=f"{exam_id}_results.xlsx")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
