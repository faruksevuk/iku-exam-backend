"""
Database service layer — bridges pipeline output → SQLAlchemy ORM.

The pipeline writes its results JSON file as the source of truth, then
calls `save_evaluation_result()` here best-effort. A failure in this
layer must NOT crash the pipeline — the JSON file is enough to recover
the run.

Re-runs of the same exam_id wipe the old exam row + cascade to all
children. Student rows are preserved across re-runs.
"""
import json
from typing import Any, Dict, Optional

from sqlalchemy import inspect, text

import config
from database import DATABASE_PATH, SessionLocal, engine, init_db
from db_models import (
    AuditLog,
    Exam,
    ExamResult,
    FinalApproval,
    LlmEvaluation,
    OcrOutput,
    QuestionResult,
    Student,
    TeacherOverride,
)


# Base64 image payloads bloat the DB and aren't queryable. Strip any
# field whose key signals an image before persisting.
_IMAGE_KEYS = {
    "studentNumberImage",
    "answerImage",
    "solutionAreaImage",
}


def _strip_large_fields(obj: Any) -> Any:
    """Recursively drop base64 image fields from a result payload."""
    if isinstance(obj, dict):
        return {
            k: _strip_large_fields(v)
            for k, v in obj.items()
            if k not in _IMAGE_KEYS and not k.lower().endswith("image")
        }
    if isinstance(obj, list):
        return [_strip_large_fields(x) for x in obj]
    return obj


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _confidence_from_question(q: Dict[str, Any]) -> float:
    if q.get("confidence") is not None:
        return _safe_float(q.get("confidence"))

    confs = q.get("ocrConfidences")
    if isinstance(confs, dict) and confs:
        vals = [_safe_float(v) for v in confs.values()]
        return sum(vals) / len(vals)

    return 0.0


def _needs_review_from_question(q: Dict[str, Any]) -> bool:
    if q.get("needsReview") is True:
        return True
    return q.get("status") in {"pending_review", "ai_flagged", "error"}


def _get_or_create_student(db, student_number: str) -> Student:
    student = (
        db.query(Student)
        .filter(Student.student_number == student_number)
        .one_or_none()
    )

    if student:
        return student

    student = Student(
        student_number=student_number,
        display_name=f"Student {student_number}",
    )
    db.add(student)
    db.flush()
    return student


def save_evaluation_result(
    exam_map: Dict[str, Any],
    result_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist a /evaluate result payload to SQLite.

    Same exam_id re-runs delete the prior Exam row (cascading children)
    before inserting fresh data. Students are preserved across runs.
    """
    init_db()
    db = SessionLocal()

    try:
        exam_id = str(
            result_payload.get("examId")
            or exam_map.get("examId")
            or "unknown"
        )

        existing_exam = (
            db.query(Exam)
            .filter(Exam.exam_id == exam_id)
            .one_or_none()
        )
        if existing_exam:
            db.delete(existing_exam)
            db.commit()

        sanitized_payload = _strip_large_fields(result_payload)

        exam = Exam(
            exam_id=exam_id,
            total_students=int(result_payload.get("totalStudents") or 0),
            total_questions=int(result_payload.get("totalQuestions") or 0),
            ai_enabled=bool(result_payload.get("aiEnabled")),
            excel_path=result_payload.get("excelPath"),
            generated_at=result_payload.get("generatedAt"),
            raw_payload_json=sanitized_payload,
        )
        db.add(exam)
        db.flush()

        students = result_payload.get("students") or []

        for student_result in students:
            student_number = str(
                student_result.get("studentNumber")
                or "unknown"
            )

            student = _get_or_create_student(db, student_number)

            questions = student_result.get("questions") or {}
            student_needs_review = any(
                _needs_review_from_question(q)
                for q in questions.values()
                if isinstance(q, dict)
            )

            exam_result = ExamResult(
                exam_id=exam.id,
                student_id=student.id,
                student_number_snapshot=student_number,
                total_score=_safe_float(student_result.get("totalScore")),
                total_max_points=_safe_float(student_result.get("totalMaxPoints")),
                needs_review=student_needs_review,
                raw_result_json=_strip_large_fields(student_result),
            )
            db.add(exam_result)
            db.flush()

            for q_num, q in questions.items():
                if not isinstance(q, dict):
                    continue

                q_type = str(q.get("type") or "unknown")
                status = q.get("status")
                confidence = _confidence_from_question(q)

                q_result = QuestionResult(
                    exam_result_id=exam_result.id,
                    question_number=str(q_num),
                    question_type=q_type,
                    score=_safe_float(q.get("score")),
                    max_points=_safe_float(q.get("maxPoints")),
                    status=status,
                    confidence=confidence,
                    needs_review=_needs_review_from_question(q),
                    is_correct=(status == "correct"),
                    expected_json=q.get("expected"),
                    raw_result_json=_strip_large_fields(q),
                )
                db.add(q_result)
                db.flush()

                # MC / MS — record the selected option as a synthetic OCR output.
                if q_type in {"multiple_choice", "multi_select"}:
                    selected = q.get("selected")
                    if isinstance(selected, (dict, list)):
                        selected_text = json.dumps(selected, ensure_ascii=False)
                    else:
                        selected_text = "" if selected is None else str(selected)

                    db.add(
                        OcrOutput(
                            question_result_id=q_result.id,
                            output_type=q_type,
                            text=selected_text,
                            confidence=confidence,
                            payload_json={
                                "selected": selected,
                                "status": status,
                                "is_correct": status == "correct",
                            },
                        )
                    )

                # Matching / fill_blanks per-cell OCR results.
                ocr_answers = q.get("ocrAnswers")
                if isinstance(ocr_answers, dict):
                    confs = q.get("ocrConfidences") or {}
                    for item_key, text_value in ocr_answers.items():
                        db.add(
                            OcrOutput(
                                question_result_id=q_result.id,
                                output_type=f"{q_type}:{item_key}",
                                text="" if text_value is None else str(text_value),
                                confidence=_safe_float(confs.get(item_key)),
                                payload_json={
                                    "item": item_key,
                                    "source": (q.get("ocrSources") or {}).get(item_key),
                                },
                            )
                        )

                # Open-ended — record the transcribed student text + LLM trace.
                if q_type == "open_ended":
                    db.add(
                        OcrOutput(
                            question_result_id=q_result.id,
                            output_type="open_ended_text",
                            text=q.get("studentReadText") or "",
                            confidence=confidence,
                            payload_json={
                                "expectedText": q.get("expectedText"),
                                "status": status,
                            },
                        )
                    )

                    reasoning = {
                        "status": status,
                        "confidence": confidence,
                        "needsReview": _needs_review_from_question(q),
                        "studentReadText": q.get("studentReadText"),
                        "expectedText": q.get("expectedText"),
                        "explanation": q.get("explanation"),
                    }

                    db.add(
                        LlmEvaluation(
                            question_result_id=q_result.id,
                            model_name=config.GRADING_MODEL,
                            score=_safe_float(q.get("score")),
                            explanation=q.get("explanation"),
                            reasoning_json=reasoning,
                        )
                    )

        db.commit()

        return {
            "ok": True,
            "databasePath": DATABASE_PATH,
            "examId": exam_id,
            "studentsSaved": len(students),
        }

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def db_health() -> Dict[str, Any]:
    """Return per-table row counts. Used by GET /db/health."""
    init_db()
    db = SessionLocal()

    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        counts = {}
        for table_name in tables:
            counts[table_name] = db.execute(
                text(f'SELECT COUNT(*) FROM "{table_name}"')
            ).scalar()

        return {
            "status": "ok",
            "databasePath": DATABASE_PATH,
            "tables": tables,
            "counts": counts,
        }
    finally:
        db.close()


def list_exams() -> list[dict]:
    """All exams, newest first."""
    init_db()
    db = SessionLocal()

    try:
        exams = (
            db.query(Exam)
            .order_by(Exam.created_at.desc())
            .all()
        )

        return [
            {
                "id": e.id,
                "examId": e.exam_id,
                "totalStudents": e.total_students,
                "totalQuestions": e.total_questions,
                "aiEnabled": e.ai_enabled,
                "excelPath": e.excel_path,
                "generatedAt": e.generated_at,
                "createdAt": e.created_at.isoformat() if e.created_at else None,
            }
            for e in exams
        ]
    finally:
        db.close()


def get_exam_results(exam_id: str) -> Optional[dict]:
    """Full result tree for one exam — students × questions × OCR/LLM."""
    init_db()
    db = SessionLocal()

    try:
        exam = (
            db.query(Exam)
            .filter(Exam.exam_id == exam_id)
            .one_or_none()
        )

        if not exam:
            return None

        students_payload = []

        for er in exam.results:
            question_payload = []

            for qr in er.question_results:
                question_payload.append(
                    {
                        "questionResultId": qr.id,
                        "questionNumber": qr.question_number,
                        "questionType": qr.question_type,
                        "score": qr.score,
                        "maxPoints": qr.max_points,
                        "status": qr.status,
                        "confidence": qr.confidence,
                        "needsReview": qr.needs_review,
                        "isCorrect": qr.is_correct,
                        "ocrOutputs": [
                            {
                                "id": o.id,
                                "type": o.output_type,
                                "text": o.text,
                                "confidence": o.confidence,
                            }
                            for o in qr.ocr_outputs
                        ],
                        "llmEvaluation": None if not qr.llm_evaluation else {
                            "id": qr.llm_evaluation.id,
                            "model": qr.llm_evaluation.model_name,
                            "score": qr.llm_evaluation.score,
                            "explanation": qr.llm_evaluation.explanation,
                            "reasoningJson": qr.llm_evaluation.reasoning_json,
                        },
                    }
                )

            students_payload.append(
                {
                    "examResultId": er.id,
                    "studentId": er.student_id,
                    "studentNumber": er.student_number_snapshot,
                    "totalScore": er.total_score,
                    "totalMaxPoints": er.total_max_points,
                    "needsReview": er.needs_review,
                    "finalApproval": None if not er.final_approval else {
                        "id": er.final_approval.id,
                        "status": er.final_approval.status,
                        "finalScore": er.final_approval.final_score,
                        "approvedBy": er.final_approval.approved_by,
                    },
                    "questions": question_payload,
                }
            )

        return {
            "examId": exam.exam_id,
            "totalStudents": exam.total_students,
            "totalQuestions": exam.total_questions,
            "students": students_payload,
        }
    finally:
        db.close()


def apply_teacher_override(
    question_result_id: int,
    new_score: float,
    reason: str = "Teacher override",
    teacher: str = "teacher",
) -> Optional[dict]:
    """Overwrite a question's score, audit-log the change, recompute exam total."""
    init_db()
    db = SessionLocal()

    try:
        qr = db.query(QuestionResult).filter(QuestionResult.id == question_result_id).one_or_none()
        if not qr:
            return None

        old_score = _safe_float(qr.score)
        new_score = _safe_float(new_score)

        override = TeacherOverride(
            question_result_id=qr.id,
            old_score=old_score,
            new_score=new_score,
            reason=reason,
            created_by=teacher,
        )
        db.add(override)

        old_value = {
            "score": old_score,
            "status": qr.status,
            "needsReview": qr.needs_review,
        }

        qr.score = new_score
        qr.status = "overridden"
        qr.needs_review = False

        new_value = {
            "score": new_score,
            "status": qr.status,
            "needsReview": qr.needs_review,
            "reason": reason,
        }

        db.add(
            AuditLog(
                table_name="question_results",
                row_id=qr.id,
                action="teacher_override",
                old_value_json=old_value,
                new_value_json=new_value,
                created_by=teacher,
            )
        )

        er = qr.exam_result
        er.total_score = sum(_safe_float(q.score) for q in er.question_results)
        er.needs_review = any(bool(q.needs_review) for q in er.question_results)

        db.commit()

        return {
            "ok": True,
            "questionResultId": qr.id,
            "oldScore": old_score,
            "newScore": new_score,
            "examResultId": er.id,
            "newExamTotalScore": er.total_score,
        }

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def approve_exam_result(
    exam_result_id: int,
    status: str = "approved",
    final_score: Optional[float] = None,
    approved_by: str = "teacher",
) -> Optional[dict]:
    """Mark an ExamResult as final-approved, audit-log the action."""
    init_db()
    db = SessionLocal()

    try:
        er = db.query(ExamResult).filter(ExamResult.id == exam_result_id).one_or_none()
        if not er:
            return None

        approval = (
            db.query(FinalApproval)
            .filter(FinalApproval.exam_result_id == exam_result_id)
            .one_or_none()
        )

        old_value = None
        if approval:
            old_value = {
                "status": approval.status,
                "finalScore": approval.final_score,
                "approvedBy": approval.approved_by,
            }
        else:
            approval = FinalApproval(exam_result_id=exam_result_id)
            db.add(approval)

        approval.status = status
        approval.final_score = _safe_float(final_score, er.total_score)
        approval.approved_by = approved_by

        if status == "approved":
            er.needs_review = False

        new_value = {
            "status": approval.status,
            "finalScore": approval.final_score,
            "approvedBy": approval.approved_by,
        }

        db.add(
            AuditLog(
                table_name="final_approvals",
                row_id=exam_result_id,
                action="final_approval",
                old_value_json=old_value,
                new_value_json=new_value,
                created_by=approved_by,
            )
        )

        db.commit()

        return {
            "ok": True,
            "examResultId": exam_result_id,
            "status": approval.status,
            "finalScore": approval.final_score,
            "approvedBy": approval.approved_by,
        }

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def seed_demo_data() -> dict:
    """Populate a small DEMO_EXAM_001 dataset without running the PDF pipeline."""
    demo_payload = {
        "examId": "DEMO_EXAM_001",
        "totalStudents": 2,
        "totalQuestions": 3,
        "aiEnabled": True,
        "excelPath": "output/DEMO_EXAM_001_results.xlsx",
        "generatedAt": "2026-05-11T00:00:00Z",
        "students": [
            {
                "studentNumber": "DEMO2025001",
                "studentNumberConfidence": 0.97,
                "questions": {
                    "1": {
                        "type": "multiple_choice",
                        "selected": "A",
                        "expected": {"correctOption": "A"},
                        "score": 10,
                        "maxPoints": 10,
                        "status": "correct",
                        "confidence": 0.96,
                        "needsReview": False,
                    },
                    "2": {
                        "type": "fill_blanks",
                        "ocrAnswers": {"1": "photosynthesis"},
                        "ocrConfidences": {"1": 0.88},
                        "expected": {"correctBlanks": {"1": "photosynthesis"}},
                        "score": 10,
                        "maxPoints": 10,
                        "status": "correct",
                        "confidence": 0.88,
                        "needsReview": False,
                    },
                    "3": {
                        "type": "open_ended",
                        "studentReadText": "It converts light energy into chemical energy.",
                        "expectedText": "Photosynthesis converts light energy into chemical energy.",
                        "score": 7,
                        "maxPoints": 10,
                        "status": "partial",
                        "confidence": 0.72,
                        "needsReview": True,
                        "explanation": "The answer captures the core idea but lacks detail about glucose and oxygen.",
                    },
                },
                "totalScore": 27,
                "totalMaxPoints": 30,
                "pages": [],
            },
            {
                "studentNumber": "DEMO2025002",
                "studentNumberConfidence": 0.93,
                "questions": {
                    "1": {
                        "type": "multiple_choice",
                        "selected": "C",
                        "expected": {"correctOption": "A"},
                        "score": 0,
                        "maxPoints": 10,
                        "status": "wrong",
                        "confidence": 0.91,
                        "needsReview": False,
                    },
                    "2": {
                        "type": "fill_blanks",
                        "ocrAnswers": {"1": "photosinthesis"},
                        "ocrConfidences": {"1": 0.64},
                        "expected": {"correctBlanks": {"1": "photosynthesis"}},
                        "score": 6,
                        "maxPoints": 10,
                        "status": "partial",
                        "confidence": 0.64,
                        "needsReview": True,
                    },
                    "3": {
                        "type": "open_ended",
                        "studentReadText": "Plants make food.",
                        "expectedText": "Photosynthesis converts light energy into chemical energy.",
                        "score": 4,
                        "maxPoints": 10,
                        "status": "partial",
                        "confidence": 0.61,
                        "needsReview": True,
                        "explanation": "The response is broadly related but too vague for full credit.",
                    },
                },
                "totalScore": 10,
                "totalMaxPoints": 30,
                "pages": [],
            },
        ],
    }

    save_info = save_evaluation_result(
        exam_map={"examId": "DEMO_EXAM_001"},
        result_payload=demo_payload,
    )

    return {
        "seed": save_info,
        "results": get_exam_results("DEMO_EXAM_001"),
    }
