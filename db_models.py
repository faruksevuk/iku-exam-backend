"""
SQLAlchemy ORM models for the exam evaluation persistence layer.

Schema overview:

  exams (1) ──┬── (N) exam_results ──┬── (N) question_results ──┬── (N) ocr_outputs
              │                      │                          ├── (1) llm_evaluations
              │                      │                          └── (N) teacher_overrides
              │                      └── (1) final_approvals
              └── (1) students (M:N via exam_results)

  audit_logs is a free-standing journal keyed by (table_name, row_id).

ON DELETE CASCADE: re-running /evaluate for the same exam_id wipes the
old exam row and every downstream child via the cascade, so the run is
idempotent.
"""
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.types import JSON

from database import Base


class Exam(Base):
    __tablename__ = "exams"

    id = Column(Integer, primary_key=True, index=True)
    exam_id = Column(String, unique=True, nullable=False, index=True)
    total_students = Column(Integer, default=0)
    total_questions = Column(Integer, default=0)
    ai_enabled = Column(Boolean, default=False)
    excel_path = Column(String, nullable=True)
    generated_at = Column(String, nullable=True)
    raw_payload_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    results = relationship(
        "ExamResult",
        back_populates="exam",
        cascade="all, delete-orphan",
    )


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    student_number = Column(String, unique=True, nullable=False, index=True)
    display_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    results = relationship("ExamResult", back_populates="student")


class ExamResult(Base):
    __tablename__ = "exam_results"

    id = Column(Integer, primary_key=True, index=True)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)

    student_number_snapshot = Column(String, nullable=False)
    total_score = Column(Float, default=0.0)
    total_max_points = Column(Float, default=0.0)
    needs_review = Column(Boolean, default=False)
    raw_result_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    exam = relationship("Exam", back_populates="results")
    student = relationship("Student", back_populates="results")

    question_results = relationship(
        "QuestionResult",
        back_populates="exam_result",
        cascade="all, delete-orphan",
    )

    final_approval = relationship(
        "FinalApproval",
        back_populates="exam_result",
        uselist=False,
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("exam_id", "student_id", name="uq_exam_student_result"),
    )


class QuestionResult(Base):
    __tablename__ = "question_results"

    id = Column(Integer, primary_key=True, index=True)
    exam_result_id = Column(Integer, ForeignKey("exam_results.id"), nullable=False)

    question_number = Column(String, nullable=False)
    question_type = Column(String, nullable=False)

    score = Column(Float, default=0.0)
    max_points = Column(Float, default=0.0)

    status = Column(String, nullable=True)
    confidence = Column(Float, default=0.0)
    needs_review = Column(Boolean, default=False)
    is_correct = Column(Boolean, default=False)

    expected_json = Column(JSON, nullable=True)
    raw_result_json = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    exam_result = relationship("ExamResult", back_populates="question_results")

    ocr_outputs = relationship(
        "OcrOutput",
        back_populates="question_result",
        cascade="all, delete-orphan",
    )

    llm_evaluation = relationship(
        "LlmEvaluation",
        back_populates="question_result",
        uselist=False,
        cascade="all, delete-orphan",
    )

    overrides = relationship(
        "TeacherOverride",
        back_populates="question_result",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint(
            "exam_result_id",
            "question_number",
            name="uq_exam_result_question",
        ),
    )


class OcrOutput(Base):
    __tablename__ = "ocr_outputs"

    id = Column(Integer, primary_key=True, index=True)
    question_result_id = Column(Integer, ForeignKey("question_results.id"), nullable=False)

    output_type = Column(String, nullable=False)
    text = Column(Text, nullable=True)
    confidence = Column(Float, default=0.0)
    payload_json = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    question_result = relationship("QuestionResult", back_populates="ocr_outputs")


class LlmEvaluation(Base):
    __tablename__ = "llm_evaluations"

    id = Column(Integer, primary_key=True, index=True)
    question_result_id = Column(Integer, ForeignKey("question_results.id"), nullable=False)

    model_name = Column(String, nullable=True)
    score = Column(Float, default=0.0)
    explanation = Column(Text, nullable=True)
    reasoning_json = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    question_result = relationship("QuestionResult", back_populates="llm_evaluation")


class TeacherOverride(Base):
    __tablename__ = "teacher_overrides"

    id = Column(Integer, primary_key=True, index=True)
    question_result_id = Column(Integer, ForeignKey("question_results.id"), nullable=False)

    old_score = Column(Float, nullable=False)
    new_score = Column(Float, nullable=False)
    reason = Column(Text, nullable=True)
    created_by = Column(String, default="teacher")
    created_at = Column(DateTime, default=datetime.utcnow)

    question_result = relationship("QuestionResult", back_populates="overrides")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)

    table_name = Column(String, nullable=False)
    row_id = Column(Integer, nullable=False)
    action = Column(String, nullable=False)

    old_value_json = Column(JSON, nullable=True)
    new_value_json = Column(JSON, nullable=True)

    created_by = Column(String, default="system")
    created_at = Column(DateTime, default=datetime.utcnow)


class FinalApproval(Base):
    __tablename__ = "final_approvals"

    id = Column(Integer, primary_key=True, index=True)
    exam_result_id = Column(Integer, ForeignKey("exam_results.id"), unique=True, nullable=False)

    status = Column(String, default="approved")
    final_score = Column(Float, default=0.0)
    approved_by = Column(String, default="teacher")
    created_at = Column(DateTime, default=datetime.utcnow)

    exam_result = relationship("ExamResult", back_populates="final_approval")
