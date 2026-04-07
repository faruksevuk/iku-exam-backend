"""
PDF Splitter — Groups scanned exam pages by student.

Logic:
1. Convert each PDF page to an image
2. Read QR code from each page to get page ID (P1_COM101_final_2026)
3. When P1 is encountered, read the student number from that page
4. Group consecutive pages: P1=new student, P2/P3/etc continue same student
5. Return list of StudentExam objects with page images and student numbers

Falls back to sequential grouping if QR reading fails.
"""

import cv2
import numpy as np
import fitz
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class StudentExam:
    """One student's complete exam (all pages)."""
    student_number: str
    student_number_confidence: float
    student_number_image: str = ""
    pages: List[Dict] = field(default_factory=list)  # [{pageIndex, image, pageId}]


def pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> List[np.ndarray]:
    """Convert all pages of a PDF to BGR images."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 3:
            img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            img = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        images.append(img)
    return images


def read_qr_from_image(image: np.ndarray) -> Optional[str]:
    """Read QR code text from a scanned page image."""
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(image)
    if data:
        return data.strip()
    return None


def parse_page_id(qr_text: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Parse QR text like 'P1_COM101_final_2026' into (page_number, exam_code).
    Returns (1, 'COM101_final_2026') or (None, None) on failure.
    """
    if not qr_text or not qr_text.startswith("P"):
        return (None, None)

    parts = qr_text.split("_", 1)
    if len(parts) < 2:
        return (None, None)

    try:
        page_num = int(parts[0][1:])  # "P1" -> 1
        exam_code = parts[1]  # "COM101_final_2026"
        return (page_num, exam_code)
    except (ValueError, IndexError):
        return (None, None)


def split_by_students(
    images: List[np.ndarray],
    exam_map: Dict,
    pages_per_exam: int,
) -> List[StudentExam]:
    """
    Split a multi-student scanned PDF into individual student exams.

    Strategy:
    1. Try QR-based splitting first (P1 = new student)
    2. Fallback: split every `pages_per_exam` pages

    Student number OCR is done later in the evaluation pipeline.
    """
    students: List[StudentExam] = []
    current: Optional[StudentExam] = None

    for i, img in enumerate(images):
        qr_text = read_qr_from_image(img)
        page_num, exam_code = parse_page_id(qr_text) if qr_text else (None, None)

        print(f"[Splitter] Page {i+1}: QR='{qr_text}' -> page={page_num}")

        # P1 or first page -> new student
        is_new_student = False
        if page_num == 1:
            is_new_student = True
        elif page_num is None and current is None:
            # No QR and no current student — treat as new
            is_new_student = True
        elif page_num is None and current is not None:
            # No QR but we have a current student — check if we exceeded expected pages
            if len(current.pages) >= pages_per_exam:
                is_new_student = True

        if is_new_student:
            if current is not None:
                students.append(current)
            current = StudentExam(
                student_number="",
                student_number_confidence=0.0,
                pages=[],
            )

        if current is not None:
            current.pages.append({
                "pdfPageIndex": i,
                "examPageNum": page_num or (len(current.pages) + 1),
                "image": img,
                "qrText": qr_text,
            })

    # Don't forget the last student
    if current is not None and current.pages:
        students.append(current)

    # Fallback: if QR detection failed completely, split by page count
    if len(students) == 0 and len(images) > 0:
        print("[Splitter] QR detection failed. Falling back to sequential split.")
        for i in range(0, len(images), pages_per_exam):
            chunk = images[i:i+pages_per_exam]
            student = StudentExam(
                student_number="",
                student_number_confidence=0.0,
                pages=[{
                    "pdfPageIndex": i + j,
                    "examPageNum": j + 1,
                    "image": img,
                    "qrText": None,
                } for j, img in enumerate(chunk)],
            )
            students.append(student)

    print(f"[Splitter] Found {len(students)} students")
    return students
