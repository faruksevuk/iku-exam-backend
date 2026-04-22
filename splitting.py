"""
Splitting — take a multi-student scanned PDF and divide it into per-student exams.

Strategy:
  1. Convert every PDF page to a BGR image (PyMuPDF @ configured DPI)
  2. Read the QR code on each page
       Generator encodes: "P1_<examId>", "P2_<examId>", ...
  3. "P1" starts a new student; "P2+" or no-QR continues the current student
  4. Fallback: if no QR ever detected, split every `pages_per_exam` pages
       (pages_per_exam comes from the map's `totalPages`)

Student number OCR is NOT done here — the pipeline reads it later from
the aligned page-1 using digit CNN.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np

import config


@dataclass
class StudentExam:
    """One student's complete exam (all their pages)."""
    student_number: str = ""
    student_number_confidence: float = 0.0
    student_number_image: str = ""  # base64 jpeg crop for review UI
    pages: List[Dict] = field(default_factory=list)
    # Each page: {"pdfPageIndex": int, "examPageNum": int,
    #             "image": ndarray, "qrText": Optional[str]}


# ── PDF → images ─────────────────────────────────────────────────

def pdf_to_images(pdf_bytes: bytes, dpi: Optional[int] = None) -> List[np.ndarray]:
    """Convert every page of a PDF to a BGR numpy image."""
    resolved_dpi = config.SCAN_DPI if dpi is None else dpi
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: List[np.ndarray] = []
    try:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=resolved_dpi, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 3:
                img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                img = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            images.append(img)
    finally:
        doc.close()
    return images


# ── QR reading ───────────────────────────────────────────────────

_qr_detector = cv2.QRCodeDetector()


def read_qr_from_image(image: np.ndarray) -> Optional[str]:
    """Return the QR code text from a page image, or None."""
    try:
        data, _, _ = _qr_detector.detectAndDecode(image)
    except Exception:
        return None
    if data:
        return data.strip()
    return None


def parse_page_id(qr_text: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
    """
    Parse QR text like 'P1_COM101_final_2026' → (1, 'COM101_final_2026').
    Returns (None, None) on failure.
    """
    if not qr_text or not qr_text.startswith("P"):
        return (None, None)
    parts = qr_text.split("_", 1)
    if len(parts) < 2:
        return (None, None)
    try:
        page_num = int(parts[0][1:])
        exam_code = parts[1]
        return (page_num, exam_code)
    except (ValueError, IndexError):
        return (None, None)


# ── Student splitting ────────────────────────────────────────────

def split_by_students(
    images: List[np.ndarray],
    exam_map: Dict,
    pages_per_exam: Optional[int] = None,
) -> List[StudentExam]:
    """
    Divide a multi-student scanned PDF into individual StudentExam objects.

    Primary: QR-based (P1 = new student).
    Fallback: sequential chunks of `pages_per_exam` pages.
    """
    if pages_per_exam is None:
        pages_per_exam = int(exam_map.get("totalPages", 1) or 1)

    students: List[StudentExam] = []
    current: Optional[StudentExam] = None

    for i, img in enumerate(images):
        qr_text = read_qr_from_image(img)
        page_num, _exam_code = parse_page_id(qr_text)

        if config.VERBOSE:
            print(f"[Split] Page {i + 1}: QR='{qr_text}' -> pageNum={page_num}")

        is_new_student = False
        if page_num == 1:
            is_new_student = True
        elif page_num is None and current is None:
            is_new_student = True
        elif page_num is None and current is not None:
            if len(current.pages) >= pages_per_exam:
                is_new_student = True

        if is_new_student:
            if current is not None:
                students.append(current)
            current = StudentExam()

        if current is not None:
            current.pages.append({
                "pdfPageIndex": i,
                "examPageNum": page_num or (len(current.pages) + 1),
                "image": img,
                "qrText": qr_text,
            })

    if current is not None and current.pages:
        students.append(current)

    # Fallback: no QR anywhere → sequential split
    if not students and images:
        print("[Split] WARN: No QR codes found - falling back to sequential split")
        for start in range(0, len(images), pages_per_exam):
            chunk = images[start:start + pages_per_exam]
            student = StudentExam(pages=[
                {
                    "pdfPageIndex": start + j,
                    "examPageNum": j + 1,
                    "image": img,
                    "qrText": None,
                }
                for j, img in enumerate(chunk)
            ])
            students.append(student)

    print(f"[Split] Found {len(students)} students")
    return students
