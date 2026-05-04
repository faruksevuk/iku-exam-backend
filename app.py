"""
FastAPI application - routes only.

Endpoints:
  GET  /health                          service status + AI mode
  POST /evaluate                        full pipeline (pdf + map -> results)
  POST /align                           debug: write aligned pages to disk
  GET  /results/{exam_id}/excel         download generated xlsx
"""

# Force UTF-8 stdout on Windows (otherwise cp1252 chokes on non-ASCII prints).
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

import base64
import json
import os
import re
from typing import Optional

# ── Thread tuning: must run BEFORE torch / opencv import models ──
# PyTorch defaults to a conservative thread count on CPU; large encoder-
# decoder models (TrOCR-large) gain 1.5-3x wallclock from using all cores.
# OMP_NUM_THREADS also affects numpy/BLAS paths used by opencv.
_NUM_THREADS = max(1, (os.cpu_count() or 1))
os.environ.setdefault("OMP_NUM_THREADS", str(_NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_NUM_THREADS))
try:
    import torch
    torch.set_num_threads(_NUM_THREADS)
except Exception:
    pass

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config
import ai_evaluation
import alignment
import exam_evaluator
import handwriting
import pipeline
from splitting import pdf_to_images

app = FastAPI(title="IKU Exam Evaluator", version="0.5.0")


# ── Request models ────────────────────────────────────────────────

class GradeOpenEndedRequest(BaseModel):
    question: str
    answer_key: str
    student_answer: str
    max_points: float
    course_name: Optional[str] = None
    specific_rules: Optional[str] = None


class OcrOpenEndedRequest(BaseModel):
    image_base64: str


class HardestQuestion(BaseModel):
    qn: int
    type: str
    avgPercent: float
    studentCount: int


class InsightsRequest(BaseModel):
    """Aggregate analytics data → 2-3 actionable bullets via Ollama.

    The frontend's old InsightsCard rendered a deterministic
    conditional copy and labelled it "AI Insight" which was
    misleading. This endpoint replaces that with a real LLM call.
    """
    examId: Optional[str] = None
    classMean: float                    # 0-100
    classStdDev: float                  # percentage points
    totalStudents: int
    belowPassing: int                   # count of students under 50%
    hardestQuestions: list[HardestQuestion] = []


os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# Cheap in-memory cache for /insights — keyed by a hash of the
# request body. 5-minute TTL avoids re-running Ollama on every
# Analytics tab toggle, which would be expensive (each call adds
# ~3-5 s of latency).
import hashlib as _hashlib
import time as _time
_INSIGHTS_CACHE: dict[str, tuple[float, dict]] = {}
_INSIGHTS_TTL_S = 300


# ── Startup: warm heavy models so the first request isn't slow ────

@app.on_event("startup")
async def _preload():
    print(f"[Startup] torch.num_threads = {_NUM_THREADS}")
    try:
        from handwriting import (
            _load_digit_cnn,
            _ensure_tesseract,
            _load_letter_cnn,
            _load_trocr,
            _ensure_trocr_bad_words,
            _ensure_trocr_digit_bad_words,
        )
        # Cheap preloads — milliseconds.
        _load_digit_cnn()
        _ensure_tesseract()
        _load_letter_cnn()
        # Heavy preloads — TrOCR weights (~1.3 GB) + bad_words token tables.
        # Without this, the first /evaluate request pays ~5-10 s building the
        # 50k-token digit-only block list. With it, that cost is paid once at
        # startup and the first request is as fast as the second.
        _load_trocr()
        _ensure_trocr_bad_words()
        _ensure_trocr_digit_bad_words()
        print("[Startup] All readers warm")
    except Exception as e:
        print(f"[Startup] Preload warning: {e}")


# ── Routes ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Service + AI mode status."""
    ai_state = await ai_evaluation.ai_health()
    return {
        "status": "ok",
        "version": "0.5.0",
        **ai_state,
    }


@app.post("/evaluate")
async def evaluate_exam_endpoint(
    pdf_file: UploadFile = File(...),
    map_file: UploadFile = File(...),
):
    """
    Full evaluation pipeline. Handles 300+ students.
    Processes one student at a time to control memory.
    """
    try:
        map_content = await map_file.read()
        exam_map = json.loads(map_content)
        pdf_content = await pdf_file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read files: {e}")

    try:
        return await pipeline.evaluate_exam(pdf_content, exam_map)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)[:300]}")


@app.post("/align")
async def align_only(
    pdf_file: UploadFile = File(...),
    map_file: UploadFile = File(...),
):
    """Debug: align every page in the PDF using the map's anchors."""
    try:
        map_content = await map_file.read()
        exam_map = json.loads(map_content)
        pdf_content = await pdf_file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read files: {e}")

    images = pdf_to_images(pdf_content)
    pages = exam_map.get("pages", []) or []
    saved = []

    for i, (img, page_data) in enumerate(zip(images, pages)):
        aligned = alignment.align_page(
            img,
            page_data.get("anchors", {}),
            page_width=int(page_data.get("pageWidth", config.PAGE_WIDTH_PX)),
            page_height=int(page_data.get("pageHeight", config.PAGE_HEIGHT_PX)),
        )
        path = os.path.join(config.OUTPUT_DIR, f"aligned_page_{i + 1}.jpg")
        cv2.imwrite(path, aligned)
        saved.append(path)

    return {"alignedPages": saved}


@app.get("/results/{exam_id}/excel")
async def download_excel(exam_id: str):
    """Serve the generated Excel for a given exam id."""
    path = os.path.join(config.OUTPUT_DIR, f"{exam_id}_results.xlsx")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Excel not found. Run /evaluate first.")
    return FileResponse(path, filename=f"{exam_id}_results.xlsx")


# Allow only safe basenames: letters, digits, underscore, dot, hyphen + .jpg.
# Matches files like `123456_page1.jpg` or `unknown_2_page3.jpg`. Rejects
# any traversal attempt (slash, backslash, "..").
_ALIGNED_PAGE_RE = re.compile(r"^[A-Za-z0-9_.-]+\.jpg$")


@app.get("/aligned-page/{filename}")
async def aligned_page(filename: str):
    """
    Serve a per-student aligned page JPEG written by the pipeline.

    Files are produced by the evaluation loop as
    `{student_number}_page{N}.jpg` under `config.OUTPUT_DIR`. The frontend
    references them via `StudentResult.pages[i].imageFilename` and renders
    them as the PageMapCanvas background.

    Filename must be a simple basename matching `[A-Za-z0-9_.-]+\\.jpg` —
    anything containing `/`, `\\`, or `..` is rejected to prevent path
    traversal. Returns 404 if the file does not exist on disk.
    """
    if not _ALIGNED_PAGE_RE.match(filename) or "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = os.path.join(config.OUTPUT_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Aligned page not found")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/debug/annotate")
async def annotate_endpoint(
    pdf_file: UploadFile = File(...),
    map_file: UploadFile = File(...),
):
    """
    Generate annotated overlay images for visual alignment validation.

    For each page in the map JSON, the corresponding scanned page is annotated
    with color-coded boxes (anchors red, student region purple, questions
    green, options blue, solution areas orange, matching yellow, fill_blanks
    light green). The original scan is NOT warped — annotations are placed
    using the canonical->scanned transform context.

    Use this BEFORE running /evaluate to confirm the map JSON aligns properly
    with the actual scan. The footer banner shows transform mode + anchor
    count.

    Returns:
        - annotatedPages: list of file paths under OUTPUT_DIR/annotated/
        - perPageInfo: transform mode + anchor count per page
    """
    try:
        map_content = await map_file.read()
        exam_map = json.loads(map_content)
        pdf_content = await pdf_file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read files: {e}")

    images = pdf_to_images(pdf_content)
    pages = exam_map.get("pages", []) or []
    if not pages:
        raise HTTPException(status_code=400, detail="Map has no 'pages' array")

    debug_dir = os.path.join(config.OUTPUT_DIR, "annotated")
    os.makedirs(debug_dir, exist_ok=True)

    saved: list = []
    per_page_info: list = []

    # Annotate the first occurrence of each canonical page (page index in the map).
    # For multi-student PDFs we just sample the first instance.
    n_pages_to_render = min(len(images), len(pages))
    for page_index in range(n_pages_to_render):
        scanned = images[page_index]
        page_data = pages[page_index]

        try:
            annotated, ctx = alignment.annotate_page(scanned, page_data)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Annotation failed for page {page_index + 1}: {str(e)[:200]}",
            )

        page_id = page_data.get("pageId", f"page_{page_index + 1}")
        out_path = os.path.join(debug_dir, f"annotated_{page_id}.jpg")
        cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved.append(out_path)
        per_page_info.append({
            "pageId": page_id,
            "transformMode": ctx.mode,
            "anchorsDetected": ctx.n_anchors_detected,
            "scaleX": round(ctx.scale_x, 4),
            "scaleY": round(ctx.scale_y, 4),
            "offsetX": round(ctx.offset_x, 2),
            "offsetY": round(ctx.offset_y, 2),
        })

    return {
        "annotatedPages": saved,
        "perPageInfo": per_page_info,
        "count": len(saved),
    }


@app.post("/grade-open-ended")
async def grade_open_ended_endpoint(req: GradeOpenEndedRequest):
    """Grade a single open-ended answer via the configured LLM grader (Ollama)."""
    try:
        kwargs = {
            "question": req.question,
            "answer_key": req.answer_key,
            "student_answer": req.student_answer,
            "max_points": req.max_points,
        }
        if req.course_name is not None:
            kwargs["course_name"] = req.course_name
        if req.specific_rules is not None:
            kwargs["specific_rules"] = req.specific_rules
        result = exam_evaluator.grade_open_ended_answer(**kwargs)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Grading error: {str(e)[:300]}")

    # Safety guard — if the justification echoes any flagged phrase, force review.
    explanation = (result.get("justification") or "").lower()
    for flag in config.AI_SAFETY_FLAGS:
        if flag in explanation:
            result["requires_human_review"] = True
            result["status"] = "ai_flagged"
            break

    return result


@app.post("/insights")
async def insights_endpoint(req: InsightsRequest):
    """Generate 2-3 actionable bullets about class performance via
    the same Ollama setup that powers /grade-open-ended.

    Replaces the renderer's deterministic stub. The "AI Insight"
    label on the frontend card becomes honest after this lands.
    """
    # Cache key — hash of every meaningful input. 5-min TTL prevents
    # re-running Ollama on every Analytics tab toggle.
    body = req.model_dump_json()
    cache_key = _hashlib.md5(body.encode()).hexdigest()
    now = _time.time()
    hit = _INSIGHTS_CACHE.get(cache_key)
    if hit and (now - hit[0]) < _INSIGHTS_TTL_S:
        return hit[1]

    # Build a tight prompt for the LLM. The expected output shape is
    # JSON with a `bullets` array of strings; we let exam_evaluator's
    # urllib wrapper drive Ollama with format=json + temperature=0.
    hardest_lines = "\n".join(
        f"  - Q{q.qn} ({q.type}): avg {q.avgPercent:.0f}% across {q.studentCount} students"
        for q in req.hardestQuestions[:5]
    )
    prompt = (
        f"You are an academic advisor reviewing class results. Produce 2-3 short, "
        f"actionable bullet points (each < 25 words) about how the teacher should "
        f"interpret these numbers. Be specific. No fluff.\n\n"
        f"Class size: {req.totalStudents}\n"
        f"Class mean: {req.classMean:.1f}%\n"
        f"Std deviation: {req.classStdDev:.1f}%\n"
        f"Below 50% (failing): {req.belowPassing}\n"
        f"Hardest questions:\n{hardest_lines or '  (none)'}\n\n"
        f"Respond ONLY with a JSON object: "
        f"{{\"bullets\": [\"...\", \"...\"]}}. No preamble."
    )

    # Reuse the existing Ollama HTTP plumbing in exam_evaluator —
    # doesn't matter that the prompt is analytics not grading.
    try:
        result = exam_evaluator.grade_open_ended_answer(
            question="Class analytics summary",
            answer_key="Bullet-point insights",
            student_answer=prompt,
            max_points=10,
            course_name="Analytics",
            specific_rules="Output JSON with a 'bullets' string array only.",
        )
    except Exception as e:
        return {"bullets": [], "error": f"LLM call failed: {str(e)[:200]}"}

    # The grader returns a justification field with the LLM's reply.
    # Try to parse JSON out of it; fall back to splitting by newlines.
    raw = result.get("justification") or ""
    bullets: list[str] = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            cand = parsed.get("bullets")
            if isinstance(cand, list):
                bullets = [str(x) for x in cand if str(x).strip()]
    except Exception:
        pass
    if not bullets:
        # Fallback: split lines, strip bullets / leading dashes
        for line in raw.splitlines():
            ln = line.strip().lstrip("-•*0123456789. ").strip()
            if ln and len(ln) > 5:
                bullets.append(ln)
        bullets = bullets[:3]

    payload = {"bullets": bullets[:3], "model": config.GRADING_MODEL}
    _INSIGHTS_CACHE[cache_key] = (now, payload)
    return payload


@app.post("/ocr-open-ended")
async def ocr_open_ended_endpoint(req: OcrOpenEndedRequest):
    """OCR a handwritten open-ended answer image (base64-encoded PNG/JPEG)."""
    try:
        try:
            raw = base64.b64decode(req.image_base64, validate=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {e}")

        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image bytes")

        result = handwriting.read_handwriting_image(img)
        return {"text": result.text, "confidence": float(result.confidence)}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OCR error: {str(e)[:300]}")


# ── Local run ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
