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

import json
import os

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
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

import config
import ai_evaluation
import alignment
import pipeline
from splitting import pdf_to_images

app = FastAPI(title="IKU Exam Evaluator", version="0.5.0")

os.makedirs(config.OUTPUT_DIR, exist_ok=True)


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


# ── Local run ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
