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
    try:
        from handwriting import _load_digit_cnn, _ensure_tesseract
        _load_digit_cnn()
        _ensure_tesseract()
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


# ── Local run ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
