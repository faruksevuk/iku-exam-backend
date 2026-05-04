# IKU Exam Backend

FastAPI service for OMR + handwriting + AI grading of scanned exam PDFs.

## Setup

### Python deps

```
pip install -r requirements.txt
```

### Tesseract (handwriting cascade)

Install the Tesseract binary (Windows default path is hard-coded in
`config.TESSERACT_CMD`):

```
winget install UB-Mannheim.TesseractOCR
```

### AI grading setup (Ollama)

The open-ended grader (`/grade-open-ended`) talks to a local
[Ollama](https://ollama.com) daemon over HTTP. No `llama-cpp-python`
needed — we call `${OLLAMA_URL}/api/chat` from stdlib `urllib.request`.

1. Install Ollama:
   ```
   winget install Ollama.Ollama
   ```
   On Windows, Ollama runs as a background service automatically — no
   need to run `ollama serve` manually. (On macOS/Linux, run `ollama
   serve` once per session.)

2. Pull the grading model:
   ```
   ollama pull qwen3:1.7b
   ```
   ~1.1 GB on disk. Fits in 4 GB VRAM; CPU fallback uses ~1.5 GB RAM.

The model name + URL are configurable in `config.py` (`GRADING_MODEL`,
`OLLAMA_URL`). To disable AI grading entirely (placeholder mode for
manual review), set `AI_ENABLED = False`.

### Hardware notes

- **8 GB RAM minimum** (TrOCR-base + Qwen3-1.7B together).
- **NVIDIA GPU with 4+ GB VRAM** is auto-detected by Ollama for offload —
  no config required.
- AI grading falls back to placeholder/manual review automatically if
  the Ollama daemon is unreachable.

## Run

```
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

## Endpoints

- `GET  /health` — service status + AI mode
- `POST /evaluate` — full pipeline (multipart `pdf_file` + `map_file`)
- `POST /align` — debug: write aligned pages to disk
- `POST /debug/annotate` — annotated overlay images for map validation
- `POST /grade-open-ended` — grade a single open-ended answer (JSON body)
- `POST /ocr-open-ended` — OCR a handwritten answer image (JSON body, base64)
- `GET  /results/{exam_id}/excel` — download generated xlsx

See `app.py` docstrings for request/response shapes.
