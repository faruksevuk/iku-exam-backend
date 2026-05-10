# IKU Exam Backend

FastAPI service that takes a scanned multi-student exam PDF and the
exam's coordinate map, and returns per-student, per-question grades.
Designed to run on-premise on a single CPU machine — no GPU, no cloud
calls, no internet at runtime beyond a local Ollama daemon.

Paired with the [iku-exam-generator](https://github.com/faruksevuk/iku-exam-generator)
Electron desktop app, which spawns this backend as a managed child
process. The backend can also be hit directly via HTTP for testing.

---

## Quick start

```bash
git clone https://github.com/faruksevuk/iku-exam-backend.git exam-backend
cd exam-backend
python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Visit `http://127.0.0.1:8000/health` to confirm it's up.

The Electron app picks a random free port at spawn time and passes
`PORT` via env, so when running through the app you don't fix the port
here yourself.

---

## External dependencies

### Tesseract OCR

Tesseract is used as the consensus reader for matching cells and
fill-in-blank answers. The path is in `config.TESSERACT_CMD`.

```bash
winget install UB-Mannheim.TesseractOCR     # Windows
brew install tesseract                       # macOS
sudo apt install tesseract-ocr               # Debian / Ubuntu
```

### Ollama (for open-ended grading)

The open-ended grader (`POST /grade-open-ended`) talks to a local
[Ollama](https://ollama.com) daemon at `http://127.0.0.1:11434/api/chat`.
No `llama-cpp-python` required — we call Ollama from stdlib
`urllib.request`.

```bash
winget install Ollama.Ollama                 # Windows (runs as background service)
brew install ollama && ollama serve &        # macOS

ollama pull qwen3:1.7b
```

The grading model and URL live in `config.py` as `GRADING_MODEL` and
`OLLAMA_URL`. To disable AI grading entirely (placeholder mode — every
open-ended answer is flagged for manual review), set `AI_ENABLED = False`.

### Hardware

| Course size | RAM | Disk | Notes |
|---|---|---|---|
| ≤ 50 students per exam | 8 GB | 5 GB | Single-batch runs comfortably |
| 50 – 200 students | 16 GB | 10 GB | TrOCR + Qwen3 1.7B both resident |
| 200+ students | 16 GB+ | 20 GB | Run overnight |

AVX2-capable CPU recommended (PyTorch CPU kernels). NVIDIA GPU is
optional — Ollama autodetects it for offload. No CUDA dependency in
this repo.

---

## Architecture

```
PDF + map.json
     │
     ▼
┌────────────────┐
│ pipeline.py    │  orchestrator: per-student loop, per-question dispatch
└────────────────┘
     │
     ├─► splitting.py     PDF → images, QR-based student grouping
     ├─► alignment.py     bullseye anchor detection + perspective warp
     ├─► omr.py           MC / multi-select bubble OMR
     ├─► handwriting.py   DigitCNN / LetterCNN / Tesseract / TrOCR cascade
     ├─► ai_evaluation.py open-ended: TrOCR transcription + Qwen3 verdict
     ├─► grading.py       per-type scoring rules
     └─► export.py        per-exam xlsx writer
```

The pipeline emits structured `[STAGE] key=value …` lines to stdout so
a host process (Electron's main) can drive a live progress UI:

```
[STAGE] stage=batch state=start eid=<examId> sTotal=<n>
[STAGE] sIdx=0 sTotal=<n> sNum=<num> stage=read
[STAGE] stage=q qNum=1 qType=multiple_choice elapsedMs=12
[STAGE] sIdx=0 stage=done sNum=<num> hadAi=1 ocrMs=16720 aiMs=91688
[STAGE] stage=batch state=end eid=<examId> sTotal=<n>
```

Partial results are durable. Each per-student iteration is wrapped in
its own `try/except` so a single bad scan doesn't abort the batch, and
the pipeline writes a sibling `<output>/<examId>_results.json` on every
run — recoverable even if the calling renderer is killed mid-evaluation.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | Service status, AI mode |
| POST | `/evaluate` | Full pipeline. Multipart `pdf_file` + `map_file`. |
| POST | `/align` | Debug: write aligned per-page JPEGs to disk |
| POST | `/debug/annotate` | Annotated overlay images for map validation |
| POST | `/grade-open-ended` | Grade a single open-ended answer (JSON body) |
| POST | `/ocr-open-ended` | Transcribe one handwritten answer (JSON, base64 image) |
| POST | `/insights` | Class-level LLM-generated insights for the analytics dashboard |
| GET  | `/results/{exam_id}/excel` | Download generated xlsx (legacy) |
| POST | `/export-excel` | Generate xlsx on demand from supplied JSON. Kept for back-compat — the desktop app now does xlsx generation locally with `exceljs` to remove the backend hop. |
| GET  | `/aligned-page/{filename}` | Per-student aligned-page JPEG (used by the review workspace). Path-traversal validated. |

See `app.py` docstrings for request / response shapes.

---

## Config knobs (`config.py`)

| Variable | Purpose |
|---|---|
| `OUTPUT_DIR` | Where per-exam xlsx + JSON sibling land |
| `SCAN_DPI` | Rasterisation DPI for input PDFs (default 200) |
| `HIGH_CONF_THRESHOLD` | Confidence floor for "needs review" flag |
| `TESSERACT_CMD` | Absolute path to the Tesseract binary |
| `AI_ENABLED` | False → open-ended answers return placeholder verdicts |
| `OLLAMA_URL` | LLM endpoint (default `http://127.0.0.1:11434`) |
| `GRADING_MODEL` | Ollama model tag for open-ended grading |
| `LLM_TIMEOUT_S` | Per-call timeout (90 s default) |

---

## Dev scripts (`scripts/`)

| Script | Use |
|---|---|
| `make_real_mock_students.py` | Overlay synthetic answers onto a blank PDF to produce test student PDFs |
| `make_exam_batches.py` | Bulk-generate exam configurations for regression testing |
| `_analyze_test_runs.py` | Compare actual evaluation results against expected outputs (per-exam, per-question-type accuracy) |
| `_extract_timings.py` | Parse backend.log for per-question and per-student timings |
| `_build_test_report.py` | Generate the comprehensive Word test report |
| `_recover_results_from_xlsx.py` | One-shot recovery if results.json is lost but xlsx survives |
| `smoke_evaluate_openended.py` | Quick functional test of the open-ended path |
| `_build_edge_case_exam.py` | Build a small edge-case test PDF (rotation, scribbled QR, etc.) |
| `_overlay_edge_students_v5.py` | Latest overlay generator with homography-corrected coordinates |
| `_compare_edge_results.py` | Compare edge-case evaluation against expected outputs and update the report |

---

## Licensing

| Component | License | Notes |
|---|---|---|
| Project source | MIT | This repo |
| FastAPI, Uvicorn, openpyxl, rapidfuzz | MIT / Apache 2.0 | Permissive |
| PyTorch, Transformers, sentence-transformers | BSD / Apache 2.0 | Permissive |
| TrOCR weights (Hugging Face) | MIT | `microsoft/trocr-base-handwritten` and `-large-handwritten` |
| Qwen3 1.7B weights | Apache 2.0 | Pulled at install via Ollama |
| Tesseract | Apache 2.0 | External binary |
| **PyMuPDF (fitz)** | **AGPL-3** | OK for on-premise use; SaaS redistribution requires a commercial PyMuPDF licence or a switch to a permissive alternative |

---

## Troubleshooting

**Backend won't start, "Could not import module 'app'"** — the working
directory passed to `uvicorn` doesn't contain `app.py`. From the
Electron app, open Backend Settings → Choose folder… and point at the
folder containing `app.py`. The choice is persisted.

**TrOCR cold-start takes 60+ s on first launch** — expected. The model
weights are downloaded from Hugging Face the first time you run the
pipeline. Subsequent launches load from the local cache (`~/.cache/
huggingface`) in ~5 s.

**"No xlsx file at output/…_results.xlsx"** — the desktop app now
generates the workbook locally on the renderer side, so this disk file
is no longer required. If you're hitting the legacy `GET /results/
{exam_id}/excel` endpoint directly, run a `/evaluate` once first or
use the newer `POST /export-excel`.

**Open-ended grading returns `status="pending_review"`** — the AI
grader is in placeholder mode. Either Ollama isn't running, the model
isn't pulled, or `AI_ENABLED=False` in `config.py`.
