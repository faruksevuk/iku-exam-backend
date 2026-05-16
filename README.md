# IKU Exam Backend

FastAPI service that takes a scanned multi-student exam PDF and the
exam's coordinate map, and returns per-student, per-question grades.
Designed to run on-premise on a single CPU machine — no GPU, no cloud
calls, no internet at runtime beyond a local Ollama daemon.

---

## ⚠ This project lives in two repositories

You almost certainly want both. This repo is just the engine.

| Repo | What it is | What you do with it |
|---|---|---|
| **iku-exam-backend** (this repo) | The Python FastAPI service that does OCR + AI grading | Install once with `pip install -r requirements.txt`. **The desktop app starts this for you** — you typically do not run it by hand. |
| **[iku-exam-generator](https://github.com/faruksevuk/iku-exam-generator)** | The Electron desktop app — the GUI teachers actually use | Authors exams, ingests scans, shows the review workspace. Start there if you are new. |

If you are just trying to get the application running for the first
time, **go to the [generator repo's README](https://github.com/faruksevuk/iku-exam-generator#readme)
first** — its step-by-step setup walks through both repositories in
the right order.

This README documents the backend in detail for developers who want
to test it standalone, debug it, or extend it.

---

## What this service does (in one paragraph)

Receives a scanned multi-student exam PDF + a `map.json` describing
where every question lives on the page. Splits per student via QR or
sequential fallback, aligns each page to the canonical layout using
corner anchors, then dispatches each question to the right reader:
classical bubble OMR for multiple-choice and multi-select; a
LetterCNN + Tesseract cascade for matching cells; a TrOCR + Tesseract
cascade for fill-in-blanks; a Qwen2.5-VL 3B vision model for
open-ended transcription that feeds into a Qwen3 1.7B rubric grader.
Returns a per-student, per-question result tree as JSON, also writes
it to disk as a recovery file, and mirrors it into a local SQLite
shadow store for queryable history.

---

## Standalone quick start (developers only)

If you have a venv set up and just want the backend running:

```bash
cd iku-exam-backend
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Visit `http://127.0.0.1:8000/health` to confirm it's up.

The Electron app picks a random free port at spawn time and passes
`PORT` via env, so when running through the app you don't fix the port
here yourself.

---

## First-time setup (if you are on a fresh machine)

These steps assume nothing is installed. If you just want the desktop
app working, the **[generator repo's README](https://github.com/faruksevuk/iku-exam-generator#readme)
is the better starting point** — it covers both repos in one
walk-through. The steps below cover the backend half only.

### Prerequisites

| Tool | Purpose | Where to get it |
|---|---|---|
| **Python 3.11+** | runs this backend | <https://www.python.org/downloads/> — tick "Add Python to PATH" |
| **Git** | clones this repo | <https://git-scm.com/downloads> |
| **Tesseract OCR** | one of the OCR readers | Windows: `winget install UB-Mannheim.TesseractOCR` / macOS: `brew install tesseract` / Debian: `sudo apt install tesseract-ocr` |
| **Ollama** | serves the two local language models | <https://ollama.com/download> — installs as a background service |

Hardware: AVX2-capable CPU, **8 GB RAM minimum** (16 GB recommended),
**~6 GB free disk** for model weights. NVIDIA GPU is optional; Ollama
will use it for offload if present.

### Step 1 — Clone and install Python deps

```bash
git clone https://github.com/faruksevuk/iku-exam-backend.git
cd iku-exam-backend
python -m venv .venv

# Activate the venv:
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux

pip install -r requirements.txt
```

This pulls FastAPI, PyTorch (CPU), TrOCR / Transformers, SQLAlchemy,
PyMuPDF and friends — ~2 GB on disk, 5–10 minutes on a fresh install.

### Step 2 — Pull the two AI models via Ollama

```bash
ollama pull qwen3:1.7b        # ~1.1 GB — the rubric grader
ollama pull qwen2.5vl:3b      # ~3 GB — the vision model for handwriting
```

If you skip the vision model, open-ended grading will still work but
falls back to TrOCR-only transcription and is noticeably less
accurate.

### Step 3 — First launch

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Expect 50–60 seconds of model warm-up the first time, then:

```
[DB] SQLite ready: …/exam_demo.db
[Startup] All readers warm
INFO: Uvicorn running on http://127.0.0.1:8000
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health
```

Should return:

```json
{"status":"ok","version":"0.5.0","ai_enabled":true,
 "provider":"ollama","vision_model":"qwen2.5vl:3b",
 "grading_model":"qwen3:1.7b"}
```

If `ai_enabled` is `false` or either model field is missing, Ollama
isn't reachable or the models aren't pulled.

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

### Ollama (for open-ended grading + handwriting transcription)

Open-ended questions go through two local models served by
[Ollama](https://ollama.com) at `http://127.0.0.1:11434`:

1. **Qwen2.5-VL 3B** — vision-language model that transcribes the
   handwritten student answer directly from the page crop.
2. **Qwen3 1.7B** — rubric grader that scores the transcription
   against the answer key.

No `llama-cpp-python` required — we call Ollama from stdlib
`urllib.request`.

```bash
winget install Ollama.Ollama                 # Windows (runs as background service)
brew install ollama && ollama serve &        # macOS

ollama pull qwen3:1.7b       # ~1.1 GB
ollama pull qwen2.5vl:3b     # ~3 GB
```

The grading model, vision model, and Ollama URL live in `config.py`
as `GRADING_MODEL`, `VISION_MODEL`, and `OLLAMA_URL`. To disable AI
grading entirely (placeholder mode — every open-ended answer is
flagged for manual review), set `AI_ENABLED = False`.

### SQLite database (zero setup)

Every saved run is mirrored into a local SQLite file
(`exam_demo.db`, next to `app.py`) via SQLAlchemy. The schema is
created automatically on first run by `database.init_db()` — you do
not need to install a database server. The JSON results file on disk
remains the source of truth for the desktop app; the database is
queryable shadow storage for the audit trail.

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
| GET  | `/db/health` | SQLite row counts per table. Use it to confirm the DB layer is alive. |
| GET  | `/db/exams` | List every saved exam run, newest first. |
| GET  | `/db/exams/{exam_id}/results` | Full result tree (students × questions × OCR/LLM rows) from the DB. |
| POST | `/db/question-results/{id}/override` | Apply a teacher override; auto-recomputes exam total and writes an audit log entry. |
| POST | `/db/exam-results/{id}/approve` | Final-approve a student's exam result; writes an audit log entry. |
| GET / POST | `/db/demo/seed` | Insert a small `DEMO_EXAM_001` dataset for tests without running the PDF pipeline. |

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
| `GRADING_MODEL` | Ollama model tag for open-ended grading (`qwen3:1.7b`) |
| `VISION_MODEL` | Ollama model tag for handwriting transcription (`qwen2.5vl:3b`) |
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
