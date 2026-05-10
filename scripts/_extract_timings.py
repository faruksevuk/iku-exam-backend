"""Parse backend.log STAGE events for the 3 test-batch evaluations and
emit per-student / per-question-type timing breakdowns + a chart.

Each batch is bracketed by:
  [STAGE] stage=batch state=start sTotal=10
  [STAGE] stage=batch state=end eid=<slug> sTotal=10

Inside each batch:
  [STAGE] sIdx=0 sTotal=10 sNum=2420000001 stage=read
  [STAGE] stage=q qNum=1 qType=multiple_choice elapsedMs=0
  [STAGE] sIdx=0 sTotal=10 sNum=2420000001 stage=done hadAi=1 ocrMs=12340 aiMs=87654
"""
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOG = Path(os.environ["APPDATA"]) / "iku-exam-generator" / "backend.log"
ASSETS_DIR = Path(r"C:/Users/faruk/Downloads/_test_report_assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

EXAM_LABELS = {
    "4012-se-midterm-2026": "Software Engineering",
    "mat102-calc2-midterm-2026": "Calculus II",
    "phy201-physics-quiz-2026": "Physics",
}

GREEN = "#10B981"
AMBER = "#F59E0B"
IKU_RED = "#ED1B24"
BLUE = "#3B82F6"

text = LOG.read_text(encoding="utf-8", errors="replace")

# Find all batch start/end blocks where eid matches one of our test exams
# We walk the log linearly tracking the current batch and grabbing per-q + per-student events
events: list[tuple[int, str]] = []
for i, line in enumerate(text.splitlines()):
    if "[STAGE]" in line:
        events.append((i, line))

# Build batches: list of (eid, [event lines])
batches: list[dict] = []
current: dict | None = None

for idx, line in events:
    if "stage=batch state=start" in line:
        current = {"eid": None, "lines": [line], "start_idx": idx}
    elif "stage=batch state=end" in line:
        if current is not None:
            current["lines"].append(line)
            m = re.search(r"eid=(\S+)", line)
            if m:
                current["eid"] = m.group(1)
            batches.append(current)
            current = None
    elif current is not None:
        current["lines"].append(line)

# Filter to 10-student batches that look like test runs.
# Earlier batches may not have eid= tagged (logging was added partway
# through the session), so we fall back to matching by sTotal=10 and
# pick the last 3 such batches in chronological order.
ten_student_batches = [
    b for b in batches
    if any("sTotal=10" in line and "stage=batch state=start" in line for line in b["lines"])
]
# Take the last 3 (chronological). Map them onto the 3 EXAM_LABELS
# in the order the user evaluated them: SE → Calculus → Physics.
last_three = ten_student_batches[-3:]
slugs_in_order = list(EXAM_LABELS.keys())
last_per_eid: dict[str, dict] = {}
for slug, b in zip(slugs_in_order, last_three):
    if not b.get("eid"):
        b["eid"] = slug
    last_per_eid[b["eid"]] = b

print(f"Total 10-student batches in log: {len(ten_student_batches)}")
print(f"Using last 3 batches mapped to: {list(last_per_eid.keys())}")

# Per-question-type elapsedMs collected across all 3 batches
qtype_times: dict[str, list[int]] = defaultdict(list)
# Per-student total ocrMs / aiMs
student_ocr_ms: list[int] = []
student_ai_ms: list[int] = []
# Per-batch totals
batch_totals: dict[str, dict] = {}

for slug, b in last_per_eid.items():
    s_count = 0
    s_ocr_total = 0
    s_ai_total = 0
    for line in b["lines"]:
        # Per-question
        if "stage=q " in line:
            qm = re.search(r"qType=(\w+).*elapsedMs=(\d+)", line)
            if qm:
                qtype = qm.group(1)
                ms = int(qm.group(2))
                # Skip multi_select and any oddballs we don't track.
                qtype_times[qtype].append(ms)
        # Per-student done
        if "stage=done" in line:
            dm = re.search(r"ocrMs=(\d+).*aiMs=(\d+)", line)
            if dm:
                ocr = int(dm.group(1))
                ai = int(dm.group(2))
                student_ocr_ms.append(ocr)
                student_ai_ms.append(ai)
                s_count += 1
                s_ocr_total += ocr
                s_ai_total += ai
    batch_totals[slug] = {
        "label": EXAM_LABELS[slug],
        "students": s_count,
        "total_ocr_ms": s_ocr_total,
        "total_ai_ms": s_ai_total,
        "total_ms": s_ocr_total + s_ai_total,
        "avg_per_student_s": (s_ocr_total + s_ai_total) / max(s_count, 1) / 1000,
    }

# ── Print summary ─────────────────────────────────────────────────
print("\n" + "═" * 60)
print("TIMING ANALYSIS — extracted from backend.log STAGE events")
print("═" * 60)

print("\nPer-question-type avg time (ms):")
for qt in ["multiple_choice", "matching", "fill_blanks", "open_ended"]:
    times = qtype_times.get(qt, [])
    if times:
        avg = sum(times) / len(times)
        med = sorted(times)[len(times) // 2]
        mx = max(times)
        print(f"  {qt:<18} n={len(times):>3}  avg={avg/1000:>6.2f}s  median={med/1000:>6.2f}s  max={mx/1000:>6.2f}s")

print("\nPer-batch totals:")
for slug, b in batch_totals.items():
    print(f"  {b['label']:<25}  students={b['students']:>2}  "
          f"OCR={b['total_ocr_ms']/1000:>6.1f}s  AI={b['total_ai_ms']/1000:>6.1f}s  "
          f"avg/student={b['avg_per_student_s']:>5.1f}s")

# ── Chart: timing breakdown ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))

qtypes_for_chart = ["multiple_choice", "matching", "fill_blanks", "open_ended"]
qtype_labels = ["Multiple Choice", "Matching", "Fill-in-Blank", "Open-Ended"]
avgs = []
medians = []
for qt in qtypes_for_chart:
    times = qtype_times.get(qt, [])
    if times:
        avgs.append(sum(times) / len(times) / 1000)
        medians.append(sorted(times)[len(times) // 2] / 1000)
    else:
        avgs.append(0)
        medians.append(0)

x = np.arange(len(qtype_labels))
w = 0.35
ax.bar(x - w/2, avgs, w, color=BLUE, label="Mean", edgecolor="white", linewidth=1)
ax.bar(x + w/2, medians, w, color=IKU_RED, label="Median", edgecolor="white", linewidth=1)
for i, (a, m) in enumerate(zip(avgs, medians)):
    ax.text(i - w/2, a + 0.5, f"{a:.1f}s", ha="center", fontsize=10, fontweight="bold")
    ax.text(i + w/2, m + 0.5, f"{m:.1f}s", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(qtype_labels, fontsize=11)
ax.set_ylabel("Time per question (seconds)", fontsize=11)
ax.set_title("Per-question processing time by type",
             fontsize=13, fontweight="bold", pad=14)
ax.legend(loc="upper left", framealpha=0.95)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "timing_per_question.png", dpi=150, facecolor="white")
plt.close()

# ── Chart: per-student total time stacked OCR + AI ───────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
labels = [batch_totals[s]["label"] for s in EXAM_LABELS if s in batch_totals]
ocrs = [batch_totals[s]["total_ocr_ms"] / 1000 / batch_totals[s]["students"]
        for s in EXAM_LABELS if s in batch_totals]
ais = [batch_totals[s]["total_ai_ms"] / 1000 / batch_totals[s]["students"]
       for s in EXAM_LABELS if s in batch_totals]
x = np.arange(len(labels))
ax.bar(x, ocrs, color=BLUE, label="OCR + reading", edgecolor="white", linewidth=1.5)
ax.bar(x, ais, bottom=ocrs, color=AMBER, label="AI grading (open-ended)", edgecolor="white", linewidth=1.5)
for i, (o, a) in enumerate(zip(ocrs, ais)):
    total = o + a
    ax.text(i, total + 2, f"{total:.0f}s/student", ha="center", fontsize=11, fontweight="bold")
    if o > 5:
        ax.text(i, o / 2, f"OCR\n{o:.0f}s", ha="center", color="white", fontsize=9, fontweight="bold")
    if a > 5:
        ax.text(i, o + a / 2, f"AI\n{a:.0f}s", ha="center", color="white", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Average seconds per student", fontsize=11)
ax.set_title("End-to-end processing time per student (OCR + AI grading)",
             fontsize=13, fontweight="bold", pad=14)
ax.legend(loc="upper right", framealpha=0.95)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "timing_per_student.png", dpi=150, facecolor="white")
plt.close()

# ── Save timings JSON ─────────────────────────────────────────────
timing_summary = {
    "per_question_type": {
        qt: {
            "n": len(times),
            "avg_ms": round(sum(times) / len(times), 1) if times else 0,
            "median_ms": sorted(times)[len(times) // 2] if times else 0,
            "max_ms": max(times) if times else 0,
        }
        for qt, times in qtype_times.items()
    },
    "per_batch": batch_totals,
    "global": {
        "total_students_timed": len(student_ocr_ms),
        "avg_ocr_ms_per_student": round(sum(student_ocr_ms) / len(student_ocr_ms), 1) if student_ocr_ms else 0,
        "avg_ai_ms_per_student": round(sum(student_ai_ms) / len(student_ai_ms), 1) if student_ai_ms else 0,
    },
}
with open(ASSETS_DIR / "_timings.json", "w", encoding="utf-8") as f:
    json.dump(timing_summary, f, ensure_ascii=False, indent=2)

print(f"\nWrote timing charts + JSON to: {ASSETS_DIR}")
