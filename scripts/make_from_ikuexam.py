"""
make_from_ikuexam.py — drive the existing make_mock_student.py renderer
from a `.ikuexam` file instead of the hardcoded MOCK-ALL POOL.

Use case: the user's CSE301 + MAT102 exams exist in userData but have
no `.map.json` (Phase A wasn't run in the app). This script reads the
.ikuexam, builds the same POOL shape make_mock_student.py uses
internally, and produces a synthesized template + map + per-archetype
filled student PDFs.

It's a fallback for when Phase A isn't realistic — same output shape
as the existing MOCK-ALL flow, so the pipeline grades the result
identically.

Usage:
    .venv/Scripts/python.exe scripts/make_from_ikuexam.py <slug>
        [--archetype A|B] [--students 2]

Outputs (under `samples/mock/<slug>/`):
    <slug>.map.json         — synthesized map (also installed to userData)
    <slug>-student-A.pdf
    <slug>-student-B.pdf
    ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Re-use the existing make_mock_student internals. We replace its
# top-level POOL + EXAM_ID + output paths, then call build_pages_and_map
# directly. The file-level globals are mutable so we monkey-patch.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_mock_student as mms  # type: ignore  # noqa: E402

def _resolve_user_data() -> Path:
    """Locate the Electron app's userData/exams directory.

    Windows quirk: when this script runs under the Microsoft Store
    Python (WindowsApps sandbox), `os.environ['APPDATA']` resolves to
    a per-package virtualized path under
    `AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.…\\LocalCache\\Roaming\\…`,
    which is NOT where the Electron app reads from. Files written
    there are silently invisible to the running app.

    We resolve the *real* roaming AppData by reading the user's
    profile directly, falling back to the env var only when that
    fails (e.g. on non-Windows platforms).
    """
    home = Path(os.path.expanduser("~"))
    candidate = home / "AppData" / "Roaming" / "iku-exam-generator" / "exams"
    if candidate.parent.exists():
        return candidate
    return Path(os.environ.get("APPDATA", "")) / "iku-exam-generator" / "exams"


USER_DATA = _resolve_user_data()


def ikuexam_to_pool(iku: dict) -> dict:
    """Translate the saved .ikuexam questions[] array into the
    make_mock_student POOL shape (one list per long-form type)."""
    pool: dict[str, list[dict]] = {
        "multiple_choice": [],
        "multi_select": [],
        "fill_blanks": [],
        "matching": [],
        "open_ended": [],
    }
    for q in iku.get("questions", []):
        t = q.get("type", "")
        if t == "mc":
            opts = list(q.get("options") or [])
            # Pad to 4 options with placeholders if the .ikuexam has fewer
            while len(opts) < 4:
                opts.append(f"Option {len(opts) + 1}")
            ci = int(q.get("correctAnswer", 0))
            ci = max(0, min(3, ci))
            pool["multiple_choice"].append({
                "prompt": q.get("text", ""),
                "options": opts[:4],
                "correct": "ABCD"[ci],
                "studentPick": "ABCD"[ci],  # archetype overrides
            })
        elif t == "ms":
            opts = list(q.get("options") or [])
            while len(opts) < 4:
                opts.append(f"Option {len(opts) + 1}")
            indices = [int(i) for i in q.get("correctAnswers", []) if 0 <= int(i) < 4]
            correct_letters = sorted({"ABCD"[i] for i in indices})
            pool["multi_select"].append({
                "prompt": q.get("text", ""),
                "options": opts[:4],
                "correct": correct_letters or ["A"],
                "studentPicks": correct_letters or ["A"],
            })
        elif t == "fill":
            answers = q.get("fillAnswers") or []
            blanks = {str(i + 1): a for i, a in enumerate(answers) if a}
            # Use fillText if present (has __ markers), else q.text
            prompt = q.get("fillText") or q.get("text", "")
            pool["fill_blanks"].append({
                "prompt": prompt,
                "blanks": blanks or {"1": "answer"},
                "studentBlanks": dict(blanks),
            })
        elif t == "match":
            left_arr = q.get("matchLeft") or []
            right_arr = q.get("matchRight") or []
            correct_arr = q.get("matchCorrect") or []
            left_items = {str(i + 1): v for i, v in enumerate(left_arr)}
            right_options = {chr(65 + i): v for i, v in enumerate(right_arr)}
            # matchCorrect can be a list of letters ("C", "A", "B") or
            # a list of indices. Coerce to letters.
            correct: dict[str, str] = {}
            for i, v in enumerate(correct_arr):
                key = str(i + 1)
                if isinstance(v, str) and len(v) == 1 and v.isalpha():
                    correct[key] = v.upper()
                elif isinstance(v, int) and 0 <= v < 26:
                    correct[key] = chr(65 + v)
                elif isinstance(v, str) and v.isdigit():
                    iv = int(v)
                    if 0 <= iv < 26:
                        correct[key] = chr(65 + iv)
            pool["matching"].append({
                "prompt": q.get("text", ""),
                "leftItems": left_items,
                "rightOptions": right_options,
                "correct": correct,
                "studentPicks": dict(correct),
            })
        elif t == "open":
            pool["open_ended"].append({
                "prompt": q.get("text", ""),
                "expected": q.get("correctText", ""),
                "studentAnswer": q.get("correctText", ""),
            })
    return pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("slug", help="Exam id (e.g. cse301-ds-quiz-2026)")
    parser.add_argument("--archetype", default=None,
                        help="A|B|C — render a single archetype. Omit for all.")
    parser.add_argument("--students", type=int, default=2,
                        help="How many archetype variants (1-3). Default 2 (A, B).")
    args = parser.parse_args()

    iku_path = USER_DATA / f"{args.slug}.ikuexam"
    if not iku_path.exists():
        print(f"[err] No .ikuexam at {iku_path}", file=sys.stderr)
        sys.exit(1)
    iku = json.loads(iku_path.read_text(encoding="utf-8"))
    pool = ikuexam_to_pool(iku)

    # Tally — let user know the layout we're about to render.
    counts = {k: len(v) for k, v in pool.items() if v}
    print(f"[info] {args.slug}: {counts}")

    # Output directory per slug — keeps generated artefacts organized.
    slug_out_dir = mms.REPO / "samples" / "mock" / args.slug
    slug_out_dir.mkdir(parents=True, exist_ok=True)

    # Pick which archetypes to render.
    if args.archetype:
        archetypes = [args.archetype.upper()]
    else:
        archetypes = ["A", "B", "C"][: args.students]

    for arche in archetypes:
        # Monkey-patch make_mock_student's module-level state.
        mms.POOL = pool
        mms.EXAM_ID = args.slug
        mms.ARCHETYPE = arche
        sn = mms.ARCHETYPE_SN.get(arche, mms.MOCK_STUDENT_NUMBER)
        mms.MOCK_STUDENT_NUMBER = sn

        # Render the page images + collect the map.
        try:
            images, map_pages = mms.build_pages_and_map()
        except Exception as e:
            print(f"[err] render failed for {arche}: {e}", file=sys.stderr)
            continue

        # Save PDF.
        out_pdf = slug_out_dir / f"{args.slug}-student-{arche}.pdf"
        images[0].save(
            out_pdf, "PDF", resolution=150.0,
            save_all=True, append_images=images[1:],
        )
        print(f"[ok] wrote {out_pdf.relative_to(mms.REPO)}  "
              f"({out_pdf.stat().st_size // 1024} KB, {len(images)} pages)  "
              f"archetype={arche}  SN={sn}")

    # Map + ikuexam are written ONCE per slug (they don't depend on
    # archetype). The map's coordinates are identical across runs;
    # we save the last archetype's map as the canonical one.
    map_data = {
        "examId": args.slug,
        "totalPages": len(map_pages),
        "pages": map_pages,
    }
    map_out = slug_out_dir / f"{args.slug}.map.json"
    map_out.write_text(json.dumps(map_data, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    print(f"[ok] wrote {map_out.relative_to(mms.REPO)}")

    # Install the map into the Electron userData so the app uses it.
    target_map = USER_DATA / f"{args.slug}.map.json"
    target_map.write_text(json.dumps(map_data, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    print(f"[ok] installed map to {target_map}")

    # Make sure no stale results.json exists for this slug — fresh exam.
    target_results = USER_DATA / f"{args.slug}.results.json"
    if target_results.exists():
        try:
            target_results.unlink()
            print(f"[ok] cleared stale {target_results.name}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
