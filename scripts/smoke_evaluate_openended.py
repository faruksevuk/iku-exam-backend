"""POST /evaluate using samples/openended PDF+map (smoke test)."""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = os.environ.get("BACKEND_URL", "http://127.0.0.1:57266").rstrip("/")

PDF = REPO / "samples/4064-midtermopenended-2026_260419_170254(5).pdf"
MAP = REPO / "samples/OPENENDED4064-midterm-2026_map(3).json"


def main() -> None:
    if not PDF.exists() or not MAP.exists():
        print("missing samples:", PDF, MAP)
        sys.exit(2)

    pdf = PDF.read_bytes()
    mp = MAP.read_text(encoding="utf-8")
    boundary = "----smoke"
    crlf = b"\r\n"
    parts: list[bytes] = []

    def add(name: str, filename: str, data: bytes, ct: str) -> None:
        head = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
            f'filename="{filename}"\r\nContent-Type: {ct}\r\n\r\n'
        )
        parts.append(head.encode() + data + crlf)

    add("map_file", "map.json", mp.encode("utf-8"), "application/json")
    add("pdf_file", "exam.pdf", pdf, "application/pdf")
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(f"{BASE}/evaluate", data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

    print(f"POST {BASE}/evaluate ...")
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(r.read().decode())
    except Exception as e:
        print("evaluate failed:", e)
        sys.exit(1)

    students = out.get("students") or []
    print("examId", out.get("examId"), "students", len(students))
    if students:
        s0 = students[0]
        print("SN", s0.get("studentNumber"), "score", s0.get("totalScore"), "/", s0.get("totalMaxPoints"))


if __name__ == "__main__":
    main()
