"""
Alignment — bullseye detection + perspective warp.

A bullseye is 3 concentric circles (black-white-black) printed at each
of the 4 page corners by the generator. At 200 DPI scan they're roughly
35-45 px diameter.

Strategy:
  1. Look only in small corner regions (~7% of page per corner)
  2. Find the darkest compact circular blob
  3. Validate concentric structure (multi-threshold scan + clustering)
  4. If ≥3 of 4 bullseyes detected → perspective warp to target canvas
  5. If <3 → fallback to cv2.resize (log a warning)

The target canvas matches the generator's render space (756×1086 by default),
so coordinates in the map JSON apply directly to the warped image.
"""

import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

import config


# ── Bullseye detection ────────────────────────────────────────────

def find_bullseye_center(gray_region: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Find the bullseye center in a small corner region.
    Returns (cx, cy) in region-local coords, or None.
    """
    h, w = gray_region.shape
    candidates = []

    for thresh_val in (80, 100, 120, 140):
        _, binary = cv2.threshold(gray_region, thresh_val, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50 or area > w * h * 0.4:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if circularity < 0.65:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            score = circularity * math.sqrt(area)
            candidates.append((cx, cy, score, area, circularity))

    if not candidates:
        return None

    # Cluster overlapping detections (bullseye produces multiple nested contours)
    candidates.sort(key=lambda c: c[2], reverse=True)
    used = set()
    best_cluster = None
    best_score = 0.0

    for i, (x1, y1, s1, _, _) in enumerate(candidates):
        if i in used:
            continue
        cluster = [(x1, y1, s1)]
        used.add(i)
        for j, (x2, y2, s2, _, _) in enumerate(candidates):
            if j in used:
                continue
            if math.hypot(x1 - x2, y1 - y2) < 25:
                cluster.append((x2, y2, s2))
                used.add(j)

        total = sum(c[2] for c in cluster)
        if len(cluster) >= 2:
            total *= 1.5  # bonus for concentric detections
        if total > best_score:
            best_score = total
            best_cluster = cluster

    if best_cluster is None:
        return None

    avg_x = sum(c[0] for c in best_cluster) / len(best_cluster)
    avg_y = sum(c[1] for c in best_cluster) / len(best_cluster)
    return (avg_x, avg_y)


def detect_corner_bullseyes(
    image: np.ndarray,
    corner_ratio: Optional[float] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Detect bullseyes in the 4 extreme corners of the image.
    Returns dict keyed "TL"/"TR"/"BL"/"BR" in absolute image coords.
    """
    r = config.BULLSEYE_CORNER_RATIO if corner_ratio is None else corner_ratio
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    cw = max(80, int(w * r))
    ch = max(80, int(h * r))

    corners = {
        "TL": (0, 0),
        "TR": (w - cw, 0),
        "BL": (0, h - ch),
        "BR": (w - cw, h - ch),
    }

    detected: Dict[str, Tuple[float, float]] = {}
    for name, (rx, ry) in corners.items():
        region = gray[ry:ry + ch, rx:rx + cw]
        result = find_bullseye_center(region)
        if result is not None:
            gx, gy = rx + result[0], ry + result[1]
            detected[name] = (gx, gy)
            if config.VERBOSE:
                print(f"[Alignment] {name}: bullseye at ({gx:.1f}, {gy:.1f})")
        else:
            if config.VERBOSE:
                print(f"[Alignment] {name}: NOT found")

    return detected


def estimate_missing_anchor(
    matched: Dict[str, Tuple[float, float]],
    missing: str,
) -> Optional[Tuple[float, float]]:
    """Estimate a missing 4th corner via the parallelogram rule."""
    known = {k: np.array(v) for k, v in matched.items()}
    if missing == "TL" and all(k in known for k in ("TR", "BL", "BR")):
        return tuple(known["TR"] + known["BL"] - known["BR"])
    if missing == "TR" and all(k in known for k in ("TL", "BL", "BR")):
        return tuple(known["TL"] + known["BR"] - known["BL"])
    if missing == "BR" and all(k in known for k in ("TL", "TR", "BL")):
        return tuple(known["BL"] + known["TR"] - known["TL"])
    if missing == "BL" and all(k in known for k in ("TL", "TR", "BR")):
        return tuple(known["BR"] + known["TL"] - known["TR"])
    return None


# ── Perspective warp ─────────────────────────────────────────────

def _anchor_center(anchor: Dict) -> Tuple[float, float]:
    """Extract (x, y) from an anchor JSON entry."""
    if "center" in anchor:
        return (anchor["center"]["x"], anchor["center"]["y"])
    if "x" in anchor:
        return (anchor["x"], anchor["y"])
    return (0.0, 0.0)


def align_page(
    image: np.ndarray,
    anchors_json: Dict,
    page_width: Optional[int] = None,
    page_height: Optional[int] = None,
) -> np.ndarray:
    """
    Full alignment: detect bullseyes → perspective transform → deskewed output.
    On failure (<3 bullseyes), falls back to a plain resize.
    """
    pw = config.PAGE_WIDTH_PX if page_width is None else page_width
    ph = config.PAGE_HEIGHT_PX if page_height is None else page_height

    img_h, img_w = image.shape[:2]
    if config.VERBOSE:
        print(f"[Alignment] Input: {img_w}x{img_h}, target: {pw}x{ph}")

    detected = detect_corner_bullseyes(image)
    if config.VERBOSE:
        print(f"[Alignment] Found {len(detected)}/4 bullseyes")

    if len(detected) < config.BULLSEYE_MIN_COUNT:
        print("[Alignment] WARN: <3 bullseyes - falling back to resize")
        return cv2.resize(image, (pw, ph))

    if len(detected) == 3:
        missing = [c for c in ("TL", "TR", "BL", "BR") if c not in detected][0]
        est = estimate_missing_anchor(detected, missing)
        if est is None:
            return cv2.resize(image, (pw, ph))
        detected[missing] = est
        if config.VERBOSE:
            print(f"[Alignment] Estimated {missing}: ({est[0]:.1f}, {est[1]:.1f})")

    src_pts = np.array([
        detected["TL"], detected["TR"], detected["BR"], detected["BL"]
    ], dtype="float32")

    dst_pts = np.array([
        _anchor_center(anchors_json.get("TL", {})),
        _anchor_center(anchors_json.get("TR", {})),
        _anchor_center(anchors_json.get("BR", {})),
        _anchor_center(anchors_json.get("BL", {})),
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    aligned = cv2.warpPerspective(
        image, matrix, (pw, ph),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    return aligned
