"""
Exam Scanner — Bullseye Detection, Perspective Correction & Region Extraction

Bullseye = 3 concentric circles (black-white-black). 16x16px in the exam generator.
At 200 DPI scan of an A4 page, the bullseye is roughly 35-45 pixels diameter.
They sit at the extreme corners of the page (3px inset from edge in screen coords).

Detection strategy:
1. Look ONLY in small corner regions (8% of page = ~130x187px at 200DPI)
2. Find the darkest compact circular blob in each corner
3. Validate it has concentric structure (bullseye, not just a dot)
"""

import cv2
import numpy as np
import math
from typing import Dict, List, Tuple, Optional


def find_bullseye_center(gray_region: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Find the bullseye center in a small corner region.
    Strategy: the bullseye is the most prominent dark circular feature near the corner.
    """
    h, w = gray_region.shape

    # Multiple threshold levels to handle different scan qualities
    candidates = []

    for thresh_val in [80, 100, 120, 140]:
        _, binary = cv2.threshold(gray_region, thresh_val, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Bullseye at 200DPI ≈ 35-45px diameter → area 960-1590
            # Be generous: 100 to 5000
            if area < 50 or area > w * h * 0.4:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * math.pi * area / (perimeter * perimeter)
            # Strict: reject squares (circularity ~0.78) and irregular shapes
            # Real circles have circularity > 0.85
            if circularity < 0.65:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            # Score: prefer circular, compact blobs
            score = circularity * math.sqrt(area)
            candidates.append((cx, cy, score, area, circularity))

    if not candidates:
        return None

    # Cluster candidates by proximity and pick the best cluster
    candidates.sort(key=lambda c: c[2], reverse=True)

    # The bullseye produces multiple overlapping contours (outer circle, inner circle)
    # Find the cluster with the highest total score
    used = set()
    best_cluster = None
    best_score = 0

    for i, (x1, y1, s1, a1, c1) in enumerate(candidates):
        if i in used:
            continue
        cluster = [(x1, y1, s1)]
        used.add(i)
        for j, (x2, y2, s2, a2, c2) in enumerate(candidates):
            if j in used:
                continue
            if math.hypot(x1 - x2, y1 - y2) < 25:
                cluster.append((x2, y2, s2))
                used.add(j)

        total_score = sum(c[2] for c in cluster)
        # Bonus for having multiple concentric detections (bullseye signature)
        if len(cluster) >= 2:
            total_score *= 1.5

        if total_score > best_score:
            best_score = total_score
            best_cluster = cluster

    if best_cluster is None:
        return None

    # Average center of the best cluster
    avg_x = sum(c[0] for c in best_cluster) / len(best_cluster)
    avg_y = sum(c[1] for c in best_cluster) / len(best_cluster)
    return (avg_x, avg_y)


def detect_corner_bullseyes(
    image: np.ndarray,
    corner_ratio: float = 0.07,
) -> Dict[str, Tuple[float, float]]:
    """
    Detect bullseyes in the 4 extreme corners of the image.
    corner_ratio=0.07 means each corner region is 7% of width/height.
    At 1656x2342 (A4 200DPI): regions are 116x164 px — just enough for the bullseye.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    cw = max(80, int(w * corner_ratio))
    ch = max(80, int(h * corner_ratio))

    corners = {
        "TL": (0, 0),
        "TR": (w - cw, 0),
        "BL": (0, h - ch),
        "BR": (w - cw, h - ch),
    }

    detected = {}
    for name, (rx, ry) in corners.items():
        region = gray[ry:ry+ch, rx:rx+cw]
        result = find_bullseye_center(region)
        if result is not None:
            gx, gy = rx + result[0], ry + result[1]
            detected[name] = (gx, gy)
            print(f"[Scanner] {name}: bullseye at ({gx:.1f}, {gy:.1f})")
        else:
            print(f"[Scanner] {name}: NOT found")

    return detected


def estimate_missing_anchor(
    matched: Dict[str, Tuple[float, float]],
    missing: str,
) -> Optional[Tuple[float, float]]:
    """Estimate missing 4th anchor via parallelogram rule."""
    known = {k: np.array(v) for k, v in matched.items()}
    if missing == "TL" and all(k in known for k in ["TR", "BL", "BR"]):
        return tuple(known["TR"] + known["BL"] - known["BR"])
    elif missing == "TR" and all(k in known for k in ["TL", "BL", "BR"]):
        return tuple(known["TL"] + known["BR"] - known["BL"])
    elif missing == "BR" and all(k in known for k in ["TL", "TR", "BL"]):
        return tuple(known["BL"] + known["TR"] - known["TL"])
    elif missing == "BL" and all(k in known for k in ["TL", "TR", "BR"]):
        return tuple(known["BR"] + known["TL"] - known["TR"])
    return None


def align_page(
    image: np.ndarray,
    anchors_json: Dict,
    page_width: int,
    page_height: int,
) -> np.ndarray:
    """
    Full alignment: detect bullseyes → perspective transform → deskewed output.
    """
    img_h, img_w = image.shape[:2]
    print(f"[Scanner] Input: {img_w}x{img_h}, target: {page_width}x{page_height}")

    detected = detect_corner_bullseyes(image)
    print(f"[Scanner] Found {len(detected)}/4 bullseyes")

    if len(detected) < 3:
        print("[Scanner] ERROR: < 3 bullseyes. Falling back to resize.")
        return cv2.resize(image, (page_width, page_height))

    if len(detected) == 3:
        missing = [c for c in ["TL", "TR", "BL", "BR"] if c not in detected][0]
        est = estimate_missing_anchor(detected, missing)
        if est:
            detected[missing] = est
            print(f"[Scanner] Estimated {missing}: ({est[0]:.1f}, {est[1]:.1f})")
        else:
            return cv2.resize(image, (page_width, page_height))

    src_pts = np.array([
        detected["TL"], detected["TR"], detected["BR"], detected["BL"]
    ], dtype="float32")

    dst_pts = np.array([
        _get_center(anchors_json.get("TL", {})),
        _get_center(anchors_json.get("TR", {})),
        _get_center(anchors_json.get("BR", {})),
        _get_center(anchors_json.get("BL", {})),
    ], dtype="float32")

    print(f"[Scanner] SRC: {[f'({p[0]:.0f},{p[1]:.0f})' for p in src_pts]}")
    print(f"[Scanner] DST: {[f'({p[0]:.0f},{p[1]:.0f})' for p in dst_pts]}")

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    aligned = cv2.warpPerspective(
        image, matrix, (page_width, page_height),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )

    # Debug: save with markers
    debug = aligned.copy()
    for name, pt in [("TL", dst_pts[0]), ("TR", dst_pts[1]), ("BR", dst_pts[2]), ("BL", dst_pts[3])]:
        cv2.circle(debug, (int(pt[0]), int(pt[1])), 12, (0, 255, 0), 2)
    cv2.imwrite("output/debug_aligned_markers.jpg", debug)

    return aligned


def extract_region(image: np.ndarray, box: Dict) -> np.ndarray:
    """Extract rectangular region. Box: {"x","y","w","h"}"""
    x = max(0, int(round(box.get("x", 0))))
    y = max(0, int(round(box.get("y", 0))))
    w = int(round(box.get("w", 0)))
    h = int(round(box.get("h", 0)))
    x = min(x, image.shape[1] - 1)
    y = min(y, image.shape[0] - 1)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    if w <= 0 or h <= 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return image[y:y+h, x:x+w]


def check_bubble_filled(bubble_img: np.ndarray, threshold: float = 0.4) -> bool:
    """Check if MC/MS bubble is filled by dark pixel ratio."""
    if bubble_img.size == 0:
        return False
    gray = cv2.cvtColor(bubble_img, cv2.COLOR_BGR2GRAY) if len(bubble_img.shape) == 3 else bubble_img
    _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    return float(np.sum(binary > 0) / binary.size) > threshold


def _get_center(anchor: Dict) -> Tuple[float, float]:
    """Extract center from anchor JSON."""
    if "center" in anchor:
        return (anchor["center"]["x"], anchor["center"]["y"])
    elif "x" in anchor:
        return (anchor["x"], anchor["y"])
    return (0.0, 0.0)
