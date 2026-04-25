"""
Alignment - bullseye detection + perspective warp + (debug) overlay.

Pipeline tiers:

  1. Anchor detection (hybrid)
     a. ROI-based search at JSON-expected position (per corner, primary).
        Hough Circle in a small window around the expected pixel.
     b. If a corner is missed, fall back to general corner-region scan
        (multi-threshold contour clustering, the original approach).
     -> Best of both worlds: precision when the scan is mild, robustness when wild.

  2. Geometric transform (three-tier)
     - 4 anchors detected -> RANSAC homography (outlier-resistant)
     - 2-3 anchors        -> affine median offset (uniform drift correction)
     - <2 anchors         -> pure resize (warning, no correction)

  3. (Optional) Coordinate-only mode via TransformContext, used by the
     /debug/annotate endpoint to overlay JSON regions on the original scan
     without warping the pixels.

Design intent: the scanned image is warped into the JSON canonical space
(default 756x1086) so downstream readers can use map coordinates directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config


# ══════════════════════════════════════════════════════════════════
#  Transform context
# ══════════════════════════════════════════════════════════════════

@dataclass
class TransformContext:
    """
    Maps canonical (JSON) coordinates to scanned-pixel coordinates.

    Forward direction is canonical -> scanned. This matches the natural
    "where does this JSON box land on the scan?" question used by the debug
    overlay. For warping the scanned image into canonical space we pass this
    matrix to OpenCV with WARP_INVERSE_MAP, since canonical->scanned IS the
    inverse of the warp's normal src->dst direction.

    Modes:
      "homography"    - 4+ anchors, RANSAC perspective
      "affine_median" - 2-3 anchors, scale + median offset (uniform drift)
      "scale"         - <2 anchors, pure scale (no correction)
    """

    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    homography: Optional[np.ndarray] = field(default=None, repr=False)
    mode: str = "scale"
    n_anchors_detected: int = 0
    detected_anchors: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def apply_point(self, x: float, y: float) -> Tuple[int, int]:
        """Map a canonical (x, y) to a scanned pixel (px, py) - integer rounded."""
        if self.homography is not None:
            src = np.array([[[x, y]]], dtype=np.float64)
            dst = cv2.perspectiveTransform(src, self.homography)
            return int(round(dst[0, 0, 0])), int(round(dst[0, 0, 1]))
        return (
            int(round(x * self.scale_x + self.offset_x)),
            int(round(y * self.scale_y + self.offset_y)),
        )

    def apply_rect(
        self, x: float, y: float, w: float, h: float
    ) -> Tuple[int, int, int, int]:
        """Map a canonical (x, y, w, h) rect to scanned (x1, y1, x2, y2)."""
        x1, y1 = self.apply_point(x, y)
        x2, y2 = self.apply_point(x + w, y + h)
        return x1, y1, x2, y2


# ══════════════════════════════════════════════════════════════════
#  Anchor detection - bullseye finders
# ══════════════════════════════════════════════════════════════════

def find_bullseye_center(gray_region: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Find the bullseye center in a small region (legacy method).
    Multi-threshold contour scan with cluster scoring.

    Returns (cx, cy) in region-local coords, or None.
    """
    h, w = gray_region.shape
    candidates: List[Tuple[float, float, float, float, float]] = []

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
    best_cluster: Optional[List[Tuple[float, float, float]]] = None
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


def detect_anchor_in_roi(
    gray: np.ndarray,
    expected_px: int,
    expected_py: int,
    search_radius: int,
) -> Optional[Tuple[float, float]]:
    """
    Search for the bullseye anchor center near (expected_px, expected_py).

    The generator's bullseye is 16px wide in canonical space. At 200 DPI scan
    (~2.2x scale), that's ~35px outer diameter, which means the visible
    circular structures Hough can latch onto are radius ~4-17px (inner dot,
    middle white ring, outer black ring).

    Strategy:
      1. Hough Circle Transform within ROI, RESTRICTED TO BULLSEYE-SIZED RADII.
         Critically, prefer the circle CLOSEST to the JSON-expected position,
         not the most prominent — the most prominent in a corner ROI may be
         a logo, QR border, or scan artifact.
      2. Largest-contour centroid fallback (low-contrast / occluded markers),
         again preferring the contour whose centroid is closest to expected.

    Returns absolute pixel coords if found, else None.
    """
    img_h, img_w = gray.shape
    roi_x1 = max(0, expected_px - search_radius)
    roi_y1 = max(0, expected_py - search_radius)
    roi_x2 = min(img_w, expected_px + search_radius)
    roi_y2 = min(img_h, expected_py + search_radius)

    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        return None

    roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return None

    # Local expected position within the ROI (for "closest to expected" scoring)
    local_expected_x = expected_px - roi_x1
    local_expected_y = expected_py - roi_y1

    blurred = cv2.GaussianBlur(roi, (5, 5), 0)

    # --- Strategy 1: Hough Circle Transform (bullseye-sized only) ---
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=10,
        param1=60,
        param2=18,
        minRadius=4,
        maxRadius=20,  # bullseye outer ring is ~17px @ 200dpi; cap below logo size
    )
    if circles is not None:
        # CLOSEST to expected position — guards against a logo/QR fragment
        # that might be a more prominent circle in the same ROI.
        best = min(
            circles[0],
            key=lambda c: math.hypot(c[0] - local_expected_x, c[1] - local_expected_y),
        )
        return float(roi_x1 + best[0]), float(roi_y1 + best[1])

    # --- Strategy 2: Largest-contour centroid fallback ---
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        valid = []
        for c in contours:
            area = cv2.contourArea(c)
            # Bullseye blob area at 200dpi ≈ pi*17^2 ≈ 900 px²; allow generous range
            if area < 30 or area > 3000:
                continue
            perim = cv2.arcLength(c, True)
            if perim == 0:
                continue
            circularity = 4 * math.pi * area / (perim * perim)
            if circularity < 0.5:
                continue
            mom = cv2.moments(c)
            if mom["m00"] > 0:
                cx = mom["m10"] / mom["m00"]
                cy = mom["m01"] / mom["m00"]
                dist = math.hypot(cx - local_expected_x, cy - local_expected_y)
                valid.append((cx, cy, dist, area))
        if valid:
            # Prefer the contour CLOSEST to expected position
            best = min(valid, key=lambda v: v[2])
            return float(roi_x1 + best[0]), float(roi_y1 + best[1])

    return None


def detect_corner_bullseyes(
    image: np.ndarray,
    corner_ratio: Optional[float] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    General corner-region scan (legacy; used as fallback when ROI fails).
    Looks for bullseye-like blobs in 7%-of-page corner windows.
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
            detected[name] = (rx + result[0], ry + result[1])

    return detected


def detect_anchors_hybrid(
    image: np.ndarray,
    anchors_json: Dict[str, Dict],
    scale_x: float,
    scale_y: float,
) -> Dict[str, Tuple[float, float]]:
    """
    Anchor detection — uses the legacy general corner-area scan.

    NOTE: We tried a per-corner ROI Hough Circle search ("v6 ExamLayoutParser
    style") but it was less reliable on real scans — Hough on tiny ROIs
    occasionally locked onto logo fragments / QR borders / scan artifacts
    rather than the actual bullseye, producing wildly wrong anchor positions
    that broke alignment for some students. The legacy multi-threshold
    contour clustering (find_bullseye_center over a 7%-corner area) is more
    forgiving and proven on these scans.

    Args `scale_x`/`scale_y` and `anchors_json` are accepted for API
    compatibility (in case future hybrid variants want them); the current
    implementation ignores them.
    """
    return detect_corner_bullseyes(image)


def estimate_missing_anchor(
    matched: Dict[str, Tuple[float, float]],
    missing: str,
) -> Optional[Tuple[float, float]]:
    """
    Estimate a missing 4th corner via the parallelogram rule.
    Kept as a utility; align_page no longer uses it (affine median is used
    for the 2-3 anchor case instead).
    """
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


def _anchor_center(anchor: Dict) -> Tuple[float, float]:
    """Extract (x, y) from an anchor JSON entry (handles both schema variants)."""
    if "center" in anchor and isinstance(anchor["center"], dict):
        return (float(anchor["center"].get("x", 0)), float(anchor["center"].get("y", 0)))
    if "x" in anchor:
        return (float(anchor["x"]), float(anchor["y"]))
    return (0.0, 0.0)


# ══════════════════════════════════════════════════════════════════
#  Transform computation
# ══════════════════════════════════════════════════════════════════

def align_page(
    image: np.ndarray,
    anchors_json: Dict,
    page_width: Optional[int] = None,
    page_height: Optional[int] = None,
) -> np.ndarray:
    """
    Warp the scanned image into canonical (JSON) coordinate space.

    Bit-identical to the original (commit 09210dc) implementation:

      1. detect_corner_bullseyes: legacy multi-threshold contour scan in the
         four 7%-of-page corner regions.
      2. If <3 anchors found, fall back to plain cv2.resize (warning logged).
      3. If exactly 3 anchors, estimate the 4th via the parallelogram rule.
      4. With 4 anchors, getPerspectiveTransform + warpPerspective.

    The earlier RANSAC + affine-median + ROI-Hough hybrid was reverted because
    on real student scans it occasionally locked onto wrong features (logo,
    QR border, scan artifacts) and broke per-box cropping for the affected
    students. The 7% corner-area scan is more forgiving and proven.
    """
    pw = config.PAGE_WIDTH_PX if page_width is None else page_width
    ph = config.PAGE_HEIGHT_PX if page_height is None else page_height

    img_h, img_w = image.shape[:2]
    if config.VERBOSE:
        print(f"[Alignment] Input: {img_w}x{img_h}, target: {pw}x{ph}")

    detected = detect_corner_bullseyes(image)
    if config.VERBOSE:
        print(f"[Alignment] Found {len(detected)}/4 bullseyes")

    if len(detected) < 3:
        print("[Alignment] WARN: <3 bullseyes - falling back to resize")
        return cv2.resize(image, (pw, ph))

    if len(detected) == 3:
        missing = next(c for c in ("TL", "TR", "BL", "BR") if c not in detected)
        est = estimate_missing_anchor(detected, missing)
        if est is None:
            return cv2.resize(image, (pw, ph))
        detected = dict(detected)
        detected[missing] = est
        if config.VERBOSE:
            print(f"[Alignment] Estimated {missing}: ({est[0]:.1f}, {est[1]:.1f})")

    src_pts = np.array([
        detected["TL"], detected["TR"], detected["BR"], detected["BL"],
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


def compute_transform_context(
    image: np.ndarray,
    anchors_json: Dict[str, Dict],
    page_width: int,
    page_height: int,
) -> TransformContext:
    """
    Build a TransformContext for the /debug/annotate endpoint.

    Uses the same legacy detection as align_page, but stores the inverse
    transform (canonical -> scanned) so apply_point/apply_rect can place
    JSON regions onto the original scanned image without warping pixels.

    Pipeline never calls this — only the debug overlay does.
    """
    img_h, img_w = image.shape[:2]
    scale_x = img_w / page_width
    scale_y = img_h / page_height

    detected = detect_corner_bullseyes(image)
    n = len(detected)

    ctx = TransformContext(
        scale_x=scale_x,
        scale_y=scale_y,
        n_anchors_detected=n,
        detected_anchors=dict(detected),
    )

    # 3 -> estimate 4th
    if n == 3:
        missing = next(c for c in ("TL", "TR", "BL", "BR") if c not in detected)
        est = estimate_missing_anchor(detected, missing)
        if est is not None:
            detected = dict(detected)
            detected[missing] = est
            n = 4

    if n >= 4 and all(c in detected for c in ("TL", "TR", "BL", "BR")):
        # scanned -> canonical via getPerspectiveTransform
        src_pts = np.array([
            detected["TL"], detected["TR"], detected["BR"], detected["BL"],
        ], dtype=np.float32)
        dst_pts = np.array([
            _anchor_center(anchors_json.get("TL", {})),
            _anchor_center(anchors_json.get("TR", {})),
            _anchor_center(anchors_json.get("BR", {})),
            _anchor_center(anchors_json.get("BL", {})),
        ], dtype=np.float32)
        try:
            M_scan_to_canon = cv2.getPerspectiveTransform(src_pts, dst_pts)
            # apply_point needs canonical -> scanned, so invert
            ctx.homography = np.linalg.inv(M_scan_to_canon)
            ctx.mode = "homography"
            return ctx
        except (cv2.error, np.linalg.LinAlgError) as e:
            print(f"[Alignment] WARN: transform context failed: {e}")

    # Fallback: pure scale
    ctx.mode = "scale"
    return ctx


# ══════════════════════════════════════════════════════════════════
#  Debug overlay (used by /debug/annotate)
# ══════════════════════════════════════════════════════════════════

class _Color:
    """BGR colors for annotated overlays (matches ExamLayoutParser palette)."""
    ANCHOR = (0, 0, 220)            # Red
    STUDENT_REGION = (180, 0, 180)  # Purple
    STUDENT_BOX = (220, 50, 220)    # Light purple
    QUESTION_BOX = (0, 180, 0)      # Green
    OPTION = (220, 100, 0)          # Blue
    SOLUTION_AREA = (0, 140, 255)   # Orange
    MATCHING_BOX = (0, 220, 220)    # Yellow
    FILL_BLANK = (100, 255, 100)    # Light green
    LABEL_BG = (30, 30, 30)         # Dark grey
    LABEL_TEXT = (255, 255, 255)    # White


def _draw_rect(
    image: np.ndarray,
    x: float, y: float, w: float, h: float,
    ctx: TransformContext,
    color: Tuple[int, int, int],
    thickness: int = 2,
    fill_alpha: float = 0.0,
) -> Tuple[int, int]:
    """Draw a calibrated rectangle. Returns (px1, py1) for label positioning."""
    x1, y1, x2, y2 = ctx.apply_rect(x, y, w, h)
    if fill_alpha > 0.0:
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, cv2.FILLED)
        cv2.addWeighted(overlay, fill_alpha, image, 1.0 - fill_alpha, 0, image)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return x1, y1


def _draw_bullseye(
    image: np.ndarray,
    cx: float, cy: float, diameter: float,
    ctx: TransformContext,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> Tuple[int, int]:
    """Draw a calibrated bullseye marker. Returns center pixel coords."""
    px, py = ctx.apply_point(cx, cy)
    avg_scale = (ctx.scale_x + ctx.scale_y) / 2.0
    radius = max(3, int(round((diameter / 2.0) * avg_scale)))
    cv2.circle(image, (px, py), radius, color, thickness)
    cv2.circle(image, (px, py), max(2, radius // 3), color, cv2.FILLED)
    return px, py


def _draw_label(
    image: np.ndarray,
    text: str,
    px: int, py: int,
    text_color: Tuple[int, int, int] = _Color.LABEL_TEXT,
    bg_color: Tuple[int, int, int] = _Color.LABEL_BG,
    font_scale: float = 0.45,
    thickness: int = 1,
) -> None:
    """Render a small label with a solid background just above (px, py)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    baseline_y = py - 4
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(
        image,
        (px - 1, baseline_y - th - bl),
        (px + tw + 1, baseline_y + bl),
        bg_color,
        cv2.FILLED,
    )
    cv2.putText(
        image, text, (px, baseline_y), font, font_scale, text_color, thickness, cv2.LINE_AA,
    )


def annotate_page(scanned_image: np.ndarray, page_data: Dict) -> Tuple[np.ndarray, TransformContext]:
    """
    Draw the JSON-defined regions on the original scanned image, color-coded.

    Used by the /debug/annotate endpoint to verify the alignment and the
    correctness of the JSON map BEFORE running an evaluation. The image is
    NOT warped - annotations are placed using the canonical->scanned
    transform context.

    Returns: (annotated_image, transform_context)
    """
    pw = int(page_data.get("pageWidth", config.PAGE_WIDTH_PX))
    ph = int(page_data.get("pageHeight", config.PAGE_HEIGHT_PX))
    anchors_json = page_data.get("anchors", {}) or {}

    ctx = compute_transform_context(scanned_image, anchors_json, pw, ph)
    image = scanned_image.copy()

    # ---- Anchors ----
    for corner_name, anchor_data in anchors_json.items():
        ref_x, ref_y = _anchor_center(anchor_data)
        diameter = float(anchor_data.get("diameter", 16))
        px, py = _draw_bullseye(image, ref_x, ref_y, diameter, ctx, _Color.ANCHOR)
        avg_scale = (ctx.scale_x + ctx.scale_y) / 2.0
        radius = max(3, int(round((diameter / 2.0) * avg_scale)))
        _draw_label(image, f"ANC-{corner_name}", px, py - radius - 4)

    # ---- Student number region + boxes ----
    sn_region = page_data.get("studentNumberRegion")
    if sn_region:
        try:
            px, py = _draw_rect(
                image,
                sn_region["x"], sn_region["y"], sn_region["w"], sn_region["h"],
                ctx, _Color.STUDENT_REGION, thickness=2, fill_alpha=0.05,
            )
            _draw_label(image, "STUDENT_REGION", px, py)
        except KeyError:
            pass

    for digit_box in page_data.get("studentNumberBoxes", []) or []:
        try:
            _draw_rect(
                image,
                digit_box["x"], digit_box["y"], digit_box["w"], digit_box["h"],
                ctx, _Color.STUDENT_BOX, thickness=1,
            )
        except KeyError:
            pass

    # ---- Questions ----
    questions = page_data.get("questions", {}) or {}
    for q_num, q_data in questions.items():
        if not isinstance(q_data, dict):
            continue
        q_type = q_data.get("type", "unknown")
        bbox = q_data.get("boundingBox")
        if bbox:
            try:
                px, py = _draw_rect(
                    image,
                    bbox["x"], bbox["y"], bbox["w"], bbox["h"],
                    ctx, _Color.QUESTION_BOX, thickness=3, fill_alpha=0.04,
                )
                _draw_label(image, f"Q{q_num} [{q_type}]", px, py)
            except KeyError:
                pass

        # Type-specific
        if q_type in ("multiple_choice", "multi_select"):
            for opt_key, opt_box in (q_data.get("options") or {}).items():
                try:
                    opx, opy = _draw_rect(
                        image,
                        opt_box["x"], opt_box["y"], opt_box["w"], opt_box["h"],
                        ctx, _Color.OPTION, thickness=1,
                    )
                    _draw_label(image, opt_key, opx, opy, font_scale=0.35)
                except KeyError:
                    pass

        elif q_type == "open_ended":
            sa = q_data.get("solutionArea")
            if sa:
                try:
                    spx, spy = _draw_rect(
                        image,
                        sa["x"], sa["y"], sa["w"], sa["h"],
                        ctx, _Color.SOLUTION_AREA, thickness=2, fill_alpha=0.06,
                    )
                    _draw_label(image, f"Q{q_num} SOLUTION", spx, spy)
                except KeyError:
                    pass

        elif q_type == "matching":
            ans_section = q_data.get("answerSection")
            if ans_section:
                try:
                    _draw_rect(
                        image,
                        ans_section["x"], ans_section["y"],
                        ans_section["w"], ans_section["h"],
                        ctx, _Color.MATCHING_BOX, thickness=1, fill_alpha=0.04,
                    )
                except KeyError:
                    pass
            for slot, slot_box in (q_data.get("answerBoxes") or {}).items():
                try:
                    bpx, bpy = _draw_rect(
                        image,
                        slot_box["x"], slot_box["y"],
                        slot_box["w"], slot_box["h"],
                        ctx, _Color.MATCHING_BOX, thickness=2,
                    )
                    _draw_label(image, f"M{slot}", bpx, bpy, font_scale=0.38)
                except KeyError:
                    pass

        elif q_type == "fill_blanks":
            ans_section = q_data.get("answerSection")
            if ans_section:
                try:
                    _draw_rect(
                        image,
                        ans_section["x"], ans_section["y"],
                        ans_section["w"], ans_section["h"],
                        ctx, _Color.FILL_BLANK, thickness=1, fill_alpha=0.04,
                    )
                except KeyError:
                    pass
            for blank_id, blank_box in (q_data.get("fillBlanks") or {}).items():
                try:
                    fpx, fpy = _draw_rect(
                        image,
                        blank_box["x"], blank_box["y"],
                        blank_box["w"], blank_box["h"],
                        ctx, _Color.FILL_BLANK, thickness=2,
                    )
                    _draw_label(image, f"B{blank_id}", fpx, fpy, font_scale=0.38)
                except KeyError:
                    pass
            for box_id, ans_box in (q_data.get("answerBoxes") or {}).items():
                try:
                    _draw_rect(
                        image,
                        ans_box["x"], ans_box["y"], ans_box["w"], ans_box["h"],
                        ctx, _Color.FILL_BLANK, thickness=2,
                    )
                except KeyError:
                    pass

    # Footer banner showing transform mode
    banner = (
        f"transform={ctx.mode}  anchors={ctx.n_anchors_detected}/4  "
        f"sx={ctx.scale_x:.3f} sy={ctx.scale_y:.3f}"
    )
    _draw_label(image, banner, 10, image.shape[0] - 10, font_scale=0.5, thickness=1)

    return image, ctx
