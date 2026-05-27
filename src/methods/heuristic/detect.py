"""Heuristic grasp detector — production refactor.

This module is the headless re-implementation of the grasp-detection pipeline
originally prototyped as a Streamlit app in
``Heuristics approach/grasp_detection_contour_80px.py``. The original file
mixes algorithmic logic with UI/plotting code (Streamlit, Plotly, file
upload widgets); here we keep only the math.

The algorithm is essentially a depth-driven contour-tangent grasp detector:

* Stage 0 — mask construction (depth percentile + Canny + morphology).
* Stage 1 — uniform contour sampling, candidates ranked by a 3-term score
  (edge response + depth-gradient magnitude + proximity to CoG).
* Stage 2 — for each top-K candidate, an analytic 2x2 PCA on a ~80 px
  contour segment gives the local tangent. The grasp direction is the
  perpendicular to that tangent. A ray is cast bidirectionally along the
  grasp direction until it leaves the binary mask (with a small gap
  tolerance for noise).
* Stage 3 — rank surviving grasps by ``line_length - cog_boost * proximity
  * 500`` (lower = better) when running the grid-search-winning
  ``ray_algorithm == "Direct Line with CoG Boost"`` configuration; output
  the top-N as ``GraspRect`` instances with width = measured line length
  and height = the contour's robust extent perpendicular to the ray.

The 80 px PCA-tangent block is the classical contour-tangent grasp-planning
trick used by Morales (2001) "Heuristic Vision-Based Computation of Planar
Antipodal Grasps", Sanz (1999) "Vision-Guided Grasping of Unknown Objects
for Service Robots", and Lei (2017) "Fast Grasping of Unknown Objects Based
on Principal Component Analysis": for a contour patch around a candidate
point, the first principal component of the patch coordinates approximates
the tangent direction, and the perpendicular is the antipodal grasp axis.

Output-convention fix (2026-05-27)
----------------------------------

The original Streamlit prototype emitted ``GraspRect`` with
``width = ray_length`` and ``height = 20.0`` (a fixed jaw-thickness
placeholder). This breaks the Cornell GraspRect convention defined in
``src.data.cornell_loader._corners_to_grasp_rect`` and consumed by
``src.eval.cornell``:

* Cornell's ``width`` is the gripper-opening size (the SHORTER rectangle
  side) — that matches our ``ray_length`` and is unchanged.
* Cornell's ``height`` is the jaw-plate length (the LONGER rectangle side),
  typically 50-200 px for elongated objects — NOT a fixed 20-px placeholder.
* Cornell's ``angle_rad`` is the direction of the gripper-opening axis,
  wrapped to ``[-pi/2, pi/2]`` — our ray direction is the opening axis but
  the raw ``atan2`` output is not wrapped.

We fix the height by measuring the contour's robust extent (5th-to-95th
percentile spread) perpendicular to the ray direction, lower-bounded at
30 px so degenerate cases do not produce zero-area rectangles. We wrap the
angle to ``[-pi/2, pi/2]`` to match the loader. This is a deliberate,
documented correction of an output-convention bug — not a change to the
detection algorithm. Where the grasp goes (CoG-biased contour-tangent
ray-cast) is unchanged.

All hyperparameters live on ``HeuristicConfig`` (see ``config.py``); this
module reads them and does not introduce hidden magic numbers.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from src.eval.cornell import GraspRect
from src.methods.heuristic.config import HeuristicConfig


# Lower bound on the perpendicular extent used as the rectangle's `height`
# (jaw-plate length). Prevents zero/near-zero-area rectangles in degenerate
# cases where the contour is essentially a line.
_MIN_HEIGHT_PX: float = 30.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_grasp(
    image_rgb: np.ndarray,
    depth: np.ndarray,
    config: HeuristicConfig,
) -> list[GraspRect]:
    """Return top-N grasp rectangles ranked best-first.

    Parameters
    ----------
    image_rgb : np.ndarray
        HxWx3 uint8 RGB image.
    depth : np.ndarray
        HxW float32 depth map, expected in [0, 1] (matches the output of
        :class:`src.methods.heuristic.depth.DepthEstimator`).
    config : HeuristicConfig
        All hyperparameters. See ``config.py`` for the grid-search winners.

    Returns
    -------
    list[GraspRect]
        Up to ``config.num_output_grasps`` grasp rectangles, ranked best
        first. May be empty if the mask is degenerate or all candidates
        are rejected by the length filter.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(
            f"image_rgb must be HxWx3, got shape {image_rgb.shape}"
        )
    if image_rgb.dtype != np.uint8:
        raise ValueError(
            f"image_rgb must be uint8, got dtype {image_rgb.dtype}"
        )
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got shape {depth.shape}")
    if depth.shape != image_rgb.shape[:2]:
        raise ValueError(
            f"depth shape {depth.shape} does not match image shape "
            f"{image_rgb.shape[:2]}"
        )

    h, w = image_rgb.shape[:2]

    # --- Stage 0: mask + auxiliary maps ----------------------------------
    mask, contour, grad_x, grad_y, _ = _build_object_mask(
        image_rgb, depth.astype(np.float32), config.depth_percentile
    )

    if contour is None or np.count_nonzero(mask) == 0:
        return []

    # Normalised depth-gradient magnitude (stage 1 "depth_quality" channel).
    depth_grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    dg_min = float(depth_grad_mag.min())
    dg_max = float(depth_grad_mag.max())
    depth_grad_norm = (depth_grad_mag - dg_min) / (dg_max - dg_min + 1e-8)

    # Canny edge response on greyscale RGB (stage 1 "edge_quality" channel).
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges_raw = cv2.Canny(gray, 50, 150)
    edges_normalized = edges_raw.astype(np.float32) / 255.0

    # Centre of gravity of the binary mask.
    cog_x, cog_y = _mask_centroid(mask, w, h)

    # --- Stage 1: candidate scoring --------------------------------------
    candidate_points = _sample_contour_points(contour, config.num_contour_samples)
    if len(candidate_points) == 0:
        return []

    max_dist_scene = float(np.sqrt(w * w + h * h))
    preliminary = _score_candidates(
        candidate_points,
        edges_normalized,
        depth_grad_norm,
        cog_x,
        cog_y,
        max_dist_scene,
        w,
        h,
        config.w_edge,
        config.w_depth,
        config.w_cog,
    )
    if not preliminary:
        return []

    preliminary.sort(key=lambda c: c["combined_quality"], reverse=True)
    n_top = max(1, config.num_output_grasps * config.candidate_multiplier)
    top_candidates = preliminary[:n_top]

    # --- Stage 2: ray casting --------------------------------------------
    survivors: list[dict] = []
    for cand in top_candidates:
        px, py = cand["point"]

        if config.ray_algorithm == "Through CoG":
            end1_x, end1_y, end2_x, end2_y, line_length = _cast_ray_through_cog(
                mask,
                (px, py),
                (cog_x, cog_y),
                config.ray_max_dist,
                config.ray_gap_tolerance,
            )
        else:
            # "Direct Line with CoG Boost" (grid-search winner).
            end1_x, end1_y, end2_x, end2_y, line_length = _cast_ray_direct_line(
                mask,
                (px, py),
                grad_x,
                grad_y,
                contour,
                config.gradient_source,
                config.ray_max_dist,
                config.ray_gap_tolerance,
            )

        # Length filter (mirrors the Streamlit script's <=/>= semantics).
        if line_length <= config.min_grasp_length:
            continue
        if line_length >= config.max_grasp_length:
            continue

        # Grasp geometry: midpoint and orientation from the ray endpoints.
        dx_line = end1_x - end2_x
        dy_line = end1_y - end2_y
        angle = float(np.arctan2(dy_line, dx_line))
        # Wrap to [-pi/2, pi/2] (parallel-jaw 180-deg symmetry, matches the
        # Cornell loader convention in `_corners_to_grasp_rect`).
        while angle > math.pi / 2:
            angle -= math.pi
        while angle < -math.pi / 2:
            angle += math.pi
        cx_mid = (end1_x + end2_x) / 2.0
        cy_mid = (end1_y + end2_y) / 2.0

        # Perpendicular extent of the contour around this grasp (jaw-plate
        # length). See `_perpendicular_extent` docstring and the
        # module-level "Output-convention fix" note.
        perp_height = _perpendicular_extent(
            contour, cx_mid, cy_mid, angle, _MIN_HEIGHT_PX
        )

        # Stage-3 rank score.
        if config.ray_algorithm == "Direct Line with CoG Boost":
            dist_to_cog = float(
                np.sqrt((cx_mid - cog_x) ** 2 + (cy_mid - cog_y) ** 2)
            )
            proximity_score = 1.0 - (dist_to_cog / max_dist_scene)
            rank_score = line_length - (config.cog_boost * proximity_score * 500.0)
        else:
            rank_score = float(line_length)

        survivors.append(
            {
                "x": cx_mid,
                "y": cy_mid,
                "angle": angle,
                "line_length": float(line_length),
                "perp_height": float(perp_height),
                "rank_score": float(rank_score),
            }
        )

    # --- Stage 3: rank + output ------------------------------------------
    # Ascending sort: lower rank_score = better grasp.
    survivors.sort(key=lambda g: g["rank_score"])

    output: list[GraspRect] = []
    for g in survivors[: config.num_output_grasps]:
        output.append(
            GraspRect(
                x=float(g["x"]),
                y=float(g["y"]),
                angle_rad=float(g["angle"]),
                width=float(g["line_length"]),
                height=float(g["perp_height"]),
            )
        )
    return output


# ---------------------------------------------------------------------------
# Output-convention helpers
# ---------------------------------------------------------------------------

def _perpendicular_extent(
    contour: np.ndarray,
    cx: float,
    cy: float,
    ray_angle_rad: float,
    min_extent: float,
) -> float:
    """Robust extent of the contour perpendicular to ``ray_angle_rad``.

    The output rectangle's ``width`` is the ray length along the grasp
    (opening) axis. To match the Cornell GraspRect convention (see
    ``src.data.cornell_loader._corners_to_grasp_rect``) the ``height`` field
    must be the jaw-plate length — i.e., the object's extent along the axis
    perpendicular to the opening. We approximate that here as the 5th-to-95th
    percentile spread of the contour's projection onto the perpendicular
    axis at the grasp centre.

    Lower-bounded at ``min_extent`` to avoid zero-area rectangles when the
    contour is degenerate (single point, near-collinear).
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    if pts.shape[0] == 0:
        return float(min_extent)
    # Perpendicular axis unit vector: ray direction rotated +90 deg.
    perp_x = -math.sin(ray_angle_rad)
    perp_y = math.cos(ray_angle_rad)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    projections = dx * perp_x + dy * perp_y
    if projections.size < 2:
        return float(min_extent)
    lo = float(np.percentile(projections, 5.0))
    hi = float(np.percentile(projections, 95.0))
    extent = hi - lo
    if not math.isfinite(extent) or extent <= 0.0:
        return float(min_extent)
    return float(max(extent, min_extent))


# ---------------------------------------------------------------------------
# Stage 0 — mask construction
# ---------------------------------------------------------------------------

def _build_object_mask(
    image_rgb: np.ndarray,
    depth_map: np.ndarray,
    depth_percentile: int,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Hybrid depth+edge saliency mask (port of ``get_combined_saliency_mask_hybrid``).

    Returns ``(object_mask, object_contour, grad_x, grad_y, edges_in_roi)``.
    """
    # 1. Depth-based ROI.
    depth_threshold = float(np.percentile(depth_map, depth_percentile))
    depth_roi = (depth_map >= depth_threshold).astype(np.uint8) * 255

    kernel_depth = np.ones((7, 7), np.uint8)
    depth_roi = cv2.morphologyEx(depth_roi, cv2.MORPH_CLOSE, kernel_depth)
    depth_roi = cv2.morphologyEx(depth_roi, cv2.MORPH_OPEN, kernel_depth)

    # 2. Edge detection on greyscale RGB.
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # 3. Edges restricted to the depth ROI, then dilated.
    edges_in_roi = cv2.bitwise_and(edges, edges, mask=depth_roi)
    kernel_dilate = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges_in_roi, kernel_dilate, iterations=3)

    # 4. Weighted blend (0.2 depth + 0.8 edges) thresholded at 100.
    depth_contribution = depth_roi.astype(np.float32) * 0.2
    edge_contribution = edges_dilated.astype(np.float32) * 0.8
    combined_blend = (depth_contribution + edge_contribution).astype(np.uint8)
    _, binary_mask = cv2.threshold(combined_blend, 100, 255, cv2.THRESH_BINARY)

    # 5. Morphological cleanup.
    kernel_close = np.ones((7, 7), np.uint8)
    kernel_open = np.ones((5, 5), np.uint8)
    object_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel_open)

    # 6. Pick the dominant contour by area (with 0.5%-40% size sanity bounds).
    contours, _ = cv2.findContours(
        object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    object_contour: Optional[np.ndarray] = None
    if contours:
        total_area = object_mask.shape[0] * object_mask.shape[1]
        valid_contours = [
            c
            for c in contours
            if 0.005 * total_area < cv2.contourArea(c) < 0.40 * total_area
        ]
        if valid_contours:
            object_contour = max(valid_contours, key=cv2.contourArea)
        else:
            object_contour = max(contours, key=cv2.contourArea)

        # Re-rasterise the mask as the filled selected contour.
        object_mask = np.zeros_like(object_mask)
        cv2.drawContours(
            object_mask, [object_contour], -1, 255, thickness=cv2.FILLED
        )

    # 7. Depth gradients (used by the "Depth Gradients" ray method).
    depth_uint8 = (depth_map * 255.0).astype(np.uint8)
    grad_x = cv2.Sobel(depth_uint8, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(depth_uint8, cv2.CV_64F, 0, 1, ksize=5)

    return object_mask, object_contour, grad_x, grad_y, edges_in_roi


def _mask_centroid(mask: np.ndarray, w: int, h: int) -> tuple[int, int]:
    """Return the integer image-moment centroid of the mask.

    Falls back to the image centre for an empty mask.
    """
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        return (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )
    return w // 2, h // 2


# ---------------------------------------------------------------------------
# Stage 1 — candidate sampling and scoring
# ---------------------------------------------------------------------------

def _sample_contour_points(
    contour: np.ndarray, num_samples: int
) -> np.ndarray:
    """Uniformly sample up to ``num_samples`` points along a contour."""
    contour_points = contour.reshape(-1, 2)
    if len(contour_points) == 0:
        return np.empty((0, 2), dtype=np.int32)
    n = min(len(contour_points), num_samples)
    indices = np.linspace(0, len(contour_points) - 1, n, dtype=int)
    return contour_points[indices]


def _score_candidates(
    candidate_points: np.ndarray,
    edges_normalized: np.ndarray,
    depth_grad_norm: np.ndarray,
    cog_x: int,
    cog_y: int,
    max_dist_scene: float,
    w: int,
    h: int,
    w_edge: float,
    w_depth: float,
    w_cog: float,
) -> list[dict]:
    """Score every contour candidate by the 3-term combined quality metric."""
    out: list[dict] = []
    for px, py in candidate_points:
        if not (0 <= py < h and 0 <= px < w):
            continue
        e_q = float(edges_normalized[py, px])
        d_q = float(depth_grad_norm[py, px])
        dist_cog = float(np.sqrt((px - cog_x) ** 2 + (py - cog_y) ** 2))
        c_q = 1.0 - (dist_cog / max_dist_scene)
        combined = w_edge * e_q + w_depth * d_q + w_cog * c_q
        out.append(
            {
                "point": (int(px), int(py)),
                "combined_quality": combined,
                "edge_quality": e_q,
                "depth_quality": d_q,
                "cog_quality": c_q,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Stage 2 — ray casting
# ---------------------------------------------------------------------------

def _cast_single_ray(
    mask: np.ndarray,
    start_x: float,
    start_y: float,
    dx: float,
    dy: float,
    max_dist: int,
    gap_tolerance: int,
) -> tuple[int, int]:
    """Walk ``(start_x, start_y)`` along ``(dx, dy)`` until leaving the mask.

    Returns the last in-mask integer pixel encountered. ``gap_tolerance``
    consecutive zero-mask pixels are tolerated before giving up (noise
    robustness).
    """
    h, w = mask.shape
    curr_x = float(start_x)
    curr_y = float(start_y)

    current_gap = 0
    last_valid_x, last_valid_y = int(curr_x), int(curr_y)

    for _ in range(max_dist):
        curr_x += dx
        curr_y += dy
        ix = int(round(curr_x))
        iy = int(round(curr_y))

        # Absolute stop: out of image bounds.
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return last_valid_x, last_valid_y

        if mask[iy, ix] == 0:
            current_gap += 1
            if current_gap > gap_tolerance:
                return last_valid_x, last_valid_y
        else:
            current_gap = 0
            last_valid_x, last_valid_y = ix, iy

    return last_valid_x, last_valid_y


def _contour_tangent_80px(
    contour: np.ndarray, gx: int, gy: int
) -> tuple[float, float]:
    """Return a unit perpendicular-to-tangent direction at ``(gx, gy)``.

    The "80 px" name comes from the grid-search-winning configuration: we
    take ~40 contour samples centred on the candidate (with wraparound),
    do an analytic 2x2 eigendecomposition of their xy-coordinate
    covariance, and use the leading eigenvector as the local tangent. The
    returned direction is the tangent rotated 90 degrees, i.e. the
    antipodal grasp axis.

    Pure fallback path (zero-norm covariance, degenerate segment) returns
    ``(1.0, 0.0)`` to match the legacy implementation.
    """
    contour_points = contour.reshape(-1, 2)
    if len(contour_points) == 0:
        return 1.0, 0.0

    # Index of the contour point closest to the candidate.
    distances = np.sqrt(
        (contour_points[:, 0] - gx) ** 2 + (contour_points[:, 1] - gy) ** 2
    )
    closest_idx = int(np.argmin(distances))

    # ~40 points around the candidate (the legacy "80 px" segment).
    num_points = min(40, len(contour_points) // 4)
    half_segment = num_points // 2

    start_idx = (closest_idx - half_segment) % len(contour_points)
    end_idx = (closest_idx + half_segment) % len(contour_points)

    if start_idx < end_idx:
        segment = contour_points[start_idx : end_idx + 1]
    else:
        segment = np.vstack(
            [contour_points[start_idx:], contour_points[: end_idx + 1]]
        )

    if len(segment) > 2:
        x_coords = segment[:, 0]
        y_coords = segment[:, 1]
        x_mean = float(np.mean(x_coords))
        y_mean = float(np.mean(y_coords))
        x_centered = x_coords - x_mean
        y_centered = y_coords - y_mean

        cov_xx = float(np.sum(x_centered * x_centered))
        cov_xy = float(np.sum(x_centered * y_centered))
        cov_yy = float(np.sum(y_centered * y_centered))

        if cov_xx + cov_yy > 1e-6:
            # Analytic 2x2 eigendecomposition: lambda1 = T/2 + sqrt((T/2)^2 - D).
            trace = cov_xx + cov_yy
            det = cov_xx * cov_yy - cov_xy * cov_xy
            lambda1 = trace / 2.0 + np.sqrt(max(0.0, (trace / 2.0) ** 2 - det))

            # Eigenvector for the largest eigenvalue.
            if abs(cov_xy) > 1e-6:
                tx = lambda1 - cov_yy
                ty = cov_xy
            elif abs(cov_xx - lambda1) > 1e-6:
                tx = cov_xy
                ty = lambda1 - cov_xx
            else:
                tx = segment[-1, 0] - segment[0, 0]
                ty = segment[-1, 1] - segment[0, 1]

            t_length = float(np.sqrt(tx * tx + ty * ty))
            if t_length > 1e-6:
                tx /= t_length
                ty /= t_length
                # Perpendicular to the tangent (the antipodal grasp axis).
                return float(-ty), float(tx)
            return 1.0, 0.0

        return 1.0, 0.0

    if len(segment) >= 2:
        tx = float(segment[-1, 0] - segment[0, 0])
        ty = float(segment[-1, 1] - segment[0, 1])
        t_length = float(np.sqrt(tx * tx + ty * ty))
        if t_length > 1e-6:
            tx /= t_length
            ty /= t_length
            return float(-ty), float(tx)

    return 1.0, 0.0


def _depth_gradient_direction(
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    gx: int,
    gy: int,
    w: int,
    h: int,
) -> tuple[float, float]:
    """Perpendicular-to-depth-gradient direction at ``(gx, gy)`` (legacy parity)."""
    if 0 <= gy < grad_y.shape[0] and 0 <= gx < grad_x.shape[1]:
        gx_val = float(grad_x[gy, gx])
        gy_val = float(grad_y[gy, gx])
    else:
        kernel_size = 3
        y_min = max(0, gy - kernel_size)
        y_max = min(h, gy + kernel_size + 1)
        x_min = max(0, gx - kernel_size)
        x_max = min(w, gx + kernel_size + 1)
        gx_val = float(np.mean(grad_x[y_min:y_max, x_min:x_max]))
        gy_val = float(np.mean(grad_y[y_min:y_max, x_min:x_max]))

    length = float(np.sqrt(gx_val * gx_val + gy_val * gy_val))
    if length < 1e-6:
        # Radial fallback from the image centre.
        dx = gx - w // 2
        dy = gy - h // 2
        rad_len = float(np.sqrt(dx * dx + dy * dy))
        if rad_len < 1e-6:
            return 1.0, 0.0
        return dx / rad_len, dy / rad_len
    return -gy_val / length, gx_val / length


def _image_edge_direction(
    mask: np.ndarray, gx: int, gy: int, w: int, h: int
) -> tuple[float, float]:
    """Perpendicular-to-local-mask-edge direction at ``(gx, gy)`` (legacy parity)."""
    kernel_size = 5
    y_min = max(0, gy - kernel_size)
    y_max = min(h, gy + kernel_size + 1)
    x_min = max(0, gx - kernel_size)
    x_max = min(w, gx + kernel_size + 1)

    local_mask = mask[y_min:y_max, x_min:x_max].astype(np.float32)
    if local_mask.size == 0:
        gx_val, gy_val = 1.0, 0.0
    else:
        local_gx = cv2.Sobel(local_mask, cv2.CV_64F, 1, 0, ksize=3)
        local_gy = cv2.Sobel(local_mask, cv2.CV_64F, 0, 1, ksize=3)

        center_y = min(kernel_size, gy - y_min)
        center_x = min(kernel_size, gx - x_min)
        if center_y < local_gy.shape[0] and center_x < local_gx.shape[1]:
            gx_val = float(local_gx[center_y, center_x])
            gy_val = float(local_gy[center_y, center_x])
        else:
            gx_val = float(np.mean(local_gx))
            gy_val = float(np.mean(local_gy))

    length = float(np.sqrt(gx_val * gx_val + gy_val * gy_val))
    if length < 1e-6:
        return 1.0, 0.0
    return -gy_val / length, gx_val / length


def _radial_direction(gx: int, gy: int, w: int, h: int) -> tuple[float, float]:
    """Unit vector pointing radially from the image centre through (gx, gy)."""
    cx = w // 2
    cy = h // 2
    dx = float(gx - cx)
    dy = float(gy - cy)
    length = float(np.sqrt(dx * dx + dy * dy))
    if length < 1e-6:
        return 1.0, 0.0
    return dx / length, dy / length


def _cast_ray_direct_line(
    mask: np.ndarray,
    grasp_point: tuple[int, int],
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    contour: Optional[np.ndarray],
    gradient_source: str,
    max_dist: int,
    gap_tolerance: int,
) -> tuple[int, int, int, int, float]:
    """Bidirectional ray cast from ``grasp_point`` along the grasp axis.

    Direction is selected by ``gradient_source``:

    * ``"Contour Direction (80px avg)"`` — analytic 2x2 PCA on a ~40-point
      contour segment around the candidate (the grid-search winner).
    * ``"Depth Gradients"`` — perpendicular to the local depth gradient.
    * ``"Image Edges"`` — perpendicular to the local mask-edge gradient.
    * ``"Radial from Center"`` — radial line from the image centre.

    Returns ``(end1_x, end1_y, end2_x, end2_y, total_line_length)``.
    """
    h, w = mask.shape
    gx, gy = grasp_point

    if gradient_source == "Contour Direction (80px avg)":
        if contour is not None:
            dx, dy = _contour_tangent_80px(contour, gx, gy)
        else:
            # No contour: fall back to radial-from-centre.
            dx, dy = _radial_direction(gx, gy, w, h)
    elif gradient_source == "Depth Gradients":
        dx, dy = _depth_gradient_direction(grad_x, grad_y, gx, gy, w, h)
    elif gradient_source == "Image Edges":
        dx, dy = _image_edge_direction(mask, gx, gy, w, h)
    elif gradient_source == "Radial from Center":
        dx, dy = _radial_direction(gx, gy, w, h)
    else:
        dx, dy = 1.0, 0.0

    end1_x, end1_y = _cast_single_ray(mask, gx, gy, dx, dy, max_dist, gap_tolerance)
    end2_x, end2_y = _cast_single_ray(mask, gx, gy, -dx, -dy, max_dist, gap_tolerance)
    total_length = float(
        np.sqrt((end1_x - end2_x) ** 2 + (end1_y - end2_y) ** 2)
    )
    return end1_x, end1_y, end2_x, end2_y, total_length


def _cast_ray_through_cog(
    mask: np.ndarray,
    grasp_point: tuple[int, int],
    cog: tuple[int, int],
    max_dist: int,
    gap_tolerance: int,
) -> tuple[int, int, int, int, float]:
    """Bidirectional ray cast from the centre of gravity through ``grasp_point``.

    Returns ``(end1_x, end1_y, end2_x, end2_y, total_line_length)``. A
    grasp point coincident with the CoG is degenerate and returns a
    zero-length grasp.
    """
    gx, gy = grasp_point
    cx, cy = cog
    dx = float(gx - cx)
    dy = float(gy - cy)
    length = float(np.sqrt(dx * dx + dy * dy))
    if length < 1e-6:
        return cx, cy, cx, cy, 0.0

    dx /= length
    dy /= length
    end1_x, end1_y = _cast_single_ray(mask, cx, cy, dx, dy, max_dist, gap_tolerance)
    end2_x, end2_y = _cast_single_ray(mask, cx, cy, -dx, -dy, max_dist, gap_tolerance)
    total_length = float(
        np.sqrt((end1_x - end2_x) ** 2 + (end1_y - end2_y) ** 2)
    )
    return end1_x, end1_y, end2_x, end2_y, total_length
