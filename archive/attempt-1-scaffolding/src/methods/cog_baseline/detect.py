"""CoG-only grasp detector (EXP-01).

See ``README.md`` for the rationale. The detector emits a single oriented
grasp rectangle from a foreground mask (or a depth map thresholded into
one). Verified against Cornell GT: ``angle_rad`` (Cornell convention,
direction of the rectangle's short side / gripper opening) lies along the
object's *major* principal axis. The grasp width is a fraction of the
*minor*-axis extent (the narrow object dimension).
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.eval.cornell import GraspRect


def _depth_to_mask(depth: np.ndarray, foreground_is_higher: bool) -> np.ndarray:
    """Threshold a depth map to a binary foreground mask.

    Strategy: subtract a heavily blurred copy of the depth map (the
    smoothed table/background) to suppress the perspective gradient, then
    threshold positive residuals inside a central crop, and keep the
    connected component that maximises ``area / (dist_to_centre + eps)``.
    Returns uint8 {0, 1}.
    """
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover
        cv2 = None  # type: ignore[assignment]

    d32 = depth.astype(np.float32)
    if cv2 is not None:
        bg = cv2.GaussianBlur(d32, (151, 151), 50.0)
    else:  # fallback: subtract the global mean
        bg = np.full_like(d32, float(np.mean(d32)))
    residual = d32 - bg if foreground_is_higher else bg - d32

    h, w = depth.shape
    central = np.zeros_like(d32, dtype=bool)
    central[int(0.15 * h) : int(0.85 * h), int(0.12 * w) : int(0.88 * w)] = True
    valid = depth > 1e-3
    region = central & valid
    if region.sum() < 100:
        return np.zeros_like(depth, dtype=np.uint8)
    thr = 0.5 * float(residual[region].std())
    fg = ((residual > thr) & region).astype(np.uint8)
    if fg.sum() < 50:
        return fg

    if cv2 is None:
        return fg
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n_labels <= 1:
        return fg
    cx_img, cy_img = w / 2.0, h / 2.0
    best_label, best_score = -1, -1.0
    for i in range(1, n_labels):
        x, y, ww, hh, area = stats[i]
        if area < 200:
            continue
        ccx, ccy = x + ww / 2.0, y + hh / 2.0
        dist2 = (ccx - cx_img) ** 2 + (ccy - cy_img) ** 2
        score = float(area) / (dist2 + 100.0)
        if score > best_score:
            best_score = score
            best_label = i
    if best_label > 0:
        return (labels == best_label).astype(np.uint8)
    return fg


def detect_cog_grasp(
    image_rgb: np.ndarray,
    depth_or_mask: np.ndarray,
    is_mask: bool = False,
    foreground_is_higher: bool = False,
    width_fraction: float = 0.6,
    height_px: float = 20.0,
    top_k: int = 5,
) -> List[GraspRect]:
    """Return ``top_k`` copies of the CoG + PCA-major-axis grasp rectangle."""
    del image_rgb  # method ignores all RGB content
    if is_mask:
        mask = (np.asarray(depth_or_mask) > 0).astype(np.uint8)
    else:
        mask = _depth_to_mask(np.asarray(depth_or_mask), foreground_is_higher)

    ys, xs = np.nonzero(mask)
    if xs.size < 10:
        return []

    cx = float(xs.mean())
    cy = float(ys.mean())
    pts = np.stack(
        [xs.astype(np.float64) - cx, ys.astype(np.float64) - cy], axis=1
    )
    cov = (pts.T @ pts) / float(pts.shape[0])
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues

    minor_vec = eigvecs[:, 0]  # narrow direction (gripper plate aligned with this)
    major_vec = eigvecs[:, 1]  # elongated direction (gripper opening along this)

    # Grasp width (gripper opening) = fraction of the object's minor extent.
    proj_minor = pts @ minor_vec
    minor_extent = float(np.percentile(proj_minor, 95) - np.percentile(proj_minor, 5))
    if not np.isfinite(minor_extent) or minor_extent <= 0:
        minor_extent = float(np.sqrt(max(eigvals[0], 1e-6)) * 4.0)

    # Cornell convention: angle_rad = direction of short side = direction
    # along which the gripper opens, which is the object's MAJOR axis.
    angle = float(np.arctan2(major_vec[1], major_vec[0]))
    while angle > np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi

    width = max(float(width_fraction * minor_extent), 5.0)
    rect = GraspRect(
        x=cx, y=cy, angle_rad=angle, width=width, height=float(height_px)
    )
    return [rect for _ in range(top_k)]
