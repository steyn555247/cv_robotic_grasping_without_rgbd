"""Canonical Cornell Grasping Dataset evaluator.

This module is the single source of truth for Cornell metrics in the project.
Every experiment (heuristic, GR-ConvNet repro, ablations, etc.) imports
``evaluate_predictions`` from here. No other code computes Cornell numbers;
do not duplicate this logic anywhere else in the repo.

The Cornell correctness criterion (Jiang et al., ICRA 2011) is:

1. Oriented-rectangle Jaccard (IoU) between predicted and ground-truth
   grasp rectangles is at least ``IOU_THRESHOLD`` (0.25).
2. Absolute angle error is at most ``ANGLE_THRESHOLD_DEG`` (30 degrees),
   computed with parallel-jaw symmetry (a 0-degree GT matches a 180-degree
   prediction).

Polygon IoU uses ``shapely >= 2.0`` (verified library, not hand-rolled).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

# shapely >= 2.0 is required for oriented polygon IoU.
from shapely.geometry import Polygon  # type: ignore[import-not-found]


# Cornell correctness thresholds (Jiang et al., ICRA 2011, "Efficient
# Grasping from RGBD Images: Learning using a new Rectangle Representation").
# Do not change these constants; they define the canonical metric.
IOU_THRESHOLD: float = 0.25
ANGLE_THRESHOLD_DEG: float = 30.0


@dataclass
class GraspRect:
    """Oriented grasp rectangle (Cornell convention).

    Geometry matches ``GraspCandidate.get_corners`` in
    ``Heuristics approach/grasp_detection_contour_80px.py``: the rectangle is
    centered at (``x``, ``y``) with width along the gripper-opening axis
    (rotated by ``angle_rad``) and height along the gripper-plate axis.
    """

    x: float
    y: float
    angle_rad: float
    width: float
    height: float = 20.0

    def get_corners(self) -> np.ndarray:
        """Return the four oriented-rectangle corners as a (4, 2) array."""
        cos_a = float(np.cos(self.angle_rad))
        sin_a = float(np.sin(self.angle_rad))
        w, h = self.width, self.height

        corners_local = np.array(
            [
                (-w / 2.0, -h / 2.0),
                (w / 2.0, -h / 2.0),
                (w / 2.0, h / 2.0),
                (-w / 2.0, h / 2.0),
            ],
            dtype=np.float64,
        )
        # 2D rotation matrix applied to each local corner.
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
        rotated = corners_local @ rotation.T
        rotated[:, 0] += self.x
        rotated[:, 1] += self.y
        return rotated


def _oriented_iou(rect_a: GraspRect, rect_b: GraspRect) -> float:
    """Return the Jaccard index (intersection / union) of two oriented rects."""
    poly_a = Polygon(rect_a.get_corners())
    poly_b = Polygon(rect_b.get_corners())
    if not poly_a.is_valid or not poly_b.is_valid:
        # Degenerate polygons (zero area) cannot overlap meaningfully.
        return 0.0
    intersection = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def _angle_error_deg(a_rad: float, b_rad: float) -> float:
    """Return the parallel-jaw angle error in degrees (0 to 90 inclusive)."""
    a_deg = float(np.degrees(a_rad))
    b_deg = float(np.degrees(b_rad))
    diff = abs(a_deg - b_deg) % 180.0
    return float(min(diff, 180.0 - diff))


def _is_match(
    pred: GraspRect,
    gt: GraspRect,
    iou_threshold: float,
    angle_threshold_deg: float,
) -> tuple[bool, float, float]:
    """Return (matched, iou, angle_error_deg) for one (pred, gt) pair."""
    angle_err = _angle_error_deg(pred.angle_rad, gt.angle_rad)
    if angle_err > angle_threshold_deg:
        # Skip the IoU computation when the angle is already disqualifying.
        # We still return the IoU value (cheap and useful for debugging).
        iou = _oriented_iou(pred, gt)
        return False, iou, angle_err
    iou = _oriented_iou(pred, gt)
    matched = iou >= iou_threshold
    return matched, iou, angle_err


def _best_match_against_gt(
    pred: GraspRect,
    ground_truth: List[GraspRect],
    iou_threshold: float,
    angle_threshold_deg: float,
) -> tuple[bool, float, float]:
    """Return (any_match, best_iou, best_angle_err) for one prediction vs all GT.

    A prediction is correct if ANY ground-truth grasp passes both thresholds.
    ``best_iou`` is the maximum IoU achieved over all GT rects (used for
    ``iou_mean`` reporting). ``best_angle_err`` is the angle error at the GT
    that yielded ``best_iou`` (so the reported angle error pairs sensibly with
    the reported IoU).
    """
    if not ground_truth:
        return False, 0.0, 0.0
    any_match = False
    best_iou = -1.0
    best_angle_err = 0.0
    for gt in ground_truth:
        matched, iou, angle_err = _is_match(
            pred, gt, iou_threshold, angle_threshold_deg
        )
        any_match = any_match or matched
        if iou > best_iou:
            best_iou = iou
            best_angle_err = angle_err
    if best_iou < 0.0:
        best_iou = 0.0
    return any_match, float(best_iou), float(best_angle_err)


def evaluate_predictions(
    predictions: List[List[GraspRect]],
    ground_truth: List[List[GraspRect]],
    iou_threshold: float = IOU_THRESHOLD,
    angle_threshold_deg: float = ANGLE_THRESHOLD_DEG,
    top_k: int = 5,
) -> dict:
    """Canonical Cornell evaluation. Returns the standard metrics dict.

    Args:
        predictions: per sample, ranked list of predicted grasps (best first).
        ground_truth: per sample, list of ground-truth grasps (any order).
        iou_threshold: minimum oriented-rectangle Jaccard for a correct grasp.
        angle_threshold_deg: maximum absolute angle error (with 180-deg
            symmetry) for a correct grasp.
        top_k: how many top predictions count for ``topK`` accuracy.

    Returns:
        Dict with keys:
            ``top1``: fraction of samples whose top-1 prediction is correct.
            ``top5``: fraction of samples where ANY of the top-k predictions
                is correct (named ``top5`` for the conventional k=5).
            ``iou_mean``: mean IoU of the top-1 prediction vs its best-matching
                GT, averaged over ALL samples (not just correct ones, to avoid
                cherry-picking). Samples with no predictions contribute 0.
            ``angle_error_deg_mean``: mean angle error of the top-1 prediction
                vs its best-IoU GT, averaged over all samples.
            ``n_correct_top1``: integer count of correct top-1 predictions.
            ``n_samples``: number of samples evaluated.
            ``per_sample_correct``: list[bool] of top-1 correctness, one entry
                per sample. Required by results-analyst for McNemar tests.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"predictions and ground_truth length mismatch: "
            f"{len(predictions)} vs {len(ground_truth)}"
        )

    n_samples = len(predictions)
    per_sample_correct: list[bool] = []
    per_sample_topk_correct: list[bool] = []
    per_sample_iou: list[float] = []
    per_sample_angle_err: list[float] = []

    for sample_preds, sample_gt in zip(predictions, ground_truth):
        # Top-1 metrics.
        if sample_preds:
            top1_pred = sample_preds[0]
            matched, best_iou, best_angle_err = _best_match_against_gt(
                top1_pred, sample_gt, iou_threshold, angle_threshold_deg
            )
        else:
            matched, best_iou, best_angle_err = False, 0.0, 0.0
        per_sample_correct.append(bool(matched))
        per_sample_iou.append(float(best_iou))
        per_sample_angle_err.append(float(best_angle_err))

        # Top-k metric: ANY of the first ``top_k`` predictions must match.
        topk_ok = False
        for pred in sample_preds[:top_k]:
            ok, _, _ = _best_match_against_gt(
                pred, sample_gt, iou_threshold, angle_threshold_deg
            )
            if ok:
                topk_ok = True
                break
        per_sample_topk_correct.append(topk_ok)

    n_correct_top1 = int(sum(per_sample_correct))
    top1 = float(n_correct_top1 / n_samples) if n_samples > 0 else 0.0
    top5 = (
        float(sum(per_sample_topk_correct) / n_samples) if n_samples > 0 else 0.0
    )
    iou_mean = float(np.mean(per_sample_iou)) if per_sample_iou else 0.0
    angle_error_deg_mean = (
        float(np.mean(per_sample_angle_err)) if per_sample_angle_err else 0.0
    )

    return {
        "top1": top1,
        "top5": top5,
        "iou_mean": iou_mean,
        "angle_error_deg_mean": angle_error_deg_mean,
        "n_correct_top1": n_correct_top1,
        "n_samples": n_samples,
        "per_sample_correct": per_sample_correct,
    }
