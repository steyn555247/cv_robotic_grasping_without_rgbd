"""Property-based tests for the canonical Cornell evaluator.

These tests live next to the evaluator (per project rule) and lock down the
geometric invariants of the Cornell metric. If any of them fails, every
downstream experiment number is suspect.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.eval.cornell import (
    ANGLE_THRESHOLD_DEG,
    IOU_THRESHOLD,
    GraspRect,
    _angle_error_deg,
    _oriented_iou,
    evaluate_predictions,
)


# ---------------------------------------------------------------------------
# Oriented-IoU geometry
# ---------------------------------------------------------------------------


def test_identity_iou() -> None:
    """Same rectangle -> IoU = 1.0."""
    rect = GraspRect(x=100.0, y=80.0, angle_rad=0.3, width=60.0, height=20.0)
    assert _oriented_iou(rect, rect) == pytest.approx(1.0, abs=1e-9)


def test_perpendicular_iou() -> None:
    """Two rectangles sharing a center but rotated 90 degrees overlap weakly."""
    base = GraspRect(x=0.0, y=0.0, angle_rad=0.0, width=80.0, height=20.0)
    perp = GraspRect(
        x=0.0, y=0.0, angle_rad=math.pi / 2.0, width=80.0, height=20.0
    )
    iou = _oriented_iou(base, perp)
    # Intersection = 20x20 square = 400. Union = 2*1600 - 400 = 2800.
    # Therefore IoU = 400/2800 ~= 0.1429, which is well below 0.5.
    assert iou < 0.5
    assert iou == pytest.approx(400.0 / 2800.0, abs=1e-6)


def test_180_symmetry_iou_and_angle() -> None:
    """Parallel-jaw grippers are symmetric -> 180 rotation matches the original."""
    base = GraspRect(x=50.0, y=50.0, angle_rad=0.0, width=60.0, height=20.0)
    flipped = GraspRect(
        x=50.0, y=50.0, angle_rad=math.pi, width=60.0, height=20.0
    )
    # Same physical rectangle -> IoU must be 1.0.
    assert _oriented_iou(base, flipped) == pytest.approx(1.0, abs=1e-9)
    # And angle error must be exactly 0 (mod 180).
    assert _angle_error_deg(0.0, math.pi) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Angle error wraparound
# ---------------------------------------------------------------------------


def test_angle_error_wraparound() -> None:
    """0 vs pi -> 0; 0 vs pi/2 -> 90; 30 vs 210 -> 0."""
    assert _angle_error_deg(0.0, math.pi) == pytest.approx(0.0, abs=1e-9)
    assert _angle_error_deg(0.0, math.pi / 2.0) == pytest.approx(90.0, abs=1e-9)
    assert _angle_error_deg(
        math.radians(30.0), math.radians(210.0)
    ) == pytest.approx(0.0, abs=1e-9)
    # 10 vs 170 -> min(160, 20) = 20.
    assert _angle_error_deg(
        math.radians(10.0), math.radians(170.0)
    ) == pytest.approx(20.0, abs=1e-9)


# ---------------------------------------------------------------------------
# evaluate_predictions
# ---------------------------------------------------------------------------


def _make_gt_set() -> list[list[GraspRect]]:
    return [
        [GraspRect(10.0, 10.0, 0.0, 40.0)],
        [GraspRect(50.0, 50.0, math.pi / 4.0, 60.0)],
        [GraspRect(-30.0, 20.0, 1.2, 50.0)],
    ]


def test_evaluate_perfect_predictions() -> None:
    """Predictions identical to GT -> top1 = top5 = 1.0, all per-sample True."""
    gt = _make_gt_set()
    preds = [[gt_list[0]] for gt_list in gt]
    metrics = evaluate_predictions(preds, gt)
    assert metrics["top1"] == pytest.approx(1.0)
    assert metrics["top5"] == pytest.approx(1.0)
    assert metrics["n_correct_top1"] == 3
    assert metrics["n_samples"] == 3
    assert metrics["iou_mean"] == pytest.approx(1.0, abs=1e-9)
    assert metrics["angle_error_deg_mean"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["per_sample_correct"] == [True, True, True]


def test_evaluate_empty_predictions() -> None:
    """No predictions for any sample -> top1 = 0.0 and all per-sample False."""
    gt = _make_gt_set()
    preds: list[list[GraspRect]] = [[] for _ in gt]
    metrics = evaluate_predictions(preds, gt)
    assert metrics["top1"] == pytest.approx(0.0)
    assert metrics["top5"] == pytest.approx(0.0)
    assert metrics["n_correct_top1"] == 0
    assert metrics["per_sample_correct"] == [False, False, False]


def test_top5_includes_top1() -> None:
    """If top-1 is correct, top-5 must also be correct (monotonicity)."""
    gt = _make_gt_set()
    # Top-1 is the perfect match; fill the rest with garbage predictions.
    garbage = GraspRect(1e4, 1e4, 0.0, 5.0)
    preds = [[gt_list[0], garbage, garbage] for gt_list in gt]
    metrics = evaluate_predictions(preds, gt)
    assert metrics["top1"] == pytest.approx(1.0)
    assert metrics["top5"] == pytest.approx(1.0)


def test_per_sample_correct_length() -> None:
    """per_sample_correct must have one entry per sample (for McNemar)."""
    gt = _make_gt_set()
    preds = [[gt_list[0]] for gt_list in gt]
    metrics = evaluate_predictions(preds, gt)
    assert len(metrics["per_sample_correct"]) == len(preds)
    assert all(isinstance(v, bool) for v in metrics["per_sample_correct"])


def test_constants_match_jiang_2011() -> None:
    """Sanity-check the canonical thresholds haven't drifted."""
    assert IOU_THRESHOLD == 0.25
    assert ANGLE_THRESHOLD_DEG == 30.0


def test_angle_just_above_threshold_fails() -> None:
    """A prediction 31 degrees off must be marked incorrect even with full overlap."""
    gt = [[GraspRect(0.0, 0.0, 0.0, 60.0, 20.0)]]
    pred = [[GraspRect(0.0, 0.0, math.radians(31.0), 60.0, 20.0)]]
    metrics = evaluate_predictions(pred, gt)
    assert metrics["top1"] == pytest.approx(0.0)
    assert metrics["per_sample_correct"] == [False]


def test_iou_below_threshold_fails() -> None:
    """Perfect angle but disjoint rectangles must be marked incorrect."""
    gt = [[GraspRect(0.0, 0.0, 0.0, 40.0, 20.0)]]
    pred = [[GraspRect(500.0, 500.0, 0.0, 40.0, 20.0)]]
    metrics = evaluate_predictions(pred, gt)
    assert metrics["top1"] == pytest.approx(0.0)
    assert metrics["iou_mean"] == pytest.approx(0.0, abs=1e-9)


def test_length_mismatch_raises() -> None:
    """Mismatched lengths must raise instead of silently producing junk."""
    with pytest.raises(ValueError):
        evaluate_predictions([[]], [[], []])


def test_get_corners_shape_and_centering() -> None:
    """get_corners returns a 4x2 array centered at (x, y) for any angle."""
    rect = GraspRect(x=7.0, y=-3.0, angle_rad=0.9, width=40.0, height=10.0)
    corners = rect.get_corners()
    assert corners.shape == (4, 2)
    center = corners.mean(axis=0)
    assert center[0] == pytest.approx(7.0, abs=1e-9)
    assert center[1] == pytest.approx(-3.0, abs=1e-9)
    # Side lengths must equal the requested width and height.
    side_w = np.linalg.norm(corners[1] - corners[0])
    side_h = np.linalg.norm(corners[2] - corners[1])
    assert side_w == pytest.approx(40.0, abs=1e-9)
    assert side_h == pytest.approx(10.0, abs=1e-9)
