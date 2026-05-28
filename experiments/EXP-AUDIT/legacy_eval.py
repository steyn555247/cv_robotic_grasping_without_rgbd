"""Faithful reproduction of the legacy ``GraspEvaluator`` from the prior
course-project notebook (``cornell comparisson/Grasp_Detection_AppWorking_
Method_With_Cropping.ipynb``, cell 24) plus the legacy GT loader.

Everything here intentionally mirrors the legacy code's quirks:

* IoU is computed by rasterising both polygons into a HARDCODED 480x640 mask
  and counting bitwise intersection / union (cv2.fillPoly + np.sum).
* GT rectangles come from ``corners[1] - corners[0]`` (jaw-plate side) for
  ``width`` and ``angle``, NOT from the Lenz-style shorter-side convention.
* Angle difference is computed via the buggy formula

      diff = abs(a1 - a2)
      diff = min(diff, np.pi - diff, abs(diff - np.pi))

  which returns NEGATIVE degrees when ``|a1-a2|`` exceeds ``pi``, causing the
  ``<= 30 deg`` threshold to trivially pass for wraparound cases.
* Predictions are converted to corners using ``height=20`` (the legacy
  ``Candidate.h`` default) regardless of the canonical detector's actual
  ``height`` field. This matches the legacy ``Candidate(... w=length, h=20.0)``
  output shape exactly.

NB: This file ONLY exists to forensically reproduce the legacy 75.89% number.
Do not import any of this from production code. ``src/eval/cornell.py`` is the
single source of truth for canonical Cornell metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Legacy data structures (mirror the notebook's GraspCandidate and the GT dict)
# ---------------------------------------------------------------------------

@dataclass
class LegacyCandidate:
    """Mirror of ``Candidate`` from notebook cell 21.

    The legacy detector emitted ``Candidate(x, y, angle, w=length, h=20.0)``.
    ``get_corners`` reproduces that class's geometry exactly.
    """
    x: float
    y: float
    angle: float  # radians
    width: float
    height: float = 20.0

    def get_corners(self) -> np.ndarray:
        cos_a = float(np.cos(self.angle))
        sin_a = float(np.sin(self.angle))
        w, h = float(self.width), float(self.height)
        pts = [
            (-w / 2.0, -h / 2.0),
            (w / 2.0, -h / 2.0),
            (w / 2.0, h / 2.0),
            (-w / 2.0, h / 2.0),
        ]
        corners = []
        for dx, dy in pts:
            px = self.x + dx * cos_a - dy * sin_a
            py = self.y + dx * sin_a + dy * cos_a
            corners.append([px, py])
        return np.array(corners)


# ---------------------------------------------------------------------------
# Legacy IoU + angle_diff (verbatim from notebook cell 24)
# ---------------------------------------------------------------------------

def legacy_polygon_iou(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """Verbatim ``GraspEvaluator.polygon_iou`` from cell 24."""
    h, w = 480, 640
    mask1 = np.zeros((h, w), dtype=np.uint8)
    mask2 = np.zeros((h, w), dtype=np.uint8)
    pts1 = np.clip(poly1.astype(np.int32), 0, [w - 1, h - 1])
    pts2 = np.clip(poly2.astype(np.int32), 0, [w - 1, h - 1])
    cv2.fillPoly(mask1, [pts1], 1)
    cv2.fillPoly(mask2, [pts2], 1)
    intersection = int(np.sum(mask1 & mask2))
    union = int(np.sum(mask1 | mask2))
    return intersection / max(union, 1e-8)


def legacy_angle_diff(angle1: float, angle2: float) -> float:
    """Verbatim ``GraspEvaluator.angle_diff`` from cell 24.

    BUG: when ``|angle1 - angle2| > pi`` (which happens whenever the two
    angles straddle the ``+pi/-pi`` branch cut), ``np.pi - diff`` becomes
    negative, and ``min(diff, np.pi - diff, abs(diff - np.pi))`` returns the
    negative value. That negative is then passed through ``np.degrees``,
    yielding a negative degree count, which trivially passes the legacy
    ``best_angle_diff <= 30`` threshold.
    """
    diff = abs(angle1 - angle2)
    diff = min(diff, np.pi - diff, abs(diff - np.pi))
    return float(np.degrees(diff))


# ---------------------------------------------------------------------------
# Legacy evaluate_grasp / evaluate_image
# ---------------------------------------------------------------------------

def _evaluate_single_grasp(
    prediction: LegacyCandidate,
    ground_truths: List[dict],
    iou_threshold: float,
    angle_threshold: float,
) -> dict:
    """Verbatim ``GraspEvaluator.evaluate_grasp`` from cell 24."""
    pred_corners = prediction.get_corners()
    best_iou = 0.0
    best_angle_diff = 180.0
    for gt in ground_truths:
        iou = legacy_polygon_iou(pred_corners, gt["corners"])
        ang = legacy_angle_diff(prediction.angle, gt["angle"])
        if iou > best_iou:
            best_iou = iou
            best_angle_diff = ang
    success = bool(
        best_iou >= iou_threshold and best_angle_diff <= angle_threshold
    )
    return {
        "success": success,
        "best_iou": best_iou,
        "best_angle_diff": best_angle_diff,
    }


def legacy_evaluate_image(
    predictions: List[LegacyCandidate],
    ground_truths: List[dict],
    iou_threshold: float = 0.25,
    angle_threshold: float = 30.0,
) -> dict:
    """Verbatim ``GraspEvaluator.evaluate_image`` from cell 24."""
    if not predictions:
        return {"top1_success": False, "top5_success": False, "avg_iou": 0.0}
    results = [
        _evaluate_single_grasp(pred, ground_truths, iou_threshold, angle_threshold)
        for pred in predictions
    ]
    return {
        "top1_success": results[0]["success"] if results else False,
        "top5_success": any(r["success"] for r in results[:5]),
        "any_success": any(r["success"] for r in results),
        "avg_iou": float(np.mean([r["best_iou"] for r in results])),
        "avg_angle_diff": float(np.mean([r["best_angle_diff"] for r in results])),
    }


# ---------------------------------------------------------------------------
# Legacy GT loader (mirror of ``CornellGraspDataset._load_grasp_rectangles``)
# ---------------------------------------------------------------------------

def legacy_load_gt_from_cpos(cpos_path: Path) -> List[dict]:
    """Return GT grasps in the legacy dict-of-fields format.

    Mirrors ``CornellGraspDataset._load_grasp_rectangles`` exactly:
    ``angle = atan2(c1 - c0)``, ``width = |c1 - c0|``, ``height = |c2 - c1|``.
    """
    grasps: List[dict] = []
    with open(cpos_path) as fh:
        lines = fh.readlines()
    for i in range(0, len(lines) - 3, 4):
        corners = []
        valid = True
        for j in range(4):
            parts = lines[i + j].strip().split()
            if len(parts) < 2:
                valid = False
                break
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                valid = False
                break
            if np.isnan(x) or np.isnan(y):
                valid = False
                break
            corners.append([x, y])
        if not (valid and len(corners) == 4):
            continue
        corners_np = np.array(corners, dtype=np.float64)
        center = corners_np.mean(axis=0)
        dx = corners_np[1, 0] - corners_np[0, 0]
        dy = corners_np[1, 1] - corners_np[0, 1]
        angle = float(np.arctan2(dy, dx))
        width = float(np.linalg.norm(corners_np[1] - corners_np[0]))
        height = float(np.linalg.norm(corners_np[2] - corners_np[1]))
        grasps.append(
            {
                "corners": corners_np,
                "center": center,
                "angle": angle,
                "angle_deg": float(np.degrees(angle)),
                "width": width,
                "height": height,
            }
        )
    return grasps


# ---------------------------------------------------------------------------
# Convenience aggregator: feed many (preds, gts) and return summary metrics
# ---------------------------------------------------------------------------

def legacy_evaluate_dataset(
    pred_lists: Iterable[List[LegacyCandidate]],
    gt_lists: Iterable[List[dict]],
    iou_threshold: float = 0.25,
    angle_threshold: float = 30.0,
) -> dict:
    """Aggregate over a dataset, mirroring how the grid search aggregates."""
    top1 = 0
    top5 = 0
    any_ok = 0
    ious: List[float] = []
    angs: List[float] = []
    total = 0
    for preds, gts in zip(pred_lists, gt_lists):
        if not preds or not gts:
            continue
        res = legacy_evaluate_image(preds, gts, iou_threshold, angle_threshold)
        total += 1
        if res.get("top1_success"):
            top1 += 1
        if res.get("top5_success"):
            top5 += 1
        if res.get("any_success"):
            any_ok += 1
        ious.append(res.get("avg_iou", 0.0))
        angs.append(res.get("avg_angle_diff", 0.0))
    return {
        "total": total,
        "top1": top1,
        "top5": top5,
        "any": any_ok,
        "top1_acc": (top1 / total * 100.0) if total else 0.0,
        "top5_acc": (top5 / total * 100.0) if total else 0.0,
        "any_acc": (any_ok / total * 100.0) if total else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_angle": float(np.mean(angs)) if angs else 0.0,
    }
