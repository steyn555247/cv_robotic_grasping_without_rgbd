"""Smoke test for the heuristic grasp detector.

End-to-end test: load Cornell sample 0, run Depth-Anything-V2-Small, run
``detect_grasp``, and sanity-check the result. Marked ``slow`` because the
depth model is downloaded (~100 MB) on first invocation.

Run with::

    pytest src/methods/heuristic/test_smoke.py -v --no-header

Deselect with::

    pytest -m "not slow"
"""

from __future__ import annotations

import math
import random
import time

import numpy as np
import pytest
import torch

from src.data.cornell_loader import CornellDataset
from src.methods.heuristic.config import HeuristicConfig
from src.methods.heuristic.depth import DepthEstimator
from src.methods.heuristic.detect import detect_grasp


@pytest.mark.slow
def test_detect_grasp_on_cornell_sample_0() -> None:
    """End-to-end: Cornell sample 0 -> depth -> detect_grasp -> sanity."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    ds = CornellDataset(split="image-wise", fold=0, partition="test")
    sample = ds[0]

    image = sample["image"]
    h, w = image.shape[:2]
    assert image.dtype == np.uint8
    assert image.ndim == 3 and image.shape[2] == 3

    estimator = DepthEstimator()
    config = HeuristicConfig()

    t0 = time.perf_counter()
    depth_pred = estimator(image)
    grasps = detect_grasp(image, depth_pred, config)
    elapsed = time.perf_counter() - t0

    # Sanity: depth map is normalised to [0, 1] and matches image size.
    assert depth_pred.shape == (h, w)
    assert depth_pred.dtype == np.float32
    assert float(depth_pred.min()) >= 0.0
    assert float(depth_pred.max()) <= 1.0 + 1e-6

    # Output count: 1..5 grasps.
    assert 1 <= len(grasps) <= 5, f"expected 1..5 grasps, got {len(grasps)}"

    # Each grasp: finite numbers, in-bounds centre.
    for g in grasps:
        assert math.isfinite(g.x), f"x not finite: {g.x}"
        assert math.isfinite(g.y), f"y not finite: {g.y}"
        assert math.isfinite(g.angle_rad), f"angle not finite: {g.angle_rad}"
        assert math.isfinite(g.width), f"width not finite: {g.width}"
        assert 0.0 <= g.x <= w, f"x={g.x} out of bounds [0, {w}]"
        assert 0.0 <= g.y <= h, f"y={g.y} out of bounds [0, {h}]"
        assert g.width > 0.0, f"width must be > 0, got {g.width}"

    # Runtime budget: < 60 s on CPU for one depth + detect call.
    assert elapsed < 60.0, f"runtime {elapsed:.1f}s exceeded 60s budget"


@pytest.mark.slow
def test_detect_grasp_output_convention_for_elongated_object() -> None:
    """Cornell GraspRect convention: for an elongated object the heuristic
    must emit ``height > width`` (jaw-plate length > gripper opening) and the
    angle must be wrapped to ``[-pi/2, pi/2]``.

    This guards the 2026-05-27 output-convention fix in ``detect.py`` —
    without it, ``height`` was hardcoded to 20 and ``angle_rad`` was an
    unwrapped ``atan2`` output, which broke alignment with
    ``src.data.cornell_loader._corners_to_grasp_rect``.

    We use pcd0100 (an elongated object in Cornell — long-axis-aligned
    GT rectangles with aspect ratio ~ 2 :: see
    ``experiments/EXP-02_full_cornell_eval/debug_predictions.py`` output).
    """
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    ds = CornellDataset(split="all")
    # Find pcd0100 in the dataset.
    id_to_idx = {f"pcd{pid:04d}": i for i, pid in enumerate(ds.ids)}
    sample = ds[id_to_idx["pcd0100"]]
    image = sample["image"]

    estimator = DepthEstimator()
    depth_pred = estimator(image)

    config = HeuristicConfig()
    grasps = detect_grasp(image, depth_pred, config)
    assert grasps, "expected at least one grasp on pcd0100"

    # GT for pcd0100 has aspect (h/w) ~ 2: object is elongated, so the
    # heuristic's perpendicular-extent estimate should put `height` > `width`.
    top = grasps[0]
    assert top.height > top.width, (
        f"elongated object expected height > width, got "
        f"width={top.width:.1f} height={top.height:.1f}"
    )

    # Angle must be wrapped into [-pi/2, pi/2] for every returned grasp.
    for i, g in enumerate(grasps):
        assert -math.pi / 2 - 1e-6 <= g.angle_rad <= math.pi / 2 + 1e-6, (
            f"grasp[{i}].angle_rad={g.angle_rad} is outside [-pi/2, pi/2]"
        )
