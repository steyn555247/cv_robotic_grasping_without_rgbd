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
