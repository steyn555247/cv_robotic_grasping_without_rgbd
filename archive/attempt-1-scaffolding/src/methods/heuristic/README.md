# Heuristic grasp detector

Training-free monocular-RGB grasp detector. This is the production refactor
of the original course-project Streamlit prototype
(`Heuristics approach/grasp_detection_contour_80px.py`, frozen archive).
The math is preserved bit-for-bit; only the UI, plotting, history, and
debug instrumentation were removed.

## Files

| File | Purpose |
| --- | --- |
| `config.py` | `HeuristicConfig` dataclass. All hyperparameters live here; defaults are the grid-search winners. |
| `depth.py` | `DepthEstimator` — frozen Depth-Anything-V2-Small wrapper. Normalises output to `[0, 1]`. |
| `detect.py` | `detect_grasp(image_rgb, depth, config) -> list[GraspRect]`. The pipeline. |
| `test_smoke.py` | End-to-end smoke test on Cornell sample 0 (marker: `slow`). |

## Pipeline

```
RGB + depth ──► depth-percentile threshold + morphology  (Stage 0)
            ──► largest contour by area
            ──► uniformly-sampled contour points         (Stage 1)
            ──► 3-term candidate score:
                  w_edge * Canny + w_depth * |∇depth| + w_cog * (1 − d/diag)
            ──► top-K = num_output_grasps * candidate_multiplier
            ──► per-candidate ray cast                   (Stage 2)
                  direction = analytic 2x2 PCA of ~40 contour points (80 px)
                  rotate 90° → antipodal grasp axis
                  walk bidirectionally until leaving the mask
            ──► length filter [min_grasp_length, max_grasp_length]
            ──► rank: line_length − cog_boost * proximity * 500   (Stage 3)
            ──► top-N as GraspRect(x, y, angle_rad, width, height=20)
```

The 80-pixel PCA-tangent block in Stage 2 is the classical contour-tangent
grasp-planning approach from
- Sanz (1999), *Vision-Guided Grasping of Unknown Objects for Service Robots*;
- Morales (2001), *Heuristic Vision-Based Computation of Planar Antipodal Grasps*;
- Lei (2017), *Fast Grasping of Unknown Objects Based on Principal Component Analysis*.

For a small contour patch around a candidate point, the first principal
component of the patch xy-coordinates approximates the local tangent. The
perpendicular to that tangent is the antipodal grasp axis along which the
fingers close.

## Grid-search winners

`config.HeuristicConfig` defaults to the configuration the course-project
sweep selected (see `cornell comparisson/GRID_SEARCH_README.md` in the
archive):

| Parameter | Default | Notes |
| --- | --- | --- |
| `w_edge`, `w_depth`, `w_cog` | 0.001, 0.001, 0.998 | CoG dominates by ~3 orders of magnitude. |
| `depth_percentile` | 30 | Top 70% of depth values form the ROI. |
| `ray_algorithm` | `"Direct Line with CoG Boost"` | Winning ray method. |
| `gradient_source` | `"Contour Direction (80px avg)"` | Winning direction method. |
| `cog_boost` | 3.75 | Stage-3 proximity bonus. |
| `candidate_multiplier` | 100 | Top-K = 5 * 100 = 500 candidates after stage 1. |
| `num_output_grasps` | 5 | Returned to the caller. |

## Usage

```python
import numpy as np
from src.methods.heuristic.config import HeuristicConfig
from src.methods.heuristic.depth import DepthEstimator
from src.methods.heuristic.detect import detect_grasp

image_rgb: np.ndarray  # HxWx3 uint8
estimator = DepthEstimator()        # downloads Depth-Anything-V2-Small once
depth = estimator(image_rgb)        # HxW float32 in [0, 1]
grasps = detect_grasp(image_rgb, depth, HeuristicConfig())
```

## Swapping the depth model

The default depth backbone is Depth-Anything-V2-Small. To switch sizes:

```python
estimator = DepthEstimator(model_name="depth-anything/Depth-Anything-V2-Base-hf")
# or:
estimator = DepthEstimator(model_name="depth-anything/Depth-Anything-V2-Large-hf")
```

MiDaS variants from the original Streamlit prototype are intentionally not
ported; the research pipeline standardised on Depth-Anything-V2. Re-add
them in `depth.py` if needed.

## Historical source

The original Streamlit notebook lives at
`Heuristics approach/grasp_detection_contour_80px.py`. That folder is
frozen — do not modify it. All future detector changes happen in this
folder (`src/methods/heuristic/`).
