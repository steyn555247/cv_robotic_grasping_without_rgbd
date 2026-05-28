# Master Results Table

Last updated: 2026-05-28 (after Cornell loader angle-convention fix)

Canonical dataset: **Cornell Grasping Dataset** (885 images, image-wise 5-fold CV, seed=42 splits in `src/data/splits/cornell.json`). All metrics computed via `src/eval/cornell.py` (Jaccard ≥ 0.25 AND |angle error| ≤ 30°, per Jiang 2011).

> **2026-05-28 correction.** All earlier numbers in this file were computed against a buggy `_corners_to_grasp_rect` that read the grasp angle from the rectangle's *shorter* side (gripper opening) instead of the *longer* side (gripper-plate / major axis). This rotated every ground-truth grasp by ~90°, making correct predictions look wrong. The convention sweep (`experiments/EXP-AUDIT/convention_sweep.py`) and visual overlays confirmed the fix. Numbers below are post-fix and verified.

## Cornell image-wise 5-fold (post loader fix)

| Method | Source | Top-1 | Top-5 | IoU mean | Angle err | n | Notes |
|---|---|---|---|---|---|---|---|
| **Heuristic (uncropped, full pipeline)** | EXP-02 | **64.41%** | 67.91% | 0.335 | 19.09° | 885 | DepthAnythingV2-Small → contour → 80px PCA tangent → ray-cast → CoG-boost rank. Training-free, RGB-only. |
| **Heuristic (with original manual crop)** | EXP-02b | **68.59%** | 72.32% | 0.356 | 17.70° | 885 | Same pipeline, image[150:450,100:500] crop then map back. |
| CoG-only baseline (monocular mask) | EXP-01 | 1.92% | 1.92% | 0.184 | 70.80° | 885 | Trivial floor: centroid + PCA major axis + fixed small box. Confirms the heuristic's machinery does ~62 pp of real work. |
| CoG-only baseline (GT-depth mask) | EXP-01 | 1.47% | 1.47% | 0.126 | 63.87° | 885 | As above, mask from Cornell GT depth. |
| GR-ConvNet v1 (reproduced) | EXP-03 (pending) | — | — | — | — | — | Public checkpoint, RGB-D input |
| Xiangli 2025 (reproduced) | EXP-08 (pending) | — | — | — | — | — | Training-free + foundation models (CLIP+SAM+DIFT) |

### Pre-fix numbers (archived — DO NOT cite; buggy loader)

| Method | Top-1 (buggy loader) |
|---|---|
| Heuristic uncropped | 16.72% |
| Heuristic cropped | 18.64% |
| CoG-only mono | 16.95% |

## Reference numbers from the literature

(For context only — not direct comparisons. May use different splits / image counts / metric tolerances.)

| Method | Reported Top-1 (image-wise) | Source |
|---|---|---|
| Jiang et al. SVM-rank | 60.5% | ICRA 2011 |
| Lenz et al. sparse autoencoder | 73.9% | IJRR 2015 / RSS 2013 |
| Redmon & Angelova CNN | ~88% | ICRA 2015 |
| GG-CNN | 78% (depth-only) | RSS 2018 |
| GR-ConvNet v1 | 97.7% | IROS 2020 |
| GR-ConvNet v2 | 98.8% | Sensors 2022 |

Project-internal prior: 75.89% Top-1 on a *manually cropped subset* of Cornell with the heuristic — EXP-02 will tell us whether that holds on full 885 images.

## Paired comparisons (McNemar's test)

Pending — populate once EXP-02 lands and we have heuristic predictions to pair against the CoG-only ones.

## Confidence-interval methodology

95% bootstrap CIs on Top-1 with 1000 resamples — to be added by `results-analyst` once it has at least two methods to compare. Current numbers above are mean ± std across the 5 folds.
