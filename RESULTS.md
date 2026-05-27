# Master Results Table

Last updated: 2026-05-27 (EXP-02 re-ran after output-convention fix)

Canonical dataset: **Cornell Grasping Dataset** (885 images, image-wise 5-fold CV, seed=42 splits in `src/data/splits/cornell.json`). All metrics computed via `src/eval/cornell.py` (Jaccard ≥ 0.25 AND |angle error| ≤ 30°, per Jiang 2011).

## Cornell image-wise 5-fold

| Method | Source | Top-1 (mean ± std) | Top-5 | IoU mean | Angle err | n | Notes |
|---|---|---|---|---|---|---|---|
| **CoG-only, GT depth mask** | EXP-01 | 8.70% ± 1.98 | 8.70% | 0.112 | 27.07° | 885 | Trivial baseline: centroid + major axis + 0.6×minor extent |
| **CoG-only, monocular depth mask** | EXP-01 | 16.95% ± 2.23 | 16.95% | 0.165 | 20.60° | 885 | Same as above, mask from DepthAnythingV2-Small |
| Heuristic (full pipeline) | EXP-02 | 16.72% ± 1.05 | 20.34% | 0.285 | 65.17° | 885 | DepthAnything → contour → 80px PCA → ray-cast → CoG-boost rank. Re-run 2026-05-27 after fixing output-convention bug (`height` was hardcoded 20 px; now measured as perpendicular contour extent; `angle_rad` now wrapped to [-π/2, π/2]). Pre-fix: 8.70% ± 2.33 Top-1. Now ≈ tied with EXP-01 mono CoG; angle error still elevated, suggests algorithmic mis-orientation on some objects (not a convention bug). |
| GR-ConvNet v1 (reproduced) | EXP-03 (pending) | — | — | — | — | — | Public checkpoint, RGB-D input |
| Xiangli 2025 (reproduced) | EXP-08 (pending) | — | — | — | — | — | Training-free + foundation models (CLIP+SAM+DIFT) |

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
