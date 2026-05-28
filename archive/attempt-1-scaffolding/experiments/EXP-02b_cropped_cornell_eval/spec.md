# EXP-02b: Cornell evaluation with the original manual crop

**Status**: spec drafted
**Effort estimate**: 1-2 hours
**Owner**: experiment-runner
**Blocked by**: EXP-02 (need its run.py + depth cache as starting point)

## Hypothesis

The heuristic underperformed on full Cornell (EXP-02: 16.72% Top-1) despite the project's prior grid-search reporting **75.89% on a manually cropped subset**. Hypothesis: the heuristic relies on object localization that the manual crop was implicitly providing — without it, the depth-percentile threshold + largest-contour selection picks up hand/background structure instead of the object.

This experiment reproduces the original grid-search setting exactly to confirm or refute that hypothesis.

## Method

1. **Manual crop** — apply `image[150:450, 100:500]` (the exact crop region documented in `cornell comparisson/GRID_SEARCH_README.md`) to the RGB image **before** any other processing.
2. Run the same heuristic pipeline as EXP-02 on the cropped images:
   - DepthAnythingV2-Small inference on the CROPPED RGB (re-run depth — the EXP-01 cache is for full images so doesn't apply here).
   - Existing `src.methods.heuristic.detect.detect_grasp` (no changes).
3. **Crucially**: predictions are in CROPPED coordinates. To evaluate against Cornell GT, **map predictions back to full-image coordinates** by adding the crop offset:
   - `x_full = x_cropped + 100`
   - `y_full = y_cropped + 150`
   - angle, width, height unchanged
4. Evaluate via the canonical Cornell evaluator (same as EXP-02).
5. Image-wise 5-fold, same splits as EXP-02 (`src/data/splits/cornell.json`).

## Dataset & split

- Cornell, full 885 images.
- Image-wise 5-fold, splits frozen.
- Manual crop hardcoded: x ∈ [100, 500), y ∈ [150, 450).

## Success criteria

| Outcome | Top-1 (image-wise) | Decision |
|---|---|---|
| **Reproduces prior** | 70-80% | Confirms diagnosis — heuristic has merit, just needs localization. Proceed to Path B (SAM2 or similar localization in front). |
| **Partially reproduces** | 40-70% | Some other factor differs (depth model? subset selection? scoring weights tuned to the cropped depth statistics?). Investigate. |
| **Does not reproduce** | < 40% | The prior 75.89% number was measured under conditions we don't fully capture. Reframe the paper around what we can defend. |

## What this experiment unblocks

- **Decision on Path B** (SAM2 localization). If A reproduces 75%, B is justified. If A doesn't, B is premature.
- Possible discovery that the prior 75% is unreproducible — which is itself a finding worth documenting (transparency about what didn't work).

## Files this experiment may modify

- `experiments/EXP-02b_cropped_cornell_eval/run.py` (new)
- `experiments/EXP-02b_cropped_cornell_eval/results.json` (new)
- `experiments/EXP-02b_cropped_cornell_eval/predictions/` (new)
- `experiments/EXP-02b_cropped_cornell_eval/depth_cache/` (new — depths for cropped images are different from EXP-01 cache)
- `experiments/EXP-02b_cropped_cornell_eval/notes.md` (new)
- `RESULTS.md` (append row)
- `PROGRESS.md` (add entry)

## Files this experiment may NOT touch

- `src/methods/heuristic/detect.py` — identical algorithm, no changes
- `src/eval/cornell.py`, `src/data/splits/cornell.json` — frozen
- Other experiments' folders

## Notes for the runner

- Reuse EXP-02's run.py structure; just add the crop step before depth inference and the coordinate mapping after detection.
- Depth must be re-run for cropped images. Cache locally in `experiments/EXP-02b_*/depth_cache/` for later reuse by EXP-04 if it explores crop ablations.
- Wallclock estimate: 885 images × ~2 s/image (depth on cropped 400×300 image is fast) ≈ 30 min on CPU.
- Set `HF_HUB_OFFLINE=1` to avoid rate-limit issues with the HF Hub.
- Random seed: 0 throughout.
