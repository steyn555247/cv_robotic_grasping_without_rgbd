# EXP-01: CoG-only trivial baseline

**Status**: spec drafted
**Effort estimate**: 4-6 hours (mostly waiting on dataset loader + canonical evaluator)
**Owner**: experiment-runner
**Blocked by**: `src/eval/cornell.py`, `src/data/cornell_loader.py`, `src/data/splits/cornell.json` (infra prerequisites — should be built first)

## Hypothesis

A grasp predictor that **completely ignores image features** — outputting a single rectangle centered at the foreground centroid with a fixed orientation derived from the segmentation mask's principal axis — achieves Cornell Top-1 ≥ 65%, demonstrating that the headline finding (CoG-dominance) is largely a property of the benchmark rather than a property of the method.

## Method

1. **Foreground segmentation** — for each Cornell sample, produce a binary foreground mask using one of:
   - (a) Ground-truth depth thresholding (uses Cornell's depth channel for the mask only — *not for grasping* — making this an upper bound on what mask-based methods can do); OR
   - (b) Monocular-depth thresholding using the same DepthAnythingV2 model used by the main method, to keep the comparison strictly RGB-only.
   Run both versions and report both. The RGB-only number is the one that goes in the paper.

2. **Compute centroid** of the foreground mask: `(c_x, c_y) = mean of mask pixel coordinates`.

3. **Compute principal axis** of the foreground mask via PCA on mask pixel coordinates. The grasp angle is the angle of the **minor** principal axis (perpendicular to the major axis, since the gripper closes across the narrow direction).

4. **Set grasp width** to a fixed fraction (default: 0.6) of the minor-axis extent — i.e., the typical width of the object along the gripper closing direction.

5. **Set grasp height** to a fixed 20 pixels (gripper jaw thickness, matching Cornell convention).

6. **Output a single grasp rectangle** per image at `(c_x, c_y, angle_minor, width, 20)`. No ranking — Top-1 = Top-5 for this method.

7. **Evaluate** with `src/eval/cornell.py:evaluate_predictions` on:
   - Image-wise 5-fold split (primary)
   - Object-wise 5-fold split (sanity check)

## Dataset & split

- Cornell Grasping Dataset, full (855 images).
- Splits: `src/data/splits/cornell.json` — both image-wise and object-wise 5-fold.
- Use `cornell comparisson/cornell_dataset/` as the data source (already on disk, do not re-download).

## Success criteria

| Outcome | Top-1 range | Paper consequence |
|---|---|---|
| **CoG-only dominates** | ≥ 72% | Pivot to "Cornell is easier than reported" critique framing. The full heuristic pipeline becomes a refinement, not the main result. |
| **CoG-only is strong baseline** | 60-72% | Keep the "training-free annotator" framing. CoG-only is the strongest classical baseline; full heuristic must beat it by ≥5pp to justify complexity. |
| **CoG-only is weak** | < 60% | Headline becomes the full pipeline. CoG-only is reported as ablation context only. |

Secondary metrics:
- Per-fold variance (CIs).
- Object-wise vs. image-wise gap (informs whether Cornell's per-object generalization is the easy / hard regime).
- Per-class breakdown if time permits.

## Files this experiment may modify

- `src/methods/cog_baseline/__init__.py` (new)
- `src/methods/cog_baseline/detect.py` (new; ~50 lines)
- `experiments/EXP-01_cog_only_baseline/run.py` (new)
- `experiments/EXP-01_cog_only_baseline/results.json` (new)
- `experiments/EXP-01_cog_only_baseline/predictions/` (new; per-sample predictions)
- `experiments/EXP-01_cog_only_baseline/notes.md` (new)
- `RESULTS.md` (append row)
- `PROGRESS.md` (mark complete)

## Files this experiment may NOT touch

- `src/eval/cornell.py` — canonical evaluator
- `src/data/splits/*.json` — frozen
- other experiments' folders

## What this experiment unblocks

- The **framing decision** for the entire paper (CoG-critique vs. annotator).
- EXP-04 (ablation) needs a "no-features" baseline to anchor the ablation table.
- The intro can't be drafted until we know the headline.

## Notes for the runner

- The mask-from-monocular-depth version is the one that goes in the paper headline; the GT-depth version is a methodological upper bound, mentioned in passing.
- Save `predictions/` so EXP-04 can do paired McNemar comparisons against the full pipeline.
- Sanity check: `evaluate_predictions(ground_truth, ground_truth)` must return Top-1 = 1.0 before running. If not, eval-harness has a bug.
- Random seed: 0 throughout.
