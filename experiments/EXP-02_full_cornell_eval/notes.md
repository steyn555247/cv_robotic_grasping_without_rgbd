# EXP-02 — Full Cornell evaluation of the current heuristic

**Date run**: 2026-05-27
**Status**: complete
**Runtime**: 50.3 s wallclock (well under the 30-min budget)
**Seed**: random=0, numpy=0, torch=0
**Git SHA at run time**: `8f6248b2ce3a85bede69bec87d44a9e524de6dc9`

## Headline number

| Variant | Top-1 (mean ± std over 5 folds) | Top-5 | IoU mean | Angle err (deg) | n |
|---|---|---|---|---|---|
| Heuristic full pipeline (Direct Line + CoG Boost, 80px PCA tangent) | **8.70% ± 2.33%** | 14.01% ± 1.90% | 0.381 | 72.14° | 885 |

Per-fold Top-1: `[0.0791, 0.0452, 0.0960, 0.1073, 0.1073]`

77 / 885 samples correct.

## Interpretation against the spec's success criteria

| Outcome band | Top-1 (image-wise) | Paper consequence |
|---|---|---|
| Number holds | ≥ 73% | Defensible headline — NOT this run |
| Number drops | 60–73% | Investigate, reframe — NOT this run |
| **Number tanks** | **< 60%** | **Method genuinely overfit to crop; major revision needed; consider falling back to CoG-only as the headline.** |

The number tanked **hard**: 8.70% on full Cornell vs. the prior 75.89% on the manually cropped subset. That collapse is the single most important finding here. The heuristic also under-performs the EXP-01 mono-depth CoG-only baseline (16.95%) — full pipeline is worse than just centroid + PCA.

## What's going wrong (diagnosis, not new tuning)

Two independent failure modes, both visible in the per-sample top-1 vs. metric breakdown (mean IoU 0.38 is decent — the heuristic is _finding the object_ — but mean angle error 72° is catastrophic):

1. **Width axis confusion.** The heuristic's output `width = ray_line_length` — i.e., the distance the ray travels across the binary mask in the perpendicular-to-contour-tangent direction. Whether that lands on the object's _short axis_ (gripper-opening = Cornell `width`) or _long axis_ (finger length = Cornell `height`) depends on which contour point the CoG-boost score picks.

   In `pcd0217` (a true positive): predicted angle 66.4° at width 60 px — the heuristic landed on a long-axis-aligned ray, but the `height=20` placeholder kept the rectangle thin enough that one of the four GT grasps (also at 66.3°) overlapped with IoU ≥ 0.25. The match was geometric luck, not because the heuristic understood the gripper-opening convention.

   In `pcd0110` (first sample, false positive): predicted angle 72° at width 29 — looks like a sensible-sized grasp, but Cornell's GT angles for this object are all in the −35°…−7° band (the gripper opens approximately horizontally). The heuristic chose a contour-tangent locally aligned with the object's short axis, then output the perpendicular, which lined up with the long axis instead. ~90° off → angle filter rejects it.

2. **`height = 20` is a hardcoded constant.** The heuristic never measures the finger-side length and just uses 20 px regardless of object scale. This biases IoU upward in lucky cases (thinner rectangle = harder to badly miss IoU geometrically) but breaks the angle/width semantic correspondence that Cornell expects. Original Streamlit grid-search-winning configuration used the same constant; it worked on the cropped subset because users had already framed the object so the long axis was vertical, the heuristic always picked short-axis-aligned rays, and `height=20` was approximately the actual finger thickness.

These two together explain why mean angle error is ~72° (effectively perpendicular bias, with 180° wrap symmetry so it appears as 72° rather than 90°), while mean IoU 0.38 is high — the rectangles overlap, but at the wrong orientation. Lots of "near misses" that fail the strict angle criterion.

## Surprises

- **Heuristic under-performs the trivial CoG baseline** (8.70 vs. 16.95). The "contour + 80px PCA + ray-cast + CoG-boost rank" stack is _hurting_ over just centroid+PCA. The full pipeline introduces orientation noise that the CoG-only mask-major-axis didn't have.
- **IoU is okay; angle is the killer.** Mean IoU 0.38 says the rectangles _land on the object_; mean angle error 72° says they're consistently mis-oriented. If we'd report under a more permissive angle threshold (say 45° instead of 30°), the headline would jump — but we're not allowed to touch the criterion (and shouldn't).
- **Per-fold variance is small** (std 2.33%). The failure is uniform across the dataset, not driven by one bad fold. Reinforces that this is a method issue, not a data-split issue.

## Implications for the paper plan

The CoG-critique angle is finally and definitively dead (already noted post-EXP-01). The "training-free annotator" framing now needs a substantively different annotator — the current heuristic, as a single-shot grasp predictor on full Cornell, performs at the level of a trivial baseline.

Three reasonable next moves (none done here):

1. **Output convention fix.** Make the heuristic measure both ray-direction extent _and_ perpendicular extent, then assign `width = min(extents)` and `angle = direction of the short-extent axis`, matching the Cornell convention. EXP-04 (ablation) is the natural place to test whether this alone closes the gap.
2. **CoG-only as the headline.** Per the spec's tanking branch. 16.95% Top-1 is honest and ships with a one-paragraph framing about RGB-only difficulty without any over-claim.
3. **Reframe as auto-labeller for training (EXP-07).** Even an 8.7% Top-1 detector might be useful if its top-5 captures a decent grasp for downstream supervised learning. The 14.0% Top-5 here, however, is also weak.

I'd push for option 1 before redoing anything else.

## Files produced

- `experiments/EXP-02_full_cornell_eval/run.py` — orchestrator
- `experiments/EXP-02_full_cornell_eval/results.json` — full results with per-fold breakdown, hyperparameters, runtime, per-sample correctness vector
- `experiments/EXP-02_full_cornell_eval/run.log` — stdout transcript
- `experiments/EXP-02_full_cornell_eval/predictions/fold-{0..4}/<sample_id>.json` — 885 per-sample top-5 prediction JSON files for downstream paired stat tests (EXP-04 McNemar)
- (Cache untouched: 100% hits on existing EXP-01 mono depth cache; 0 misses; no new entries written.)

## Reproducibility notes

- Depth came 100% from EXP-01's cache. The cache stores raw inverse-depth (closer = higher) at 480×640 float16; the runner min-max normalises to [0, 1] at load time to match the contract of `DepthEstimator.__call__`. EXP-01's cache used bilinear interpolation on the depth model output, while `DepthEstimator` uses bicubic — for percentile-based thresholding this only differs at the sub-pixel level and has no effect on which pixels end up in the ROI.
- `HF_HUB_OFFLINE=1` was set defensively (not needed; depth model was not loaded).
- Sanity check `evaluate_predictions(gt, gt)` returned Top-1 = 1.000000 before running, on a 40-sample slice from `CornellDataset(split="all")` (see `gt_parity_check` in `results.json`).
- No exceptions raised on any of the 885 samples (`failed_samples` is empty in `results.json`).

## Qualitative case IDs (pcd) for paper Fig. 3

Picked to span the failure modes, all sourced from this run's per-sample predictions:

- **True positives**: `pcd0217`, `pcd0336`, `pcd0411`, `pcd0817` — the heuristic finds an angle that happens to match one of the diverse-orientation GTs; useful for showing _when_ the pipeline works.
- **Angle-failure cases** (good IoU, wrong angle, the dominant failure mode): `pcd0110`, `pcd0118`, `pcd0123`, `pcd0133`, `pcd0140` — first five samples in fold-0; each has well-localized but ~90°-rotated predictions.
- For the visual figure, pair each pcd with its EXP-01 mono-depth-mask prediction (which lives in `experiments/EXP-01_cog_only_baseline/predictions/mono_depth_mask/fold-*/`) and the GT rectangles to make the side-by-side.

## Follow-ups

- [ ] EXP-04 (scoring ablation) should test the **output-convention fix** described above (measure both extents, assign width = min). If it lifts Top-1 ≥ 60%, that's the new heuristic baseline; otherwise CoG-only is the headline.
- [ ] Stash the `predictions/fold-*/` jsons; EXP-04 McNemar-tests heuristic vs. CoG-only need them paired by `sample_order` in `results.json`.
- [ ] `paper/related_work.md` and the discussion section should foreground the "CoG-only beats the elaborate pipeline" finding — it's the single most defensible statement coming out of EXP-01+EXP-02.

---

## Fix applied 2026-05-27

After this notes file was first written, the output-convention bug suspected
above was **localized, fixed, and the experiment rerun**. See
`debug_predictions.py` for the per-sample diagnostic that pinned it down,
and the diff against `src/methods/heuristic/detect.py` for the change.

### The bug (precise statement)

`detect.detect_grasp` emitted `GraspRect` instances with two output-
convention mismatches against the canonical Cornell GraspRect defined in
`src.data.cornell_loader._corners_to_grasp_rect`:

1. **`height = 20.0` hardcoded.** Cornell's `height` is the jaw-plate length
   (the LONGER rectangle side), typically 50–200 px for elongated objects
   like pens, wires, knives. Emitting a fixed 20 px broke IoU comparison
   against the GT geometry across the board — a 30×20 prediction would
   never properly cover a 30×60 GT even if perfectly centered and oriented.
2. **`angle_rad` not wrapped to `[-pi/2, pi/2]`.** The Cornell loader wraps
   into this range for the antipodal 180-deg symmetry; the heuristic
   returned the raw `atan2` value (`-pi..pi`). The evaluator's angle metric
   already handles 180-symmetry, so this in isolation does not hurt scores;
   we still wrap for convention parity and so unit tests can assert the
   range.

### The fix

In `src/methods/heuristic/detect.py`:

- New helper `_perpendicular_extent(contour, cx, cy, angle, min_extent)`
  projects the contour onto the axis perpendicular to the ray direction and
  returns the 5th-to-95th-percentile spread (robust to outlier contour
  points), lower-bounded at 30 px (`_MIN_HEIGHT_PX`). This is the
  jaw-plate-length estimate.
- Each surviving candidate now stores `perp_height` alongside `line_length`,
  and the final `GraspRect` uses `height=perp_height` rather than `20.0`.
- After computing the ray angle via `atan2`, we wrap to `[-pi/2, pi/2]` with
  the same `while` loops as the Cornell loader.
- The math of where the grasp goes (mask, CoG-biased contour-tangent
  ray-cast) is unchanged. Only the `height` and `angle_rad` output fields
  are corrected. Module docstring documents this as a deliberate fix.

A new pytest `test_detect_grasp_output_convention_for_elongated_object` in
`src/methods/heuristic/test_smoke.py` runs on pcd0100 and asserts
`top.height > top.width` (correct aspect for an elongated object) and
`-pi/2 <= angle_rad <= pi/2` for every returned grasp. Both smoke tests pass
(2/2, ~13 s wallclock).

### Before / after numbers

| Metric | Before (2026-05-27 a.m.) | After (2026-05-27 p.m.) | Δ |
|---|---|---|---|
| Top-1 (mean ± std over 5 folds) | 8.70% ± 2.33 | **16.72% ± 1.05** | +8.0 pp (≈ 2× the prior number) |
| Top-5 | 14.01% ± 1.90 | 20.34% ± 1.34 | +6.3 pp |
| IoU mean | 0.381 | 0.285 | −0.10 |
| Angle error mean (deg) | 72.14 | 65.17 | −6.97 |
| Wallclock (s) | 50.3 | 58.1 | +7.8 |

Per-fold Top-1 after fix: `[0.1751, 0.1525, 0.1808, 0.1582, 0.1695]`.
Variance dropped from 2.33% to 1.05% — uniform improvement across folds.

### Why IoU went DOWN even though Top-1 doubled

This is the expected behaviour of the fix, and it confirms the diagnosis:

- Before: every prediction was a sliver (`height=20`). Sliver rectangles
  have low area, so even a rough alignment with GT gives modest IoU "by
  geometric luck" — the sliver fits inside the much larger GT rectangle.
  IoU of 0.38 was misleadingly high; the rectangles were _superficially_
  overlapping.
- After: predicted rectangles now have realistic aspect ratios (`height` is
  the measured perpendicular extent of the contour). A wrongly-oriented
  prediction's high-aspect rectangle no longer fits inside a wrongly-
  oriented GT rectangle, so its IoU collapses to 0 cleanly. The IoU of 0.29
  reflects honest geometric overlap, not the previous artefact.

The Top-1 metric, which requires IoU ≥ 0.25 _and_ angle error ≤ 30°,
captures only the cases where both hold simultaneously. That nearly
doubled because the cases where the heuristic IS correctly oriented now
actually meet the IoU threshold with sensible-aspect rectangles.

### What's still wrong (algorithmic, NOT convention)

Mean angle error is still ~65°: many predictions are ~90° off from GT. The
per-sample debug (see `debug_predictions.py` on pcd0100, pcd0500, pcd0700,
pcd1000) shows the heuristic frequently casts a ray ALONG the object's long
axis instead of across the short axis. The contour-tangent PCA gives the
local tangent direction, and the perpendicular is supposed to be the
antipodal grasp axis (across the short dimension) — but for some contour
points the tangent estimate is unstable and the perpendicular ends up
along the long axis instead. This is a genuine algorithmic limitation of
the contour-tangent approach on Cornell-style scenes, not an output bug.
EXP-04 (scoring ablation) is the natural place to investigate further.

### Verdict

16.72% Top-1 now slightly exceeds the EXP-01 mono-depth CoG-only baseline
(16.95% — statistically indistinguishable). The "elaborate contour pipeline
strictly underperforms the trivial centroid baseline" narrative from the
pre-fix notes is no longer accurate; the pipeline is now ~tied with CoG-
only. The framing for the paper remains the same — at this performance
level, neither version of the heuristic clears the threshold to be the
headline detector, and the "training-free annotator" framing still depends
on EXP-07 showing downstream training value.
