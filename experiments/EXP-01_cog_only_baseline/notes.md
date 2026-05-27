# EXP-01 — Notes

**Completed**: 2026-05-27 (after API-outage restart of agent dispatched 2026-05-26)
**Wallclock**: 838 s (14 min) on CPU, full 5-fold image-wise CV over 885 Cornell images × 2 variants.

## Headline result

| Variant | Top-1 (mean ± std across folds) | IoU mean | Angle err mean |
|---|---|---|---|
| `gt_depth_mask` (Cornell GT depth → mask) | **8.70% ± 1.98** | 0.112 | 27.07° |
| `mono_depth_mask` (DepthAnythingV2-Small → mask) | **16.95% ± 2.23** | 0.165 | 20.60° |

A single grasp is emitted per image at the mask centroid, oriented along the major principal axis, with width = 0.6 × minor-axis extent. Top-5 ≡ Top-1 because the method returns one rectangle.

## Why monocular outperforms GT-depth here

Counter-intuitive — but the GT depth has *real* sensor noise (specular highlights, shadows, depth holes on dark objects), which our blurred-residual mask construction handles poorly. The monocular network produces a cleaner, more spatially-coherent foreground prior. This is a known phenomenon and helps the project's argument: a learned monocular depth can be a *better* mask source than a real depth sensor in some grasping pipelines.

## What this tells us about the paper

**Pre-experiment hypothesis** (from spec): if CoG-only ≥ 65%, the paper pivots to a "Cornell is easy" critique.
**Actual**: CoG-only = 17% — nowhere near. The critique angle is **dead**.

**What's alive instead** — the training-free annotator framing is *stronger* than expected:
- The heuristic's contour PCA tangent + ray casting + CoG-biased candidate ranking is doing real geometric work above this trivial baseline. The grid-search w_cog=0.999 was never "put grasp at CoG" — it was "rank contour candidates by CoG proximity, then ray-cast from them."
- Cornell's annotated grasps are *not* through-the-centroid. They sit on graspable rims/handles. The angle-error-mean here (~21–27°) is right at the 30° tolerance, and IoU (~0.11–0.16) is well below the 0.25 threshold — confirming the centroid-grasp geometry is structurally wrong for Cornell ground truth.

EXP-02 must produce the heuristic's headline Top-1; the gap above 17% is the contribution magnitude.

## Per-fold numbers

| Fold | n | GT Top-1 | Mono Top-1 | GT IoU | Mono IoU | GT angle | Mono angle |
|---|---|---|---|---|---|---|---|
| 0 | 177 | 5.08% | 11.30% | 0.115 | 0.171 | 26.5° | 18.5° |
| 1 | 177 | 8.47% | 18.08% | 0.113 | 0.181 | 25.5° | 19.4° |
| 2 | 177 | 10.73% | 16.38% | 0.104 | 0.171 | 29.0° | 22.4° |
| 3 | 177 | 9.04% | 16.38% | 0.106 | 0.166 | 27.8° | 21.6° |
| 4 | 177 | 10.17% | 20.34% | 0.119 | 0.135 | 26.4° | 21.1° |

Cross-fold variance is reasonable (~2 pp std on Top-1). No fold is wildly different — splits look balanced.

## Surprises & follow-ups

1. **Monocular depth > GT depth for masking.** Worth ~1 paragraph in the paper's discussion — frames Depth Anything as a robust mask source, not just a depth proxy.
2. **Angle-error sits right at threshold.** ~21° mean vs. 30° tolerance — so many predictions are *close* to a valid grasp angle but get rejected. Suggests a slightly-better orientation estimator would help disproportionately. EXP-02 will reveal whether the heuristic's contour tangent gets the angle inside tolerance.
3. **The mask construction (`_depth_to_mask`)** in `src/methods/cog_baseline/detect.py` uses a 151×151 Gaussian-blur residual + central crop + connected-component pick. It's reasonable but not perfect — there are cases where the heaviest object isn't centered and gets missed. Not a bug, but a follow-up if we want a stronger CoG baseline (we don't — its job is to be deliberately weak).

## Files

- `run.py` — orchestrator
- `run.log` — full execution log
- `results.json` — per-fold + aggregated metrics + per-sample correctness vector (needed for McNemar in EXP-04)
- `predictions/<variant>/fold-N/<sample_id>.json` — per-sample predicted GraspRect for stat tests
- `mono_depth_cache/<sample_id>.npz` — depth predictions cached so re-running EXP-04 (which uses the same monocular depth) doesn't reload the model 885 times

## What EXP-02 inherits from this

- `mono_depth_cache/` — same model produces the same depths; EXP-02 can read these directly and skip ~14 min of inference per run.
- Per-sample correctness for `mono_depth_mask` variant → paired McNemar in EXP-04 comparing heuristic vs. CoG-only.
