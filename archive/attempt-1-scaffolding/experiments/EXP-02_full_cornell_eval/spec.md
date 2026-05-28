# EXP-02: Full Cornell evaluation of the current heuristic

**Status**: spec drafted
**Effort estimate**: 1-2 days (most of it is refactoring the Streamlit script into a clean library function)
**Owner**: experiment-runner
**Blocked by**: `src/eval/cornell.py`, `src/data/cornell_loader.py` (infra prerequisites)

## Hypothesis

The current heuristic pipeline (DepthAnythingV2 + contour + local PCA tangent + ray-cast + weighted score), evaluated on the **full Cornell dataset** with the **canonical 5-fold image-wise split** (not the manually cropped subset used in the original grid search), achieves Top-1 in the range 70-80% — establishing a defensible headline number that survives reviewer scrutiny.

## Method

1. **Extract the algorithm from the Streamlit script.** The current code lives in `Heuristics approach/grasp_detection_contour_80px.py` and is intertwined with `streamlit.*` calls. Refactor the pure-algorithm portion into `src/methods/heuristic/detect.py` exposing a single function:

   ```python
   def detect_grasp(
       image_rgb: np.ndarray,
       depth_model: DepthEstimator,
       config: HeuristicConfig,
   ) -> list[GraspRect]:
       """Return top-N grasp rectangles ranked by combined score."""
   ```

   Use the grid-search winners as default config:
   - `w_edge=0.001, w_depth=0.001, w_cog=0.998`
   - `depth_percentile=30`
   - `cog_boost=3.75`
   - `gradient_source="Contour Direction (80px avg)"`
   - `min_grasp_length=1, max_grasp_length=1000`
   - `candidate_multiplier=100`
   - `num_output_grasps=5`

2. **Wire up depth model.** Use HuggingFace `depth-anything/Depth-Anything-V2-Small-hf` by default (fits on consumer GPU, quick inference). Document the version.

3. **Run on all 855 Cornell images.** No cropping. Image-wise 5-fold split, then object-wise.

4. **Evaluate** with `src/eval/cornell.py:evaluate_predictions`. Save per-sample predictions to `predictions/` so EXP-04 can ablate against this baseline.

5. **Report**: Top-1, Top-5, IoU mean, angle error, runtime per image, both folds.

## Dataset & split

- Cornell Grasping Dataset, full (855 images).
- Splits: `src/data/splits/cornell.json` — both image-wise and object-wise 5-fold.

## Success criteria

| Outcome | Top-1 (image-wise) | Paper consequence |
|---|---|---|
| Number holds | ≥ 73% | Defensible headline, matches/beats Lenz 2015. Continue plan. |
| Number drops | 60-73% | Investigate cause (cropping was helping a lot?). Reframe modestly. |
| Number tanks | < 60% | Method genuinely overfit to crop. Major revision needed; consider falling back to CoG-only as the headline. |

Secondary observations to note in `notes.md`:
- Image-wise vs. object-wise gap (large gap = overfits to per-object features).
- Per-fold variance.
- Sample categories where the heuristic fails (qualitative cases for paper Fig. 3).
- Runtime budget (informs throughput claim in EXP-05).

## Files this experiment may modify

- `src/methods/heuristic/__init__.py` (new)
- `src/methods/heuristic/detect.py` (new; ~300 lines refactored from Streamlit)
- `src/methods/heuristic/config.py` (new; HeuristicConfig dataclass)
- `src/methods/heuristic/depth.py` (new; DepthEstimator wrapper around HF)
- `experiments/EXP-02_full_cornell_eval/run.py` (new)
- `experiments/EXP-02_full_cornell_eval/results.json` (new)
- `experiments/EXP-02_full_cornell_eval/predictions/` (new)
- `experiments/EXP-02_full_cornell_eval/notes.md` (new)
- `RESULTS.md`, `PROGRESS.md`

## Files this experiment may NOT touch

- `src/eval/cornell.py` — canonical evaluator
- `src/data/splits/*.json` — frozen
- `Heuristics approach/` and `cornell comparisson/` — these are the **archive** of the course project; do not modify, only read for reference

## What this experiment unblocks

- EXP-03 (GR-ConvNet head-to-head) needs a number to compare against on the same split.
- EXP-04 (ablation) needs the full pipeline's predictions as the reference.
- EXP-07 (auto-label training) needs the full pipeline as the label generator.
- `method.tex` can be drafted from `src/methods/heuristic/`.

## Notes for the runner

- Before any refactor of the Streamlit code, **suggest the user run `/freeze refactor-heuristic`** to tag a known-good state.
- The Streamlit code has a lot of UI cruft — the algorithmic core is maybe 200 lines. Don't carry the cruft into `src/`.
- The PCA tangent computation in the original code analytically solves a 2x2 eigenproblem — that's fine, keep it but add a one-line comment citing the math.
- If the DepthAnythingV2-Small produces noticeably worse results than DepthAnythingV2-Large, note it but use Small for the headline (cheaper, the cost story matters).
- Random seed: 0 throughout.
