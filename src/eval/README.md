# `src/eval/` — Canonical evaluators

**Only `eval-harness` touches these files.** Every other agent and experiment imports from here.

## Files

- `cornell.py` — Cornell grasp-rectangle evaluator. Jaccard ≥ 0.25 AND angle error ≤ 30°. Image-wise + object-wise.
- `jacquard.py` — Jacquard equivalent. Same thresholds.
- `tests/` — property-based tests. `evaluate(gt, gt) == 1.0`, perpendicular grasps == 0.0, 180° rotation symmetry holds.

## Why one source of truth

Grasp papers get rejected for inconsistent evaluation as often as for weak methods. Numbers between methods are only comparable if they came out of the same evaluator. So there is one.

## When to extend

- New dataset → add `<dataset>.py` and `tests/test_<dataset>.py`.
- New metric → add a function, do not modify `evaluate_predictions` silently.
