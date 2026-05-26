---
description: Re-run canonical evaluation on every method's saved predictions, regenerate RESULTS.md
---

Dispatch the **eval-harness** subagent with this task:

1. Find every `experiments/EXP-*/predictions/` folder in the repo.
2. For each, re-run `src/eval/cornell.py:evaluate_predictions` on the saved predictions vs. ground truth.
3. Confirm the resulting metrics match what's in that experiment's `results.json` — flag any discrepancy.
4. Update / regenerate `RESULTS.md` with the current canonical numbers (point estimates only here; CIs come from results-analyst).

Then dispatch **results-analyst** to:
1. Recompute 95% bootstrap CIs for every Top-1 in RESULTS.md.
2. Recompute pairwise McNemar comparisons.
3. Update `RESULTS.md`'s comparison table.

Final output: a diff summary of which numbers changed since last `/eval-all`.
