---
description: Interactively draft a new experiment spec.md (no code yet — just the contract)
argument-hint: <experiment-name>
---

I want to create a new experiment spec at `experiments/EXP-XX_$ARGUMENTS/spec.md`.

Before writing anything, work through these questions with me:

1. **Hypothesis** — what specifically are we trying to find out? (One sentence, falsifiable.)
2. **Method** — what gets implemented, in 3-5 numbered steps? What goes in `src/` vs. what is one-off glue?
3. **Dataset & split** — Cornell image-wise / object-wise / Jacquard / something custom?
4. **Success criteria** — what numeric outcome makes us call this experiment "done"? What range is interesting?
5. **Dependencies** — does this need a prior experiment's predictions, a downloaded dataset, a refactored module?
6. **Effort estimate** — your honest guess in hours.
7. **Why now?** — what decision does this unblock?

Once we agree on those, write the spec to `experiments/EXP-XX_$ARGUMENTS/spec.md` in this format:

```markdown
# EXP-XX: <title>

**Status**: spec drafted | running | done
**Effort estimate**: <N> hours
**Owner**: experiment-runner
**Blocked by**: EXP-YY (if any)

## Hypothesis
<one sentence>

## Method
1. ...
2. ...
3. ...

## Dataset & split
<dataset>, <split definition path>

## Success criteria
- Primary metric: <metric> in range [X, Y]
- Decision: if primary metric falls in <range>, then <consequence for paper>
- Secondary outputs: ...

## What this experiment unblocks
- ...

## Files this experiment may modify
- experiments/EXP-XX_<name>/run.py (new)
- src/methods/<name>/... (new or extended)
- experiments/EXP-XX_<name>/results.json (new)
- RESULTS.md (append row)
- PROGRESS.md (mark complete)

## Files this experiment may NOT touch
- src/eval/cornell.py (canonical evaluator — only eval-harness modifies)
- src/data/splits/*.json (frozen)
- other experiments' folders
```

After we agree on the spec, the next step is `/run-experiment EXP-XX_$ARGUMENTS` — not now.

Number the experiment by reading `experiments/` and finding the next free EXP-XX.
