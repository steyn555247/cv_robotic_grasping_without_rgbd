# `src/` — Research code (production)

This is the **only** place algorithm code may live. Anything you'd want a coauthor to read or reproduce goes here.

## Layout

```
src/
├── methods/             # one folder per method
│   ├── heuristic/       # our main contribution (EXP-02 builds this)
│   ├── cog_baseline/    # trivial 2D-centroid baseline (EXP-01)
│   └── grconvnet_repro/ # public checkpoint wrapper (EXP-03)
├── eval/                # canonical evaluators — ONE per dataset
│   ├── cornell.py       # owned by eval-harness, never edited by experiment-runner
│   └── jacquard.py
├── data/                # dataset loaders + frozen split definitions
│   ├── cornell_loader.py
│   ├── jacquard_loader.py
│   └── splits/          # *.json — IMMUTABLE once committed
│       └── cornell.json
└── utils/               # shared utilities (geometry, io, viz)
```

## Hard rules

1. **Methods import from eval, eval never imports from methods.** Eval is downstream.
2. **One evaluator per dataset.** No method computes its own metrics.
3. **Splits are frozen.** Once `splits/cornell.json` is committed, never edit. Add `cornell_v2.json` if a different split is needed.
4. **No notebooks in `src/`.** Notebooks live in `experiments/` if they live anywhere.
5. **All public functions get type hints + a one-line docstring.** Reviewers read the code.

## Not in here

- Course-project archive (`Heuristics approach/`, `cornell comparisson/`, etc.) — those are frozen historical snapshots, kept for reference but not modified.
- The Streamlit demo app — that lives where it already lives.
- Experiment-specific glue (one-off plotting, etc.) — lives in `experiments/EXP-XX/`.
