# `src/methods/` — Method implementations

One folder per method. Each method exposes a callable that maps `(image, [aux inputs]) -> list[GraspRect]`, ranked best-first.

## Convention

```
methods/<name>/
├── __init__.py
├── detect.py          # the main entry point
├── config.py          # @dataclass holding all hyperparameters
└── README.md          # one paragraph: what this method does, where it came from
```

## Current / planned methods

| Name | Folder | Source | Experiment |
|---|---|---|---|
| Heuristic (ours) | `heuristic/` | refactor of Streamlit script | EXP-02 builds it |
| CoG-only baseline | `cog_baseline/` | trivial centroid + PCA | EXP-01 builds it |
| GR-ConvNet | `grconvnet_repro/` | public checkpoint wrapper | EXP-03 builds it |
| Xiangli 2025 | `xiangli_repro/` | reproduction (if code released) | EXP-08 builds it |

## Hard rules

- **No method computes its own Cornell metric.** Import from `src.eval.cornell`.
- **No method writes results.json.** That's the experiment's job; methods only return predictions.
- **No method mutates global state.** Pure function, deterministic given seed.
