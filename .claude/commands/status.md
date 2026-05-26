---
description: Print the current PROGRESS.md dashboard with derived state
---

Read `PROGRESS.md`, `BACKLOG.md`, and the file listing of `experiments/` and `paper/sections/`.

Produce a concise status report:

```
## EXPERIMENTS
EXP-01 cog_only_baseline       [✓ done   ] Top-1: 71.5%
EXP-02 full_cornell_eval       [⧖ running] started 2026-01-30
EXP-03 grconvnet_repro         [⊘ blocked] needs GR-ConvNet checkpoint
EXP-04 ablation                [○ pending]
...

## PAPER
abstract.tex     [○ not started]
intro.tex        [○ not started]
related_work.tex [⧖ drafted, needs revision]
method.tex       [○ not started]
...

## NEXT ACTIONS (from BACKLOG.md critical path)
1. Complete EXP-02 (currently running)
2. Start EXP-03 (download GR-ConvNet checkpoint first)
3. Draft method.tex

## BLOCKERS
- GR-ConvNet checkpoint download — needs ~2 GB, gated on user
- Penn lab robot access — outreach pending
```

Use these glyphs:
- ✓ done
- ⧖ in progress
- ⊘ blocked
- ○ not started
- ✗ failed / abandoned
