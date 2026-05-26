---
description: Draft or revise a paper section
argument-hint: <section-name: abstract|intro|related_work|method|experiments|discussion|conclusion>
---

Dispatch the **paper-writer** subagent to draft / revise `paper/sections/$ARGUMENTS.tex`.

Before dispatch, verify the prerequisites for this section exist:

- **abstract / intro / conclusion** — needs RESULTS.md to have at least the headline number filled in.
- **related_work** — needs `paper/related_work.md` to have at least 8 entries with verdicts (lit-scout should have run at least once).
- **method** — needs the heuristic method extracted into `src/methods/heuristic/` (not just in the Streamlit script).
- **experiments** — needs at least EXP-01 and EXP-02 completed with results.json.
- **discussion** — needs the ablation (EXP-04) results.

If a prereq is missing, abort and tell the user what to do first.

After the agent returns, ask the user if they want to immediately follow with `/red-team` on the new section.
