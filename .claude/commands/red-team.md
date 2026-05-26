---
description: Adversarial peer review of the current draft
---

Dispatch the **red-team-reviewer** subagent to review the current state of `paper/main.tex` + `paper/sections/*.tex` + `RESULTS.md`.

The agent will:
1. Read everything.
2. Write a hostile review to `paper/reviews/<YYYY-MM-DD>_red_team.md`.
3. Return the verdict + top 3 weaknesses to me.

After the review comes back, present the top 3 weaknesses to the user and ask which ones they want to address now (creating new experiments via `/spec-experiment` or section rewrites via `/draft-section`).
