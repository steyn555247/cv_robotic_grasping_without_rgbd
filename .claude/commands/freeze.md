---
description: Snapshot the current state to a git tag before a risky change
argument-hint: <tag-suffix-optional>
---

Create a git tag of the form `freeze/$(date +%Y%m%d-%H%M)-$ARGUMENTS` (drop the suffix if no argument).

Run, in order:
1. `git status` — confirm clean or near-clean tree
2. If there are uncommitted changes, ask the user whether to commit them first or stash
3. `git tag freeze/YYYYMMDD-HHMM-<suffix>` 
4. `git push --tags`
5. Report the tag name to the user

The intent: before any agent does a risky refactor in `src/`, the user runs `/freeze refactor-eval` (or similar) so we have a known-good point to revert to.
