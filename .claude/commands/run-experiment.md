---
description: Dispatch experiment-runner on a signed-off spec.md
argument-hint: <experiment-folder-name>
---

Dispatch the **experiment-runner** subagent to execute `experiments/$ARGUMENTS/spec.md`.

Before dispatching, verify:
1. `experiments/$ARGUMENTS/spec.md` exists. If not, suggest `/spec-experiment` first.
2. The spec's "Blocked by" experiments are completed (their `results.json` exists).
3. The user has had a chance to review the spec.

Then call the Agent tool with `subagent_type: experiment-runner` and a prompt that:
- Names the spec file path.
- Reminds the agent to follow its standing rules (canonical eval, seeds, results.json schema, PROGRESS.md update).
- Asks for the 4-line summary back when done.

After the agent returns, append a one-line entry to `PROGRESS.md` under the relevant section.
