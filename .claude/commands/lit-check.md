---
description: Verify a novelty claim against the literature via lit-scout
argument-hint: "<claim in quotes>"
---

Dispatch the **lit-scout** subagent to verify the following claim:

**Claim**: $ARGUMENTS

The agent must:
1. First check `paper/related_work.md` to see if the claim's territory is already mapped.
2. If not adequately covered, do targeted web searches (arXiv, Google Scholar, IEEE Xplore).
3. Return a structured verdict in its standard format (VERIFIED-NOVEL / PARTIAL-PRIOR-ART / NOT-NOVEL with citations).
4. Append any new findings to `paper/related_work.md` under the relevant section, dated.

After the agent returns, summarize the verdict in one sentence so I can decide whether to keep, kill, or rephrase the claim.
