# Project Progress Dashboard

**Last updated**: 2026-05-25
**Target venues**: ICRA / IROS 2026 workshop (primary) → RA-L 2026 (extension)
**Framing**: Training-free monocular-RGB grasp annotator (NOT a novel detector)

---

## Experiments

| ID | Name | Status | Headline | Notes |
|---|---|---|---|---|
| EXP-01 | cog_only_baseline | ○ pending | — | Gating experiment. Trivial 2D-centroid baseline on full Cornell. |
| EXP-02 | full_cornell_eval | ○ pending | — | Standardized eval of current heuristic (no manual crop). |
| EXP-03 | grconvnet_repro | ○ pending | — | Public GR-ConvNet checkpoint on same Cornell split. |
| EXP-04 | scoring_ablation | ○ pending | — | Remove each scoring term; confirm CoG dominance. |
| EXP-05 | throughput | ○ pending | — | Images/sec on CPU + GPU. The cost story. |
| EXP-06 | jacquard_generalization | ○ pending | — | Does CoG dominance transfer beyond Cornell? |
| EXP-07 | autolabel_training_value | ○ pending | — | Train GR-ConvNet on our auto-labels vs. human labels. **Headline.** |
| EXP-08 | xiangli_comparison | ○ pending | — | Head-to-head with the direct training-free competitor. |
| EXP-09 | real_robot_pickup | ○ pending | — | Optional, RA-L only. 10-20 object pickup study. |

**Critical path for workshop**: 01 → 02 → 03 → 04 → 07
**Decision branch**: After 01 + 02 complete, choose between "CoG-critique" framing and "annotator" framing.

---

## Paper sections

| Section | Status | Word count target |
|---|---|---|
| abstract.tex | ○ not started | 150 |
| intro.tex | ○ not started | 600 |
| related_work.tex | ○ not started (scratchpad pre-seeded) | 600 |
| method.tex | ○ not started | 700 |
| experiments.tex | ○ not started | 900 |
| discussion.tex | ○ not started | 400 |
| conclusion.tex | ○ not started | 150 |

---

## Infrastructure readiness

| Component | Status |
|---|---|
| .claude/agents/ (7 subagents) | ✓ done |
| .claude/commands/ (8 slash commands) | ✓ done |
| .claude/settings.json | ✓ done |
| src/eval/cornell.py | ✓ done |
| src/data/cornell_loader.py | ✓ done |
| src/methods/heuristic/ (refactor from Streamlit) | ○ not started — needed before EXP-02 |
| src/data/splits/cornell.json | ✓ done |
| paper/refs.bib | ○ not started |
| paper/main.tex (IEEE template) | ○ not started |

---

## Data readiness

| Dataset | Local | Splits frozen | Notes |
|---|---|---|---|
| Cornell | ✓ at `cornell comparisson/cornell_dataset/` (12 GB) | ✓ frozen (`src/data/splits/cornell.json`, seed=42) | **885 images = canonical count** (verified vs. Jiang 2011, Redmon 2015, Morrison 2018, Pinto 2016 — see `paper/related_work.md` § "Dataset audit"). 950-999 gap is a folder boundary, not withheld. Image-wise 5-fold ready; object-wise curation deferred to RA-L extension (no public `backgroundMapping.txt`). |
| Jacquard | ○ | ○ | ~75 GB; fetch only when EXP-06 dispatched |
| GraspNet-1B | ○ | ○ | Optional, very large |

---

## Outreach (parallel to experimentation)

- [ ] Email Penn robotics lab PIs about EXP-09 robot access
- [ ] Identify 2-3 candidate labs (GRASP Lab, ModLab, Kod*lab)
- [ ] Draft 1-paragraph collaboration ask

---

## Blockers

- (none yet)

---

## Recent activity

- 2026-05-25: Project repo created on GitHub. Initial commit.
- 2026-05-25: Research infrastructure laid down (.claude/agents, commands, settings).
- 2026-05-25: PROGRESS, BACKLOG, related_work scratchpad initialized.
- 2026-05-25: dataset-ops built `src/data/cornell_loader.py` + immutable `src/data/splits/cornell.json` (885 images, seed=42 5-fold). Cornell splits frozen.
