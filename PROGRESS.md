# Project Progress Dashboard

**Last updated**: 2026-05-27
**Target venues**: ICRA / IROS 2026 workshop (primary) → RA-L 2026 (extension)
**Framing**: Training-free monocular-RGB grasp annotator (NOT a novel detector)

---

## Experiments

| ID | Name | Status | Headline | Notes |
|---|---|---|---|---|
| EXP-01 | cog_only_baseline | ✓ done | GT 8.70% / Mono 16.95% Top-1 | Trivial baseline well below threshold → CoG-critique angle dead; annotator framing strengthened. See `experiments/EXP-01_*/notes.md`. |
| EXP-02 | full_cornell_eval | ✓ done | 16.72% ± 1.05 Top-1 (re-run after fix) | Output-convention bug fixed 2026-05-27 (`height` was hardcoded 20 px; now measured as perpendicular contour extent; angle wrapped to [-π/2, π/2]). Pre-fix: 8.70%. Now ≈ tied with EXP-01 mono CoG (16.95%). Mean angle error still 65° — genuine algorithmic mis-orientation on long-thin objects, not a convention issue. See `experiments/EXP-02_*/notes.md` § "Fix applied 2026-05-27". |
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
| src/methods/heuristic/ (refactor from Streamlit) | ✓ done |
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
- 2026-05-25: eval-harness built `src/eval/cornell.py` (13/13 property tests passing) — canonical Cornell evaluator.
- 2026-05-25: lit-scout corrected canon: 885 is the canonical Cornell image count (not 855). Confirmed against Jiang 2011, Redmon 2015, Morrison 2018, Pinto 2016.
- 2026-05-26: Python 3.11 venv with full ML stack (torch 2.12 CPU, transformers 5.9, shapely 2.1).
- 2026-05-26: src/methods/heuristic/ refactored from Streamlit (647 lines, smoke test passes, ~2.1s/call). Three flags surfaced — most notably cog_quality normalized by image diagonal (must test in EXP-04).
- 2026-05-26: src/methods/cog_baseline/ built (124 lines).
- 2026-05-27: **EXP-01 complete**. CoG-only baseline: GT depth 8.70%, mono depth 16.95% Top-1. CoG-critique paper framing dead; training-free annotator framing strengthened. Monocular depth beats GT depth as a mask source (surprise finding worth its own paragraph in discussion).
- 2026-05-27: **EXP-02 complete**. Heuristic full pipeline on full Cornell image-wise 5-fold: 8.70% ± 2.33 Top-1, 14.01% Top-5, IoU 0.38, ang err 72°. Number tanked (< 60% band); spec says "method overfit to crop; consider falling back to CoG-only as the headline." Pipeline performs _below_ the EXP-01 mono CoG-only baseline (16.95%), driven by an angle/width output-convention issue (heuristic outputs `width = ray length`, no perpendicular check). Wallclock 50 s on 100% EXP-01 depth cache hits.
- 2026-05-27: **EXP-02 re-run after output-convention fix.** Localised the bug to two issues in `src/methods/heuristic/detect.py`: `height` was hardcoded to 20 px instead of measuring the contour's perpendicular extent, and `angle_rad` was raw `atan2` rather than wrapped to [-π/2, π/2] like the Cornell loader does. Fixed both (new helper `_perpendicular_extent` uses 5–95th percentile spread of contour projected on the perpendicular axis, lower-bounded at 30 px). Added a pytest covering elongated-object aspect and angle range. Re-ran: 8.70% → **16.72% ± 1.05 Top-1**, Top-5 14.01% → 20.34%, IoU 0.38 → 0.29 (down because the old sliver rectangles had inflated IoU), angle err 72° → 65°. Now ≈ tied with EXP-01 mono CoG. Remaining 65° mean angle error is algorithmic (heuristic mis-orients some long-thin objects), not a convention bug.
