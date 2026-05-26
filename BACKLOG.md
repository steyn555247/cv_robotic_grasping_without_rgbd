# Experiment Backlog

Prioritized for **ICRA / IROS 2026 workshop submission (March 2026)** → **RA-L extension (mid-2026)**.

Workshop-blocking experiments marked **W**. RA-L-only marked **R**.

---

## Critical path (workshop)

### **W** EXP-01: CoG-only trivial baseline
- **Effort**: 1 day
- **Hypothesis**: A 2D centroid grasp (no contour, no depth, no edges) achieves ≥65% Top-1 on Cornell — close enough to the current 75.89% to be the headline finding.
- **Decision**: If Top-1 ≥ 70%, paper pivots to "Cornell-is-easy" critique; if <60%, paper stays on "training-free annotator" angle.
- **Status**: spec needed

### **W** EXP-02: Standardized Cornell eval of current heuristic
- **Effort**: 1 day (mostly refactor of Streamlit code into a library function)
- **Hypothesis**: The 75.89% from the cropped-subset grid search generalizes to full Cornell with the canonical 5-fold image-wise split.
- **Decision**: If full-Cornell Top-1 drops below 70%, reconsider the headline.
- **Status**: spec needed

### **W** EXP-03: GR-ConvNet head-to-head
- **Effort**: 2 days
- **Hypothesis**: GR-ConvNet v2 reaches ~98% on our split — required as the upper-bound comparator.
- **Decision**: Establishes the gap our method must accept or close.
- **Status**: spec needed; pre-req: download checkpoint (~700 MB)

### **W** EXP-04: Scoring-term ablation
- **Effort**: 2 days
- **Hypothesis**: Removing CoG term drops Top-1 by ≥15pp; removing edge or depth terms changes Top-1 by <2pp.
- **Decision**: Confirms (or kills) the CoG-dominance finding as a structural claim, not a tuning artifact.
- **Status**: spec needed; pre-req: EXP-02

### **W** EXP-07: Auto-label training value (HEADLINE)
- **Effort**: 5 days with GPU
- **Hypothesis**: GR-ConvNet trained on our auto-labels reaches within 5pp of GR-ConvNet trained on human Cornell labels.
- **Decision**: Determines whether the "annotator" framing has empirical backing. **This is the single most important experiment.**
- **Status**: spec needed; pre-req: EXP-02, EXP-03

---

## Supporting experiments

### **W** EXP-05: Throughput & latency
- **Effort**: 1 day
- **Outputs**: images/sec on CPU + GPU, vs. Xiangli 2025, vs. human annotation time. Powers the "cheap" claim.
- **Status**: spec needed

### EXP-06: Jacquard generalization
- **Effort**: 3 days (most of it Jacquard download + loader)
- **Hypothesis**: CoG dominance either generalizes (strong claim) or collapses (interesting critique). Either way publishable.
- **Status**: spec needed; pre-req: dataset-ops downloads Jacquard

### EXP-08: Xiangli 2025 head-to-head
- **Effort**: 3-5 days (depends on code availability)
- **Outputs**: Direct comparison with the closest training-free competitor.
- **Status**: spec needed; gated on Xiangli code release

---

## RA-L extension

### **R** EXP-09: Real-robot pickup study
- **Effort**: 1-2 weeks lab access
- **Outputs**: Success rate on 10-20 household objects with a real parallel-jaw gripper.
- **Status**: blocked on lab outreach
- **Pre-reqs**: A Penn lab agrees to host this. Likely candidates: GRASP Lab, ModLab.

### **R** Build curated Cornell object-wise CV (`cornell_v2.json`)
- **Effort**: 4-6 hours (depends on whether external mapping is sourced or built from scratch)
- **Why deferred**: No public `pcd_id → object_id` mapping exists; Lenz/Redmon/Kumra all built private partitions. Workshop reviewers usually accept image-wise-only with a footnote. RA-L reviewers will flag missing object-wise — so this becomes blocking for the RA-L extension, not the workshop.
- **Approach**: (a) ask a Cornell paper author for their partition, (b) cluster images by depth-image background similarity, or (c) hand-curate from visual inspection. Best: combination of (b) auto-cluster + (c) human spot-check.
- **Outputs**: `src/data/splits/cornell_v2.json` with proper ~240-object partition. `cornell.json` v1 stays frozen as discovery artifact.

---

## Out of scope (do not pursue)

- Beating GR-ConvNet on Cornell Top-1 — not the contribution.
- Tactile feedback, multi-finger grippers, 6-DOF grasps — orthogonal.
- Training our own depth model — uses an off-the-shelf one.
- Cluttered scenes — single isolated object is the Cornell setting.

---

## Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-25 | Workshop-first, RA-L second | User priority; matches expected lab access timeline |
| 2026-05-25 | Same repo, paper/ + experiments/ alongside course folders | User preference; lower setup overhead |
| 2026-05-25 | Framing = training-free annotator, not novel detector | Lit search confirms detector framing is dead; annotator angle is open |
| 2026-05-25 | Cornell = 885 images (canonical, verified) | lit-scout audit vs. Jiang 2011, Redmon 2015, Morrison 2018, Pinto 2016 — unanimous |
| 2026-05-25 | Object-wise CV deferred to RA-L | No public mapping; workshop uses image-wise only (matches most prior papers) |
| 2026-05-25 | Python 3.11 (not 3.12) for research venv | 3.12 not on PATH; 3.11.9 is available and ML-stable |
