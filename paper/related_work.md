# Related Work — Working Scratchpad

Maintained by `lit-scout`. Append-only. Each entry is a citation + one-line method summary + verdict on how it relates to our paper. The LaTeX `related_work.tex` is derived from this; don't write prose here.

**Last sweep**: 2026-05-25 (initial seed from project setup).

---

## Section 1 — RGB-only / monocular grasp detection (direct competition)

This is the section we must defend hardest in review.

- **Le, Hoang, Le, Roy, "Attention-Based Grasp Detection With Monocular Depth Estimation," IEEE RA-L 2024.** Monocular depth → predicted point cloud → attention-based grasp net. **Closest learned competitor to our pipeline.** URL: https://ieeexplore.ieee.org/document/10521649
- **Atar, Chiu, Smith, et al., "OptiGrasp," arXiv:2409.19494, 2024 (UW + Amazon).** DINOv2 + Depth Anything for warehouse picking; 82.3% real-robot success. RGB-only, learned. URL: https://arxiv.org/abs/2409.19494
- **Xu et al., "RGBSQGrasp," arXiv:2503.02387, 2025.** Single-RGB → metric depth foundation model → superquadric primitive fitting → grasp. Bin-picking, not Cornell. URL: https://arxiv.org/abs/2503.02387
- **Guo, Huang, Yu (Rutgers), "MOMA: Monocular One-Shot Metric-Depth Alignment for RGB-Based Robot Grasping," arXiv:2506.17110, 2025.** Depth Anything + metric calibration → standard grasp pipelines. 82% real-robot. **Most similar high-level pipeline to ours (depth-foundation-model → grasp), but uses learned downstream.** URL: https://arxiv.org/abs/2506.17110
- **Li, "Modular Anti-noise Deep Learning Network for Robotic Grasp Detection Based on RGB Images," arXiv:2310.19223, 2023.** Pure-RGB grasp rectangle regression w/ segmentation. URL: https://arxiv.org/abs/2310.19223
- **Saxena, Driemeyer, Ng, IJRR 2008.** Foundational paper on learning grasp points from 2+ RGB images without 3D models. The original RGB-only grasping paper.

**Verdict for our paper**: Our position is "training-free + classical CV downstream of monocular depth." Le 2024 and MOMA 2025 use the same depth-from-RGB premise but pair it with learned grasp nets. **Differentiator: no training, no foundation models downstream.**

---

## Section 2 — Training-free / zero-shot grasp methods (most direct competitor)

- **Xiangli et al., "Foundation Model-Driven Grasping of Unknown Objects via Center of Gravity Estimation," arXiv:2507.19242, 2025.** CLIP + SAM + DIFT estimate CoG; grasps scored by CoG proximity to maximize GraspNet score. 76% real-robot, +49% over KGNv2. **This is the paper our work must position against most carefully.** URL: https://arxiv.org/abs/2507.19242
- **DISF, arXiv:2512.24550, 2025.** "Grasp pose alignment to object center of mass" in the title. Need to read; pulled in by similarity-search.
- **QuickGrasp, arXiv:2504.19716, 2025.** Antipodal heuristic citing CoM-axis argument for cylinders.

**Verdict for our paper**: Xiangli 2025 has the same headline claim (CoG-driven training-free grasping) but with heavy foundation models (CLIP+SAM+DIFT). **Our differentiator: monocular depth + classical CV only — order of magnitude cheaper compute.** Need EXP-08 head-to-head to substantiate.

---

## Section 3 — Heuristic / classical Cornell baselines

- **Jiang, Moseson, Saxena, ICRA 2011.** Introduced Cornell + the oriented grasp rectangle. SVM-rank over Laws masks / oriented edges / YCbCr. **Cornell Top-1: 60.5% image-wise / 58.3% object-wise.** This is THE baseline we beat to claim "best non-DL Cornell number." URL: https://www.semanticscholar.org/paper/3c104b0e182a5f514d3aebecc93629bbcf1434ac
- **Lenz, Lee, Saxena, IJRR 2015 (RSS 2013).** Early sparse-autoencoder grasp net. Cornell: 73.9% image-wise / 75.6% object-wise. URL: https://www.cs.cornell.edu/~asaxena/papers/lenz_lee_saxena_deep_learning_grasping_ijrr2014.pdf
- **Fischinger & Vincze, IJRR 2015.** HAF (Height Accumulated Features) for grasping in clutter. Geometric/heightmap heuristic but not benchmarked on Cornell. URL: https://journals.sagepub.com/doi/abs/10.1177/0278364915577105

**Verdict for our paper**: We aim to beat Jiang 2011 (60.5%) and match/beat Lenz 2015 (73.9-75.6%) on the *same metric* — that's the classical-baseline story.

---

## Section 4 — Contour-based / geometric grasp planning (geometric prior art)

- **Morales, Recatalá, Sanz, del Pobil, "Heuristic Vision-Based Computation of Planar Antipodal Grasps on Unknown Objects," ICRA 2001.** Local contour tangent + perpendicular ray + antipodal pair selection. **THE closest geometric prior art — same construction as ours.** Predates Cornell, so no benchmark number. Must cite prominently in related work and explain differentiator (depth-from-RGB front-end).
- **Sanz, Iñesta, del Pobil, "Planar Grasping Characterization Based on Curvature-Symmetry Fusion," Applied Intelligence 1999.** Local tangent/normal vectors as basis for parallel-jaw orientation on 2D silhouettes. URL: https://link.springer.com/article/10.1023/A:1008381314159
- **Lei et al., "Fast grasping of unknown objects using principal component analysis," AIP Advances 7(9), 2017.** PCA on contour for parallel-jaw alignment + concave/convex contour grasp regions. **Direct PCA-on-contour prior art for the orientation step.** URL: https://pubs.aip.org/aip/adv/article/7/9/095126/939733
- **Bone, Lambert, Edwards, ICRA 2008.** Ray-shooting on 3D silhouettes. URL: https://www.researchgate.net/publication/239412543
- **Richtsfeld & Vincze, ECCV-W 2008.** Grasping unknown objects from tabletop via silhouette analysis.
- **Popović et al., IROS 2008.** 2D-contour grasp methods extended to 3D via ray-shooting on silhouettes. URL: https://ieeexplore.ieee.org/document/4650632

**Verdict for our paper**: Every individual geometric ingredient is prior art (1999-2017). We must NOT claim geometric novelty. We CAN claim novelty in (a) putting this stack downstream of a pretrained monocular depth model, (b) the empirical finding on Cornell that CoG-dominance holds, (c) the demonstration as a training-free annotator with downstream value.

---

## Section 5 — Cornell SOTA (the upper-bound ceiling)

- **Kumra, Joshi, Sahin, GR-ConvNet v2, Sensors 2022.** **98.8% Cornell image-wise.** This is the practical ceiling. URL: https://www.mdpi.com/1424-8220/22/16/6208
- **Kumra et al., GR-ConvNet v1, IROS 2020.** 97.7%.
- **Morrison, Corke, Leitner, GG-CNN, RSS 2018.** **78% image-wise (depth-only input).** Closest in spirit to a "lightweight" deep baseline — single-stage, FCN-style. URL: https://arxiv.org/abs/1804.05172
- **TF-Grasp, 2022.** 97.99%.
- **Bilateral Cross-Modal Fusion Net, 2023.** 99.4% image-wise.
- **SimAM-GRCNN.** 98.8%.

**Verdict for our paper**: Cornell is saturated above 98% for RGB-D deep nets. **We do not compete in this league. We compete on the cost/training-data axis.**

---

## Section 6 — Center-of-mass as a grasp scoring criterion (theoretical grounding)

- **Bicchi & Kumar, "Robotic Grasping and Contact: A Review," ICRA 2000.** Canonical survey. Discusses CoM proximity as quality criterion. URL: https://www.centropiaggio.unipi.it/sites/default/files/surveys-icra00.pdf
- **Sahbani, El-Khoury, Bidaud, "An overview of 3D object grasp synthesis algorithms," Robotics and Autonomous Systems 60(3), 2012.** Reviews CoM-based heuristics among grasp-quality metrics. URL: https://hal.science/hal-00731127
- **Roa & Suarez, "Grasp quality measures: review and performance," Autonomous Robots 2015.** Explicitly: grasp quality improves as distance between object CoM and contact-polygon centroid is minimized. URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC4457357/
- **Chen et al., "Center-of-Mass-based Robust Grasp Pose Adaptation," arXiv:2205.01048, 2022.** Modern instantiation.
- **Wang et al., 2020 (arXiv:2006.00906).** CoM-driven grasp planning with tactile-visual sensors. URL: https://arxiv.org/abs/2006.00906
- **Kanoulas et al., MAG (Multi-Aspect Grasp) index, 2011.** Defines CoM as optimal grasp point maximizing MAG. URL: https://www.sciencedirect.com/science/article/pii/S1026309811000186

**Verdict for our paper**: CoM as grasp criterion is canonical, traceable to Bicchi & Kumar 2000. **We do not claim to discover it — we claim to empirically isolate its dominance on Cornell via a 432-config grid search.** Frame as: "the literature has long argued for CoM-proximity; our grid search quantifies just how dominant it is when other heuristic features are available."

---

## Section 7 — Antipodal grasp theory (force-closure grounding)

- **Nguyen, "Constructing Force-Closure Grasps," IJRR 7(3), 1988.** Foundational definition of force-closure conditions on planar shapes from contour normals.
- **Chen & Burdick, "Finding antipodal point grasps on irregularly shaped objects," IEEE T-RA 1993.** Antipodal grasp computation on arbitrary 2D shapes.

**Verdict for our paper**: Our ray-cast across the silhouette computes antipodal pairs in the planar frictionless sense (Nguyen 1988). Cite for theoretical grounding.

---

## Section 8 — Datasets

- **Cornell Grasping Dataset** (Jiang et al., 2011) — 855 RGB-D images, ~240 objects. Our primary benchmark.
- **Jacquard** (Depierre et al., 2018) — 11k synthetic objects, ~50k images. EXP-06 generalization target.
- **GraspNet-1Billion** (Fang et al., CVPR 2020) — large-scale, RA-L extension only.

---

## Sweep log

| Date | Trigger | Notes |
|---|---|---|
| 2026-05-25 | initial-seed | Pre-loaded from project's prior literature-review agents. Coverage: 30 papers across 8 themes. Next sweep due 2026-06-15 or before related_work.tex draft. |
