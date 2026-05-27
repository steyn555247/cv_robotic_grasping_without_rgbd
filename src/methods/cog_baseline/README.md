# `cog_baseline` — Centre-of-gravity-only trivial baseline

This method ignores all image content except a binary foreground mask. It
computes the mask centroid (CoG), runs PCA on the mask pixel coordinates to
recover the object's principal axes, and emits a single oriented grasp
rectangle centred at the CoG with orientation along the **minor** principal
axis (the gripper closes across the narrow direction of the object). The
grasp width is a fixed fraction (default 0.6) of the minor-axis extent and
the height is a fixed 20 pixels (Cornell convention).

There is no ranking — Top-1 = Top-5 for this method (the same rectangle is
returned five times).

The foreground mask can be obtained either from the ground-truth Cornell
depth channel (informational upper bound, **not RGB-only**) or from a frozen
monocular depth estimator (`Depth-Anything-V2-Small`) applied to RGB only.
The latter is the headline number reported in the paper.

Spec: `experiments/EXP-01_cog_only_baseline/spec.md`.
