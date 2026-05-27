"""Hyperparameter configuration for the heuristic grasp detector.

All fields default to the grid-search-winning values determined during the
prior course-project sweep (see notes in the project archive under
``Heuristics approach/``). The detection pipeline reads every knob from this
dataclass — no magic numbers live in ``detect.py``.

The defaults below correspond to the ``Direct Line with CoG Boost`` ray
algorithm with ``Contour Direction (80px avg)`` as the gradient source,
which was the winning combination from the grid search.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HeuristicConfig:
    """Hyperparameters for ``detect_grasp``.

    The weight triple ``(w_edge, w_depth, w_cog)`` controls the candidate
    ranking score (stage 1). The remaining knobs control mask construction,
    candidate sampling, ray-cast filtering, and output count (stages 2-4).

    Defaults are the grid-search winners from the course-project sweep.
    """

    # --- candidate scoring weights (stage 1) ---
    # Stage-1 ranks contour samples by w_edge*edge + w_depth*depth + w_cog*cog.
    # CoG term dominates by ~3 orders of magnitude in the winning config; that
    # is intentional and documented in the project's related-work notes.
    w_edge: float = 0.001
    w_depth: float = 0.001
    w_cog: float = 0.998

    # --- mask construction ---
    # Percentile of the depth map used as the "object is closer than this"
    # threshold. 30 means: pixels with depth >= 30th percentile become the ROI.
    depth_percentile: int = 30

    # --- ray casting ---
    # Algorithm: "Direct Line with CoG Boost" (grid-search winner) or
    # "Through CoG" (alternative). The Streamlit UI exposed both; the
    # winning grid-search configuration uses "Direct Line with CoG Boost".
    ray_algorithm: str = "Direct Line with CoG Boost"

    # When ray_algorithm == "Direct Line with CoG Boost", the rank score is
    # ``line_length - cog_boost * proximity * 500``. Larger cog_boost biases
    # ranking toward grasps whose centre is close to the CoG.
    cog_boost: float = 3.75

    # Direction-finding mode for "Direct Line with CoG Boost". Kept as a
    # switchable option even though the grid-search winner is the 80px PCA
    # tangent method.
    # One of:
    #   "Contour Direction (80px avg)"  -- grid-search winner
    #   "Depth Gradients"
    #   "Image Edges"
    #   "Radial from Center"
    gradient_source: str = "Contour Direction (80px avg)"

    # Maximum pixels traced per ray (both directions cast from grasp point).
    ray_max_dist: int = 500

    # Allow this many consecutive background pixels inside the mask before
    # the ray gives up (noise tolerance).
    ray_gap_tolerance: int = 10

    # --- grasp length filtering ---
    # Grasps with measured line length outside (min_grasp_length,
    # max_grasp_length) are rejected. Defaults are deliberately permissive:
    # the grid-search winner essentially disables the lower bound.
    min_grasp_length: float = 1.0
    max_grasp_length: float = 1000.0

    # --- candidate sampling ---
    # Number of contour samples to evaluate at stage 1 (linearly spaced
    # around the contour).
    num_contour_samples: int = 100

    # Stage 2 keeps ``num_output_grasps * candidate_multiplier`` of the
    # highest-scoring candidates for ray casting.
    candidate_multiplier: int = 100

    # --- output ---
    # Number of grasps returned (best first).
    num_output_grasps: int = 5

    # --- depth model ---
    # HuggingFace model id used by ``DepthEstimator`` when not overridden.
    depth_model_name: str = "depth-anything/Depth-Anything-V2-Small-hf"
