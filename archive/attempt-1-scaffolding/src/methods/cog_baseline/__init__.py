"""CoG-only trivial baseline (EXP-01).

A grasp predictor that ignores all image content except a foreground mask.
See ``detect.py`` for the implementation and ``README.md`` for the rationale.
"""

from src.methods.cog_baseline.detect import detect_cog_grasp

__all__ = ["detect_cog_grasp"]
