"""Probe the legacy vs canonical angle_diff to demonstrate wraparound bug."""
import numpy as np


def legacy_angle_diff(a1: float, a2: float) -> float:
    """Copy of GraspEvaluator.angle_diff from notebook cell 24."""
    diff = abs(a1 - a2)
    diff = min(diff, np.pi - diff, abs(diff - np.pi))
    return np.degrees(diff)


def canonical_angle_diff(a_rad: float, b_rad: float) -> float:
    """Copy of _angle_error_deg from src/eval/cornell.py."""
    a_deg = float(np.degrees(a_rad))
    b_deg = float(np.degrees(b_rad))
    diff = abs(a_deg - b_deg) % 180.0
    return float(min(diff, 180.0 - diff))


def main() -> None:
    tests = [
        (0.0, np.pi / 2),
        (0.0, -np.pi / 2),
        (np.pi, 0.0),
        (np.pi - 0.1, -np.pi + 0.1),
        (np.pi * 0.9, -np.pi * 0.9),
        (3.0, -3.0),
        (2.5, -2.5),
        (1.0, -2.5),
        (0.0, 2.0),
        (0.0, -2.0),
        (np.pi, -np.pi),
    ]
    print("a1, a2, legacy_deg, canonical_deg")
    for a1, a2 in tests:
        L = legacy_angle_diff(a1, a2)
        C = canonical_angle_diff(a1, a2)
        flag = "  <-- diff" if abs(L - C) > 0.5 else ""
        print(f"{a1:+7.3f}, {a2:+7.3f}, {L:+9.3f}, {C:+9.3f}{flag}")


if __name__ == "__main__":
    main()
