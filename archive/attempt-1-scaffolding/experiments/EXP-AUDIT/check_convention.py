"""Decide once and for all what Cornell's angle convention is.

For pcd0500 (a horizontal thin object — wait, let's check):
- Read the raw cpos.txt
- Print the 4 corners of the first grasp
- Compute both interpretations:
    A) angle = direction of SHORTER side  (current loader convention)
    B) angle = direction of LONGER side   (alternative)
- Sanity check by also reading the image and saving an overlay PNG
"""
from __future__ import annotations
import os
import math
import numpy as np
import cv2

DATA_ROOT = r"C:\Users\steyn\OneDrive\Desktop\CIS5810 Final Project\cornell comparisson\cornell_dataset"

def find_pcd(sid: str) -> tuple[str, str]:
    """Return (rgb_path, cpos_path) for a given pcd id."""
    for folder in range(1, 11):
        sub = os.path.join(DATA_ROOT, f"{folder:02d}")
        rgb = os.path.join(sub, f"{sid}r.png")
        if os.path.exists(rgb):
            return rgb, os.path.join(sub, f"{sid}cpos.txt")
    raise FileNotFoundError(sid)

def load_corners(cpos: str) -> list[np.ndarray]:
    """Parse cpos.txt -> list of (4,2) arrays."""
    rects = []
    with open(cpos) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    for i in range(0, len(lines), 4):
        if i + 4 > len(lines): break
        pts = []
        ok = True
        for line in lines[i:i+4]:
            parts = line.split()
            if len(parts) != 2: ok = False; break
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                ok = False; break
            if not (math.isfinite(x) and math.isfinite(y)): ok = False; break
            pts.append([x, y])
        if ok: rects.append(np.array(pts))
    return rects

def analyse(sid: str) -> None:
    rgb_path, cpos_path = find_pcd(sid)
    print(f"\n=== {sid} ===")
    print(f"rgb: {rgb_path}")
    rects = load_corners(cpos_path)
    print(f"{len(rects)} GT rectangles in cpos.txt")

    img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    print(f"image: {H}x{W}")

    out = img.copy()

    for idx, pts in enumerate(rects[:3]):
        print(f"\nGT rect {idx}:")
        for p in pts:
            print(f"  corner: ({p[0]:7.2f}, {p[1]:7.2f})")

        side_a = pts[1] - pts[0]
        side_b = pts[2] - pts[1]
        len_a = np.linalg.norm(side_a)
        len_b = np.linalg.norm(side_b)
        print(f"  |side_a (pts[0]->pts[1])| = {len_a:.2f}")
        print(f"  |side_b (pts[1]->pts[2])| = {len_b:.2f}")

        ang_a = math.degrees(math.atan2(side_a[1], side_a[0]))
        ang_b = math.degrees(math.atan2(side_b[1], side_b[0]))
        # wrap to [-90, 90]
        while ang_a > 90: ang_a -= 180
        while ang_a < -90: ang_a += 180
        while ang_b > 90: ang_b -= 180
        while ang_b < -90: ang_b += 180

        if len_a <= len_b:
            shorter, longer = "a", "b"
            angle_shorter, angle_longer = ang_a, ang_b
        else:
            shorter, longer = "b", "a"
            angle_shorter, angle_longer = ang_b, ang_a

        print(f"  shorter side = {shorter}, longer = {longer}")
        print(f"  CONVENTION A (current loader): angle = shorter-side direction = {angle_shorter:+.2f} deg")
        print(f"  CONVENTION B (alternative)   : angle = longer-side  direction = {angle_longer:+.2f} deg")

        # Draw it
        poly = pts.astype(np.int32)
        cv2.polylines(out, [poly], True, (0, 255, 0), 2)
        # Mark side_a with a colour (red) -> "pts[0]->pts[1]"
        cv2.line(out, tuple(poly[0]), tuple(poly[1]), (255, 0, 0), 2)
        # Mark side_b with a colour (blue) -> "pts[1]->pts[2]"
        cv2.line(out, tuple(poly[1]), tuple(poly[2]), (0, 0, 255), 2)

    out_path = os.path.join(os.path.dirname(__file__), f"convention_{sid}.png")
    cv2.imwrite(out_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"\noverlay saved to: {out_path}")
    print("  green polygon = full rectangle")
    print("  red   line    = pts[0]->pts[1] (side_a)")
    print("  blue  line    = pts[1]->pts[2] (side_b)")

if __name__ == "__main__":
    import sys
    sids = sys.argv[1:] if len(sys.argv) > 1 else ["pcd0500", "pcd0110", "pcd0411"]
    for sid in sids:
        analyse(sid)
