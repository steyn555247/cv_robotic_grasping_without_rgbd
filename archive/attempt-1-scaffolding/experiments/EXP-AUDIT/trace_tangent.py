"""Trace the tangent -> perpendicular -> ray chain for the top candidate of one sample.

Goal: determine WHERE the ~90-degree error enters. Prints, for the winning
candidate of pcd0411:
  - the local contour tangent direction (deg)
  - the perpendicular direction the code returns (deg)
  - the ray endpoints and the final ray angle (deg)
  - the nearest GT angle (deg)
"""
from __future__ import annotations
import math
import numpy as np

from src.data.cornell_loader import CornellDataset
from src.methods.heuristic import detect as D
from src.methods.heuristic.config import HeuristicConfig
from src.methods.heuristic.depth import DepthEstimator

SID = "pcd0411"

# locate sample
for fold in range(5):
    ds = CornellDataset(split="image-wise", fold=fold, partition="test")
    hit = None
    for i in range(len(ds)):
        if ds[i]["sample_id"] == SID:
            hit = ds[i]; break
    if hit:
        break
s = hit
img = s["image"]
H, W = img.shape[:2]

# crop exactly like EXP-02b
CX, CY, CW, CH = 100, 150, 400, 300
img_c = img[CY:CY+CH, CX:CX+CW]

de = DepthEstimator()
depth = de(img_c)
cfg = HeuristicConfig()

# Reproduce the internal pipeline up to the winning candidate.
mask, contour, grad_x, grad_y, edges = D._build_object_mask(img_c, depth, cfg.depth_percentile)
print("mask nonzero:", int(np.count_nonzero(mask)), "contour pts:", 0 if contour is None else len(contour))

cog_x, cog_y = D._mask_centroid(mask, img_c.shape[1], img_c.shape[0])
cand_pts = D._sample_contour_points(contour, cfg.num_contour_samples)
max_dist_scene = float(np.sqrt(img_c.shape[1]**2 + img_c.shape[0]**2))
prelim = D._score_candidates(cand_pts, edges, np.sqrt(grad_x**2+grad_y**2)/(np.sqrt(grad_x**2+grad_y**2).max()+1e-8),
                             cog_x, cog_y, max_dist_scene, img_c.shape[1], img_c.shape[0],
                             cfg.w_edge, cfg.w_depth, cfg.w_cog)
prelim.sort(key=lambda c: c["combined_quality"], reverse=True)

def ang(dx, dy):
    a = math.degrees(math.atan2(dy, dx))
    while a > 90: a -= 180
    while a < -90: a += 180
    return a

print(f"\nTracing top 3 candidates for {SID} (crop frame):")
for cand in prelim[:3]:
    px, py = cand["point"]
    # tangent + perpendicular as the code computes it
    perp_x, perp_y = D._contour_tangent_80px(contour, px, py)
    # the function returns the PERPENDICULAR (it already rotated). Recover tangent:
    tan_x, tan_y = perp_y, -perp_x  # inverse of (-ty, tx)
    # ray
    e1x, e1y, e2x, e2y, L = D._cast_ray_direct_line(
        mask, (px, py), grad_x, grad_y, contour, cfg.gradient_source,
        cfg.ray_max_dist, cfg.ray_gap_tolerance)
    ray_ang = ang(e1x - e2x, e1y - e2y)
    print(f"  cand=({px},{py})")
    print(f"    tangent dir      = {ang(tan_x, tan_y):+7.2f} deg")
    print(f"    perpendicular dir= {ang(perp_x, perp_y):+7.2f} deg  (code returns this as ray dir)")
    print(f"    ray endpoints    = ({e1x},{e1y}) -> ({e2x},{e2y})  len={L:.1f}")
    print(f"    ray angle        = {ray_ang:+7.2f} deg")

print("\nGT angles (loader):")
for g in s["grasps_gt"][:5]:
    print(f"    {math.degrees(g.angle_rad):+7.2f} deg  (w={g.width:.0f} h={g.height:.0f})")
