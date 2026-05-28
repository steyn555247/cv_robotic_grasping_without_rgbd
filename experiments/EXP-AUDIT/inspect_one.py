"""Quick inspector: predicted vs GT for a single sample."""
from __future__ import annotations
import json, sys
import numpy as np
from src.data.cornell_loader import CornellDataset

SID = sys.argv[1] if len(sys.argv) > 1 else "pcd0500"

# Find the sample
for fold in range(5):
    ds = CornellDataset(split="image-wise", fold=fold, partition="test")
    for i in range(len(ds)):
        s = ds[i]
        if s["sample_id"] == SID:
            break
    else:
        continue
    break
else:
    print(f"{SID} not found in any test fold"); sys.exit(1)

print(f"=== {SID} (fold {fold}) ===")
print(f"image shape: {s['image'].shape}")
print()
print("GT grasps (first 5):")
for g in s["grasps_gt"][:5]:
    print(f"  x={g.x:7.1f}  y={g.y:7.1f}  angle={np.degrees(g.angle_rad):+7.2f}deg  w={g.width:6.1f}  h={g.height:6.1f}  aspect={min(g.width,g.height)/max(g.width,g.height):.2f}")

print()
print("Predictions (top 5):")
preds_path = f"experiments/EXP-02b_cropped_cornell_eval/predictions/fold-{fold}/{SID}.json"
preds = json.load(open(preds_path))["predictions"]
for p in preds[:5]:
    print(f"  x={p['x']:7.1f}  y={p['y']:7.1f}  angle={np.degrees(p['angle_rad']):+7.2f}deg  w={p['width']:6.1f}  h={p['height']:6.1f}")

print()
print("Closest-GT angle diff (with 180-symmetry):")
for p in preds[:1]:
    best = 999.0
    for g in s["grasps_gt"]:
        d = abs(np.degrees(p["angle_rad"] - g.angle_rad)) % 180.0
        d = min(d, 180.0 - d)
        if d < best: best = d
    print(f"  top-1 prediction is {best:.2f}deg off the nearest GT angle")
