"""Debug script: compare EXP-02 predictions against GT for 5 sample images.

Investigates the convention mismatch suspected from EXP-02 results:
- IoU = 0.38 (rectangles ~ overlap)
- Angle err = 72° (orientation off ~ 90°)

For each sample, print GT[0], top-1 pred, computed angle errors, and IoU.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.cornell_loader import CornellDataset  # noqa: E402
from src.eval.cornell import GraspRect, _angle_error_deg, _oriented_iou  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
PRED_ROOT = EXP_DIR / "predictions"

SAMPLES = [100, 200, 500, 700, 1000]


def _find_pred_file(pcd_id: int) -> Path | None:
    sid = f"pcd{pcd_id:04d}"
    for fold in range(5):
        p = PRED_ROOT / f"fold-{fold}" / f"{sid}.json"
        if p.exists():
            return p
    return None


def _load_pred(path: Path) -> list[GraspRect]:
    with path.open() as fh:
        data = json.load(fh)
    return [
        GraspRect(
            x=p["x"], y=p["y"], angle_rad=p["angle_rad"], width=p["width"], height=p["height"]
        )
        for p in data["predictions"]
    ]


def main() -> None:
    ds = CornellDataset(split="all")
    id_to_idx = {f"pcd{pid:04d}": i for i, pid in enumerate(ds.ids)}

    for pcd_id in SAMPLES:
        sid = f"pcd{pcd_id:04d}"
        if sid not in id_to_idx:
            print(f"== {sid}: NOT IN DATASET ==")
            continue
        s = ds[id_to_idx[sid]]
        gts: list[GraspRect] = s["grasps_gt"]
        pred_path = _find_pred_file(pcd_id)
        if not pred_path:
            print(f"== {sid}: NO PREDICTION FOUND ==")
            continue
        preds = _load_pred(pred_path)

        print(f"== {sid} ==")
        if gts:
            gt = gts[0]
            print(
                f"  GT[0]:    x={gt.x:6.1f} y={gt.y:6.1f} "
                f"ang={math.degrees(gt.angle_rad):7.2f}deg "
                f"w={gt.width:6.1f} h={gt.height:6.1f}  "
                f"aspect(h/w)={gt.height/max(gt.width,1e-6):4.2f}"
            )
            # Also show angle spread across GTs
            gt_angles = [math.degrees(g.angle_rad) for g in gts]
            print(
                f"  All {len(gts)} GT angles (deg): "
                f"[{', '.join(f'{a:.1f}' for a in gt_angles[:8])}{'...' if len(gt_angles)>8 else ''}]"
            )
            gt_w_list = [g.width for g in gts]
            gt_h_list = [g.height for g in gts]
            print(
                f"  GT widths: min={min(gt_w_list):.1f} max={max(gt_w_list):.1f} "
                f"mean={sum(gt_w_list)/len(gt_w_list):.1f}"
            )
            print(
                f"  GT heights: min={min(gt_h_list):.1f} max={max(gt_h_list):.1f} "
                f"mean={sum(gt_h_list)/len(gt_h_list):.1f}"
            )
        else:
            print("  No GTs")
            continue
        if preds:
            p = preds[0]
            print(
                f"  Pred top-1: x={p.x:6.1f} y={p.y:6.1f} "
                f"ang={math.degrees(p.angle_rad):7.2f}deg "
                f"w={p.width:6.1f} h={p.height:6.1f}  "
                f"aspect(h/w)={p.height/max(p.width,1e-6):4.2f}"
            )
            # Compute angle error and IoU vs GT[0]
            raw_diff = math.degrees(p.angle_rad - gt.angle_rad)
            ang_err = _angle_error_deg(p.angle_rad, gt.angle_rad)
            iou = _oriented_iou(p, gt)
            print(f"  vs GT[0]: raw_angle_diff={raw_diff:7.2f}deg  symmetric_angle_err={ang_err:.2f}deg  IoU={iou:.3f}")

            # Best-match across GTs
            best_iou = 0.0
            best_ang = 0.0
            best_idx = -1
            for j, g in enumerate(gts):
                iou_j = _oriented_iou(p, g)
                if iou_j > best_iou:
                    best_iou = iou_j
                    best_ang = _angle_error_deg(p.angle_rad, g.angle_rad)
                    best_idx = j
            print(f"  best vs any GT: idx={best_idx} IoU={best_iou:.3f} ang_err={best_ang:.2f}deg")
        else:
            print("  No predictions")
        print()


if __name__ == "__main__":
    main()
