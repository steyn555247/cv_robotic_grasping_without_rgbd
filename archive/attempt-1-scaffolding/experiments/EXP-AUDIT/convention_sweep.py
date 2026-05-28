"""Decisive test: re-evaluate EXP-02b predictions under 4 angle-convention pairings.

We hold the predictions fixed and vary how GT is interpreted, to find which
convention pairing the heuristic + original pipeline actually assumed.

Conventions for GT angle:
  A = direction of the SHORTER side (current loader)
  B = direction of the LONGER  side (gripper-plate axis)

We also test predicting angle as-is vs angle+90, for completeness.
"""
from __future__ import annotations
import json, math, os, glob
import numpy as np
from src.eval.cornell import GraspRect, evaluate_predictions
from src.data.cornell_loader import CornellDataset, _parse_cpos_file  # type: ignore

PRED_DIR = r"experiments/EXP-02b_cropped_cornell_eval/predictions"

def gt_from_corners(pts: np.ndarray, longer_side: bool) -> GraspRect:
    """Build a GraspRect choosing angle along shorter (A) or longer (B) side."""
    side_a = pts[1] - pts[0]
    side_b = pts[2] - pts[1]
    la = float(np.linalg.norm(side_a)); lb = float(np.linalg.norm(side_b))
    cx = float(np.mean(pts[:, 0])); cy = float(np.mean(pts[:, 1]))
    if la <= lb:
        shorter_vec, longer_vec, short_len, long_len = side_a, side_b, la, lb
    else:
        shorter_vec, longer_vec, short_len, long_len = side_b, side_a, lb, la
    vec = longer_vec if longer_side else shorter_vec
    ang = math.atan2(float(vec[1]), float(vec[0]))
    while ang > math.pi/2: ang -= math.pi
    while ang < -math.pi/2: ang += math.pi
    # width=opening(short), height=plate(long) regardless; only angle axis varies
    return GraspRect(x=cx, y=cy, angle_rad=ang, width=short_len, height=long_len)

def load_preds(sid: str, fold: int):
    path = os.path.join(PRED_DIR, f"fold-{fold}", f"{sid}.json")
    if not os.path.exists(path):
        return None
    data = json.load(open(path))["predictions"]
    return [GraspRect(x=p["x"], y=p["y"], angle_rad=p["angle_rad"],
                      width=p["width"], height=p["height"]) for p in data]

def run(longer_side_gt: bool, rotate_pred_90: bool) -> dict:
    all_pred, all_gt = [], []
    for fold in range(5):
        ds = CornellDataset(split="image-wise", fold=fold, partition="test")
        for i in range(len(ds)):
            s = ds[i]; sid = s["sample_id"]
            preds = load_preds(sid, fold)
            if preds is None:
                continue
            # rebuild GT corners from the loader's raw parse
            raw = s.get("_raw_corners")
            if raw is None:
                # fall back: use loader grasps but re-derive from stored corners not available;
                # use the grasps_gt directly and optionally swap axis by +90
                gts = []
                for g in s["grasps_gt"]:
                    if longer_side_gt:
                        a = g.angle_rad + math.pi/2
                        while a > math.pi/2: a -= math.pi
                        while a < -math.pi/2: a += math.pi
                        gts.append(GraspRect(g.x, g.y, a, g.width, g.height))
                    else:
                        gts.append(g)
            else:
                gts = [gt_from_corners(c, longer_side_gt) for c in raw]
            if rotate_pred_90:
                pr = []
                for p in preds:
                    a = p.angle_rad + math.pi/2
                    while a > math.pi/2: a -= math.pi
                    while a < -math.pi/2: a += math.pi
                    pr.append(GraspRect(p.x, p.y, a, p.width, p.height))
                preds = pr
            all_pred.append(preds); all_gt.append(gts)
    res = evaluate_predictions(all_pred, all_gt)
    return {"top1": res["top1"], "top5": res["top5"], "iou": res["iou_mean"],
            "n": res["n_samples"]}

if __name__ == "__main__":
    print("Sweeping convention pairings on EXP-02b predictions...\n")
    print(f"{'GT angle axis':<20}{'pred rot':<12}{'Top-1':<10}{'Top-5':<10}{'IoU':<8}{'n'}")
    for longer in (False, True):
        for rot in (False, True):
            r = run(longer, rot)
            label = "LONGER(plate)" if longer else "SHORTER(open)"
            print(f"{label:<20}{('+90' if rot else 'as-is'):<12}"
                  f"{r['top1']*100:<10.2f}{r['top5']*100:<10.2f}{r['iou']:<8.3f}{r['n']}")
