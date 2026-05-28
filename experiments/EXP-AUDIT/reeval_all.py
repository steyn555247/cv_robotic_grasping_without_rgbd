"""Re-evaluate existing predictions against the FIXED Cornell loader.

The detector predictions never changed; only the loader's angle convention
was buggy. So we just reload GT (now correct) and re-score saved predictions.
Covers EXP-01 (both variants), EXP-02, EXP-02b.
"""
from __future__ import annotations
import json, os
import numpy as np
from src.eval.cornell import GraspRect, evaluate_predictions
from src.data.cornell_loader import CornellDataset


def load_pred_file(path: str) -> list[GraspRect]:
    data = json.load(open(path))
    preds = data["predictions"] if isinstance(data, dict) else data
    out = []
    for p in preds:
        out.append(GraspRect(x=p["x"], y=p["y"], angle_rad=p["angle_rad"],
                             width=p["width"], height=p["height"]))
    return out


def reeval(pred_root: str, variant_sub: str = "") -> dict:
    """pred_root/fold-N/<sid>.json  ->  metrics over all 5 folds."""
    all_pred, all_gt = [], []
    n_missing = 0
    for fold in range(5):
        ds = CornellDataset(split="image-wise", fold=fold, partition="test")
        fold_dir = os.path.join(pred_root, variant_sub, f"fold-{fold}")
        if not os.path.isdir(fold_dir):
            fold_dir = os.path.join(pred_root, f"fold-{fold}")
        for i in range(len(ds)):
            s = ds[i]; sid = s["sample_id"]
            pf = os.path.join(fold_dir, f"{sid}.json")
            if not os.path.exists(pf):
                n_missing += 1
                continue
            all_pred.append(load_pred_file(pf))
            all_gt.append(s["grasps_gt"])
    res = evaluate_predictions(all_pred, all_gt)
    return {"top1": res["top1"], "top5": res["top5"], "iou": res["iou_mean"],
            "angle": res["angle_error_deg_mean"], "n": res["n_samples"], "missing": n_missing}


if __name__ == "__main__":
    jobs = [
        ("EXP-02b heuristic (cropped)", "experiments/EXP-02b_cropped_cornell_eval/predictions", ""),
        ("EXP-02  heuristic (uncropped)", "experiments/EXP-02_full_cornell_eval/predictions", ""),
        ("EXP-01  CoG-only (mono)", "experiments/EXP-01_cog_only_baseline/predictions", "mono_depth_mask"),
        ("EXP-01  CoG-only (GT)", "experiments/EXP-01_cog_only_baseline/predictions", "gt_depth_mask"),
    ]
    print(f"{'Run':<34}{'Top-1':<9}{'Top-5':<9}{'IoU':<8}{'Angle':<8}{'n':<6}{'miss'}")
    print("-" * 80)
    for label, root, sub in jobs:
        try:
            r = reeval(root, sub)
            print(f"{label:<34}{r['top1']*100:<9.2f}{r['top5']*100:<9.2f}"
                  f"{r['iou']:<8.3f}{r['angle']:<8.2f}{r['n']:<6}{r['missing']}")
        except Exception as e:
            print(f"{label:<34}ERROR: {type(e).__name__}: {e}")
