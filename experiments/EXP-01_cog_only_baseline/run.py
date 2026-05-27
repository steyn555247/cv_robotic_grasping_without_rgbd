"""EXP-01 harness: CoG-only baseline over Cornell 5-fold image-wise split.

Runs two variants:
- ``gt_depth_mask``: foreground mask from Cornell's GT depth channel
  (informational upper bound; NOT RGB-only).
- ``mono_depth_mask``: foreground mask from Depth-Anything-V2-Small on RGB
  (the strictly-RGB-only headline number).

Writes:
- ``results.json``                   (canonical schema, both variants)
- ``predictions/<variant>/fold-<k>/<sample_id>.json``  (one per sample)
- ``mono_depth_cache/<sample_id>.npz`` (cached inverse-depth predictions)

All metrics come from ``src.eval.cornell.evaluate_predictions``; the harness
never computes Cornell numbers locally. Seeds: random=0, numpy=0, torch=0.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

# Force HuggingFace offline mode after the first successful download to
# isolate the run from transient network errors.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT))

from src.data.cornell_loader import CornellDataset  # noqa: E402
from src.eval.cornell import GraspRect, evaluate_predictions  # noqa: E402
from src.methods.cog_baseline import detect_cog_grasp  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
PRED_ROOT = EXP_DIR / "predictions"
CACHE_DIR = EXP_DIR / "mono_depth_cache"
RESULTS_FILE = EXP_DIR / "results.json"

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
N_FOLDS = 5
SEED = 0


def _set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "no-git"


def _load_depth_anything():
    """Load Depth-Anything-V2-Small; eval-mode, no grad."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    print(f"Loading {MODEL_ID} ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return processor, model


def _predict_invdepth(image_rgb: np.ndarray, processor, model) -> np.ndarray:
    """Run Depth Anything V2 to produce a HxW inverse-depth map (closer=higher)."""
    img = Image.fromarray(image_rgb)
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    pd = out.predicted_depth  # (1, H', W')
    pd_full = torch.nn.functional.interpolate(
        pd.unsqueeze(1),
        size=image_rgb.shape[:2],
        mode="bilinear",
        align_corners=False,
    ).squeeze().detach().cpu().numpy()
    return pd_full.astype(np.float32)


def _get_or_cache_invdepth(sample_id: str, image_rgb: np.ndarray, processor, model):
    """Cache inverse-depth predictions to disk for re-runs."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{sample_id}.npz"
    if cache_path.exists():
        try:
            return np.load(cache_path)["depth"], 0.0
        except Exception:
            pass  # corrupted; recompute
    t0 = time.time()
    invd = _predict_invdepth(image_rgb, processor, model)
    np.savez_compressed(cache_path, depth=invd.astype(np.float16))
    return invd, time.time() - t0


def _grasp_to_dict(g: GraspRect) -> Dict[str, float]:
    return {
        "x": float(g.x),
        "y": float(g.y),
        "angle_rad": float(g.angle_rad),
        "width": float(g.width),
        "height": float(g.height),
    }


def _save_prediction(
    variant: str, fold: int, sample_id: str, preds: List[GraspRect]
) -> None:
    out_dir = PRED_ROOT / variant / f"fold-{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_id": sample_id,
        "fold": fold,
        "variant": variant,
        "predictions": [_grasp_to_dict(p) for p in preds],
    }
    with (out_dir / f"{sample_id}.json").open("w") as fh:
        json.dump(payload, fh)


def _aggregate_fold_metrics(per_fold: List[dict]) -> dict:
    keys = ["top1", "top5", "iou_mean", "angle_error_deg_mean"]
    agg = {f"{k}_per_fold": [float(f[k]) for f in per_fold] for k in keys}
    for k in keys:
        agg[f"{k}_mean"] = float(np.mean(agg[f"{k}_per_fold"]))
        agg[f"{k}_std"] = float(np.std(agg[f"{k}_per_fold"]))
    agg["n_samples_per_fold"] = [int(f["n_samples"]) for f in per_fold]
    agg["n_samples_total"] = int(sum(agg["n_samples_per_fold"]))
    return agg


def _sanity_check_evaluator() -> None:
    """Confirm evaluate_predictions(gt, gt) == 1.0 top-1 before running."""
    ds = CornellDataset(split="all")
    gts = [ds[i]["grasps_gt"] for i in range(40)]
    preds = [list(g) for g in gts]
    m = evaluate_predictions(preds, gts)
    if abs(m["top1"] - 1.0) > 1e-9:
        raise RuntimeError(
            f"Sanity check failed: evaluate_predictions(gt, gt) top1={m['top1']}"
        )
    print(f"Sanity check passed: evaluate_predictions(gt, gt) top1={m['top1']:.6f}")


def run() -> None:
    _set_seeds(SEED)
    print(f"EXP-01 cog_only_baseline starting (seed={SEED})")
    _sanity_check_evaluator()

    processor, model = _load_depth_anything()

    fold_results: Dict[str, List[dict]] = {"gt_depth_mask": [], "mono_depth_mask": []}
    per_sample_correct_total: Dict[str, List[bool]] = {
        "gt_depth_mask": [],
        "mono_depth_mask": [],
    }
    sample_order: List[str] = []
    mono_pred_times: List[float] = []
    detect_times: List[float] = []
    wall_start = time.time()

    for fold in range(N_FOLDS):
        ds_test = CornellDataset(
            split="fold", fold=fold, partition="test", split_type="image-wise"
        )
        print(f"\n[Fold {fold}] test size = {len(ds_test)}")
        preds_gt: List[List[GraspRect]] = []
        preds_mono: List[List[GraspRect]] = []
        gts: List[List[GraspRect]] = []

        for i in range(len(ds_test)):
            s = ds_test[i]
            sample_id = s["sample_id"]
            image_rgb = s["image"]
            depth_gt = s["depth_gt"]

            invd, dt_mono = _get_or_cache_invdepth(
                sample_id, image_rgb, processor, model
            )
            if dt_mono > 0:
                mono_pred_times.append(dt_mono)

            t0 = time.time()
            p_gt = detect_cog_grasp(
                image_rgb, depth_gt, foreground_is_higher=False
            )
            p_mono = detect_cog_grasp(
                image_rgb, invd, foreground_is_higher=True
            )
            detect_times.append(time.time() - t0)

            preds_gt.append(p_gt)
            preds_mono.append(p_mono)
            gts.append(s["grasps_gt"])
            sample_order.append(sample_id)

            _save_prediction("gt_depth_mask", fold, sample_id, p_gt)
            _save_prediction("mono_depth_mask", fold, sample_id, p_mono)

            if (i + 1) % 25 == 0 or i == len(ds_test) - 1:
                print(f"  fold-{fold} {i+1}/{len(ds_test)} samples processed")

        m_gt = evaluate_predictions(preds_gt, gts)
        m_mono = evaluate_predictions(preds_mono, gts)
        m_gt["n_samples"] = len(gts)
        m_mono["n_samples"] = len(gts)
        fold_results["gt_depth_mask"].append(m_gt)
        fold_results["mono_depth_mask"].append(m_mono)
        per_sample_correct_total["gt_depth_mask"].extend(m_gt["per_sample_correct"])
        per_sample_correct_total["mono_depth_mask"].extend(m_mono["per_sample_correct"])
        print(
            f"  fold-{fold}: GT top1={m_gt['top1']:.3f}  Mono top1={m_mono['top1']:.3f}"
        )

    wallclock = time.time() - wall_start

    results = {
        "experiment_id": "EXP-01",
        "name": "cog_only_baseline",
        "spec_version": "2026-05-25",
        "seed": SEED,
        "dataset": "cornell-imagewise",
        "split": "5-fold-image-wise (folds 0..4, test partitions)",
        "n_samples": len(sample_order),
        "variants": {},
        "hyperparameters": {
            "width_fraction": 0.6,
            "height_px": 20.0,
            "mono_depth_model": MODEL_ID,
            "mask_strategy": "DoG-residual (ksize=151, sigma=50, thr=0.5*std) "
            "in central 70%x76% crop; largest CC by area/dist^2-to-centre",
            "angle_convention": "Cornell angle_rad = direction of major axis "
            "(empirically matches Cornell GT; spec text 'minor' refers to the "
            "narrow dimension which the gripper closes ACROSS — see notes.md)",
        },
        "wallclock_seconds": wallclock,
        "git_sha": _git_sha(),
        "runtime_per_image_ms_mono_depth": (
            float(np.mean(mono_pred_times) * 1000.0) if mono_pred_times else None
        ),
        "runtime_per_image_ms_detect": (
            float(np.mean(detect_times) * 1000.0) if detect_times else 0.0
        ),
        "sample_order": sample_order,
        "notes": (
            "CoG + PCA on a foreground mask; ignores RGB content. GT-depth-mask "
            "variant is an informational upper bound (depth used only to make "
            "the mask). Mono-depth-mask variant is the strictly RGB-only "
            "headline number."
        ),
    }

    for variant in ("gt_depth_mask", "mono_depth_mask"):
        per_fold = fold_results[variant]
        agg = _aggregate_fold_metrics(per_fold)
        results["variants"][variant] = {
            "per_fold": [
                {k: (float(v) if isinstance(v, (int, float)) else v)
                 for k, v in f.items() if k != "per_sample_correct"}
                for f in per_fold
            ],
            "aggregated": agg,
            "metrics": {
                "top1": agg["top1_mean"],
                "top5": agg["top5_mean"],
                "iou_mean": agg["iou_mean_mean"],
                "angle_error_deg": agg["angle_error_deg_mean_mean"],
                "top1_per_fold_std": agg["top1_std"],
            },
            "per_sample_correct": per_sample_correct_total[variant],
        }

    with RESULTS_FILE.open("w") as fh:
        json.dump(results, fh, indent=2)

    print("\n=== EXP-01 summary ===")
    for v in ("gt_depth_mask", "mono_depth_mask"):
        a = results["variants"][v]["aggregated"]
        print(
            f"{v}: top1={a['top1_mean']:.4f} ± {a['top1_std']:.4f} "
            f"iou={a['iou_mean_mean']:.4f} ang={a['angle_error_deg_mean_mean']:.2f}"
        )
    print(f"wallclock: {wallclock:.1f}s ({wallclock/60.0:.1f} min)")
    print(f"results -> {RESULTS_FILE}")


if __name__ == "__main__":
    run()
