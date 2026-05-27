"""EXP-02b harness: heuristic pipeline over Cornell with the original manual crop.

This experiment reproduces the original grid-search setting that reported
75.89% Top-1 on a manually-cropped Cornell subset. The only difference vs.
EXP-02 is a fixed manual crop applied to the RGB image before any
processing. Depth is re-run on the CROPPED image (cannot reuse EXP-01's
full-image depth cache), and predictions are mapped back to FULL-image
coordinates before evaluation against the canonical full-image GT.

Pipeline per sample
-------------------
1. Load full 480x640 RGB image and full-image GT grasps.
2. Crop to ``image[150:450, 100:500]`` (the exact crop region documented in
   ``cornell comparisson/GRID_SEARCH_README.md``). Result: 300x400 RGB.
3. Run DepthAnythingV2-Small on the CROPPED RGB (cache locally; full-image
   EXP-01 cache does not apply here).
4. Run ``src.methods.heuristic.detect.detect_grasp`` on the cropped image
   and cropped depth. Output is in CROPPED coordinates.
5. Map predictions back to FULL-image coordinates by adding the crop offset
   (``x += 100``, ``y += 150``). Angle, width, height are unchanged.
6. Evaluate full-coordinate predictions against full-coordinate GT via the
   canonical Cornell evaluator.

Writes
------
- ``results.json``                              (canonical schema)
- ``predictions/fold-<k>/<sample_id>.json``     (per-sample top-5, FULL coords)
- ``depth_cache/<sample_id>.npz``               (cropped-image depth, float16)
- ``notes.md``                                  (written by the runner)

All metrics come from ``src.eval.cornell.evaluate_predictions``; the harness
never computes Cornell numbers locally. Seeds: random=0, numpy=0, torch=0.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Force HuggingFace Hub offline (model is already cached locally from EXP-01).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.cornell_loader import CornellDataset  # noqa: E402
from src.eval.cornell import GraspRect, evaluate_predictions  # noqa: E402
from src.methods.heuristic.config import HeuristicConfig  # noqa: E402
from src.methods.heuristic.detect import detect_grasp  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
PRED_ROOT = EXP_DIR / "predictions"
DEPTH_CACHE_DIR = EXP_DIR / "depth_cache"
RESULTS_FILE = EXP_DIR / "results.json"

# ---- Manual crop region (matches the prior grid-search configuration) ------
# image[CROP_Y : CROP_Y + CROP_H, CROP_X : CROP_X + CROP_W]
# i.e. x in [100, 500), y in [150, 450) -> 300x400 crop from the 480x640 frame.
CROP_X: int = 100
CROP_W: int = 400
CROP_Y: int = 150
CROP_H: int = 300

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
N_FOLDS = 5
SEED = 0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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


def _crop_image(image_full: np.ndarray) -> np.ndarray:
    """Crop the full 480x640 RGB image to the 300x400 manual crop region."""
    return image_full[CROP_Y:CROP_Y + CROP_H, CROP_X:CROP_X + CROP_W]


def _load_cached_depth(sample_id: str, expected_shape: tuple[int, int]) -> Optional[np.ndarray]:
    """Return cropped-image depth from this experiment's cache, or None."""
    cache_path = DEPTH_CACHE_DIR / f"{sample_id}.npz"
    if not cache_path.exists():
        return None
    try:
        raw = np.load(cache_path)["depth"]
    except Exception:
        return None
    if raw.shape != expected_shape:
        return None
    # Already stored in [0, 1] float16 -> just cast to float32.
    return raw.astype(np.float32)


def _save_cached_depth(sample_id: str, depth: np.ndarray) -> None:
    try:
        DEPTH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            DEPTH_CACHE_DIR / f"{sample_id}.npz",
            depth=depth.astype(np.float16),
        )
    except Exception:
        pass  # cache write best-effort


def _map_prediction_to_full(g: GraspRect) -> GraspRect:
    """Translate a cropped-coordinate prediction to full-image coordinates."""
    return GraspRect(
        x=float(g.x) + float(CROP_X),
        y=float(g.y) + float(CROP_Y),
        angle_rad=float(g.angle_rad),
        width=float(g.width),
        height=float(g.height),
    )


def _grasp_to_dict(g: GraspRect) -> Dict[str, float]:
    return {
        "x": float(g.x),
        "y": float(g.y),
        "angle_rad": float(g.angle_rad),
        "width": float(g.width),
        "height": float(g.height),
    }


def _save_prediction(fold: int, sample_id: str, preds_full: List[GraspRect]) -> None:
    """Save per-sample top-5 predictions in FULL-image coordinates."""
    out_dir = PRED_ROOT / f"fold-{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_id": sample_id,
        "fold": fold,
        "coord_frame": "full_image_480x640",
        "crop_applied": {
            "x": CROP_X, "y": CROP_Y, "w": CROP_W, "h": CROP_H,
        },
        "predictions": [_grasp_to_dict(p) for p in preds_full],
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


def _sanity_check_evaluator() -> dict:
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
    return {
        "n": 40,
        "top1": float(m["top1"]),
        "top5": float(m["top5"]),
        "iou_mean": float(m["iou_mean"]),
        "angle_error_deg_mean": float(m["angle_error_deg_mean"]),
    }


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run() -> None:
    _set_seeds(SEED)
    print(f"EXP-02b cropped_cornell_eval starting (seed={SEED})")
    print(
        f"Crop region: x in [{CROP_X}, {CROP_X + CROP_W}), "
        f"y in [{CROP_Y}, {CROP_Y + CROP_H}) -> {CROP_H}x{CROP_W} crop"
    )
    gt_parity = _sanity_check_evaluator()

    config = HeuristicConfig()
    print(f"HeuristicConfig defaults: {asdict(config)}")

    # Lazy depth estimator — only instantiated on cache miss.
    estimator = None
    cache_hits = 0
    cache_misses = 0
    failed_samples: List[Dict] = []

    fold_results: List[dict] = []
    per_sample_correct_total: List[bool] = []
    sample_order: List[str] = []

    detect_times_ms: List[float] = []
    depth_load_times_ms: List[float] = []
    wall_start = time.time()

    for fold in range(N_FOLDS):
        ds_test = CornellDataset(
            split="image-wise", fold=fold, partition="test", split_type="image-wise"
        )
        print(f"\n[Fold {fold}] test size = {len(ds_test)}")
        preds_list: List[List[GraspRect]] = []  # FULL-image coords
        gts_list: List[List[GraspRect]] = []     # FULL-image coords (unchanged)

        fold_start = time.time()
        for i in range(len(ds_test)):
            s = ds_test[i]
            sample_id = s["sample_id"]
            image_full = s["image"]
            gt_grasps = s["grasps_gt"]  # already in full-image coords

            # --- Manual crop ---
            image_cropped = _crop_image(image_full)
            h_c, w_c = image_cropped.shape[:2]
            assert (h_c, w_c) == (CROP_H, CROP_W), (
                f"Crop sanity failed: got {(h_c, w_c)}, expected {(CROP_H, CROP_W)}"
            )

            # --- Depth on CROPPED image: cache first, fall back to live inference ---
            t_d0 = time.perf_counter()
            depth_cropped = _load_cached_depth(sample_id, (h_c, w_c))
            if depth_cropped is not None:
                cache_hits += 1
            else:
                if estimator is None:
                    from src.methods.heuristic.depth import DepthEstimator
                    print(f"  cache miss for {sample_id}; loading DepthEstimator ...")
                    estimator = DepthEstimator(model_name=MODEL_ID)
                try:
                    depth_cropped = estimator(image_cropped)
                except Exception as e:
                    failed_samples.append(
                        {
                            "sample_id": sample_id,
                            "fold": fold,
                            "stage": "depth_inference",
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )
                    preds_list.append([])
                    gts_list.append(gt_grasps)
                    sample_order.append(sample_id)
                    _save_prediction(fold, sample_id, [])
                    continue
                _save_cached_depth(sample_id, depth_cropped)
                cache_misses += 1
            depth_load_times_ms.append((time.perf_counter() - t_d0) * 1000.0)

            # --- Heuristic on CROPPED inputs ---
            t_h0 = time.perf_counter()
            try:
                preds_cropped = detect_grasp(image_cropped, depth_cropped, config)
            except Exception as e:
                failed_samples.append(
                    {
                        "sample_id": sample_id,
                        "fold": fold,
                        "stage": "detect_grasp",
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(limit=3),
                    }
                )
                preds_cropped = []
            detect_times_ms.append((time.perf_counter() - t_h0) * 1000.0)

            # --- CRITICAL: map predictions back to FULL-image coordinates ---
            preds_full = [_map_prediction_to_full(p) for p in preds_cropped]

            preds_list.append(preds_full)
            gts_list.append(gt_grasps)
            sample_order.append(sample_id)

            _save_prediction(fold, sample_id, preds_full)

            if (i + 1) % 25 == 0 or i == len(ds_test) - 1:
                elapsed = time.time() - fold_start
                rate = (i + 1) / max(elapsed, 1e-6)
                print(
                    f"  fold-{fold} {i+1}/{len(ds_test)} samples "
                    f"({rate:.1f} img/s, elapsed {elapsed:.1f}s)"
                )

        m = evaluate_predictions(preds_list, gts_list)
        m["n_samples"] = len(gts_list)
        fold_results.append(m)
        per_sample_correct_total.extend(m["per_sample_correct"])
        print(
            f"  fold-{fold} DONE: top1={m['top1']:.4f}  top5={m['top5']:.4f}  "
            f"iou={m['iou_mean']:.4f}  ang={m['angle_error_deg_mean']:.2f}  "
            f"n={m['n_samples']}"
        )

    wallclock = time.time() - wall_start
    agg = _aggregate_fold_metrics(fold_results)

    per_fold_clean = []
    for f in fold_results:
        per_fold_clean.append(
            {k: (float(v) if isinstance(v, (int, float)) else v)
             for k, v in f.items() if k != "per_sample_correct"}
        )

    results = {
        "experiment_id": "EXP-02b",
        "name": "cropped_cornell_eval",
        "spec_version": "2026-05-27",
        "seed": SEED,
        "dataset": "cornell-imagewise",
        "split": "5-fold-image-wise (folds 0..4, test partitions)",
        "n_samples": len(sample_order),
        "crop": {
            "x": CROP_X, "y": CROP_Y, "w": CROP_W, "h": CROP_H,
            "description": (
                "Manual crop image[150:450, 100:500] applied to the 480x640 "
                "RGB before depth and heuristic. Predictions are mapped back "
                "to full-image coordinates (x += 100, y += 150) before "
                "evaluation against full-image GT."
            ),
        },
        "metrics": {
            "top1": agg["top1_mean"],
            "top1_std": agg["top1_std"],
            "top5": agg["top5_mean"],
            "top5_std": agg["top5_std"],
            "iou_mean": agg["iou_mean_mean"],
            "iou_mean_std": agg["iou_mean_std"],
            "angle_error_deg": agg["angle_error_deg_mean_mean"],
            "angle_error_deg_std": agg["angle_error_deg_mean_std"],
            "runtime_per_image_ms": (
                float(np.mean(detect_times_ms)) if detect_times_ms else 0.0
            ),
        },
        "gt_parity_check": gt_parity,
        "per_fold": per_fold_clean,
        "aggregated": agg,
        "hyperparameters": asdict(config),
        "depth": {
            "model": MODEL_ID,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "input_size": [CROP_H, CROP_W],
            "normalization": (
                "DepthEstimator outputs depth pre-normalised to [0, 1]; "
                "stored float16 in this experiment's depth_cache/."
            ),
        },
        "runtime": {
            "wallclock_seconds": wallclock,
            "wallclock_minutes": wallclock / 60.0,
            "detect_ms_mean": (
                float(np.mean(detect_times_ms)) if detect_times_ms else 0.0
            ),
            "detect_ms_p50": (
                float(np.percentile(detect_times_ms, 50)) if detect_times_ms else 0.0
            ),
            "detect_ms_p95": (
                float(np.percentile(detect_times_ms, 95)) if detect_times_ms else 0.0
            ),
            "depth_load_ms_mean": (
                float(np.mean(depth_load_times_ms)) if depth_load_times_ms else 0.0
            ),
        },
        "wallclock_seconds": wallclock,
        "git_sha": _git_sha(),
        "sample_order": sample_order,
        "per_sample_correct": per_sample_correct_total,
        "failed_samples": failed_samples,
        "notes": (
            "Heuristic pipeline applied AFTER the original 300x400 manual "
            "crop (x[100:500], y[150:450]). Depth re-run on the cropped "
            "image; predictions mapped back to full-image coordinates for "
            "evaluation against full-image GT. Tests whether the prior "
            "75.89% Top-1 number can be reproduced when the heuristic is "
            "given the localized input it was originally tuned for."
        ),
    }

    with RESULTS_FILE.open("w") as fh:
        json.dump(results, fh, indent=2)

    print("\n=== EXP-02b summary ===")
    print(
        f"top1 = {agg['top1_mean']:.4f} ± {agg['top1_std']:.4f}  "
        f"top5 = {agg['top5_mean']:.4f} ± {agg['top5_std']:.4f}  "
        f"iou = {agg['iou_mean_mean']:.4f}  ang = {agg['angle_error_deg_mean_mean']:.2f}"
    )
    print(f"per-fold top1: {agg['top1_per_fold']}")
    print(
        f"depth cache: hits={cache_hits}, misses={cache_misses}; "
        f"failed_samples={len(failed_samples)}"
    )
    print(f"wallclock: {wallclock:.1f}s ({wallclock/60.0:.1f} min)")
    print(f"results -> {RESULTS_FILE}")


if __name__ == "__main__":
    run()
