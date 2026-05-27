"""EXP-02 harness: full heuristic pipeline over Cornell 5-fold image-wise split.

Runs the production heuristic (``src.methods.heuristic.detect.detect_grasp``)
with the grid-search-winning ``HeuristicConfig`` defaults across all 885
Cornell images using the canonical 5-fold image-wise split.

Depth strategy
--------------
Reuses the Depth-Anything-V2-Small inverse-depth cache produced by EXP-01 at
``experiments/EXP-01_cog_only_baseline/mono_depth_cache/<sample_id>.npz``.
The cache stores RAW inverse-depth (closer = higher) at 480x640; we min-max
normalise to [0, 1] at load time to match the contract of
:class:`src.methods.heuristic.depth.DepthEstimator.__call__`. The EXP-01
cache used bilinear interpolation on the depth model output while
``DepthEstimator`` uses bicubic; the difference is negligible (heuristic
thresholds the depth map at a percentile, which depends only on rank).

If a cache file is missing or the shape does not match the image, we fall
back to live ``DepthEstimator`` inference and write the result back to the
cache.

Writes
------
- ``results.json``                              (canonical schema)
- ``predictions/fold-<k>/<sample_id>.json``     (per-sample top-5)
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
RESULTS_FILE = EXP_DIR / "results.json"

# Shared depth cache from EXP-01 (raw inverse-depth, float16, 480x640).
EXP01_CACHE_DIR = (
    REPO_ROOT / "experiments" / "EXP-01_cog_only_baseline" / "mono_depth_cache"
)

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


def _normalize_depth(raw: np.ndarray) -> np.ndarray:
    """Min-max normalise an HxW inverse-depth array to [0, 1] float32."""
    arr = raw.astype(np.float32)
    d_min = float(arr.min())
    d_max = float(arr.max())
    return (arr - d_min) / (d_max - d_min + 1e-8)


def _load_cached_depth(sample_id: str, expected_shape: tuple[int, int]) -> Optional[np.ndarray]:
    """Return normalised depth from the EXP-01 cache, or None if unusable."""
    cache_path = EXP01_CACHE_DIR / f"{sample_id}.npz"
    if not cache_path.exists():
        return None
    try:
        raw = np.load(cache_path)["depth"]
    except Exception:
        return None
    if raw.shape != expected_shape:
        return None
    return _normalize_depth(raw)


def _grasp_to_dict(g: GraspRect) -> Dict[str, float]:
    return {
        "x": float(g.x),
        "y": float(g.y),
        "angle_rad": float(g.angle_rad),
        "width": float(g.width),
        "height": float(g.height),
    }


def _save_prediction(fold: int, sample_id: str, preds: List[GraspRect]) -> None:
    out_dir = PRED_ROOT / f"fold-{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_id": sample_id,
        "fold": fold,
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
    print(f"EXP-02 full_cornell_eval starting (seed={SEED})")
    gt_parity = _sanity_check_evaluator()

    config = HeuristicConfig()
    print(f"HeuristicConfig defaults: {asdict(config)}")

    # Lazy depth estimator — only instantiated on cache miss / shape mismatch.
    estimator = None
    cache_hits = 0
    cache_misses = 0
    failed_samples: List[Dict] = []  # records errors per sample

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
        preds_list: List[List[GraspRect]] = []
        gts_list: List[List[GraspRect]] = []

        fold_start = time.time()
        for i in range(len(ds_test)):
            s = ds_test[i]
            sample_id = s["sample_id"]
            image_rgb = s["image"]
            gt_grasps = s["grasps_gt"]
            h, w = image_rgb.shape[:2]

            # --- Depth: cache first, fall back to live inference ---
            t_d0 = time.perf_counter()
            depth = _load_cached_depth(sample_id, (h, w))
            if depth is not None:
                cache_hits += 1
            else:
                if estimator is None:
                    # Lazy import to avoid loading torch transformers if cache covers all.
                    from src.methods.heuristic.depth import DepthEstimator
                    print(f"  cache miss for {sample_id}; loading DepthEstimator ...")
                    estimator = DepthEstimator(model_name=MODEL_ID)
                try:
                    depth = estimator(image_rgb)
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
                # Write back to cache for future runs (float16 for parity).
                try:
                    EXP01_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        EXP01_CACHE_DIR / f"{sample_id}.npz",
                        depth=depth.astype(np.float16),
                    )
                except Exception:
                    pass  # cache write best-effort
                cache_misses += 1
            depth_load_times_ms.append((time.perf_counter() - t_d0) * 1000.0)

            # --- Heuristic ---
            t_h0 = time.perf_counter()
            try:
                preds = detect_grasp(image_rgb, depth, config)
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
                preds = []
            detect_times_ms.append((time.perf_counter() - t_h0) * 1000.0)

            preds_list.append(preds)
            gts_list.append(gt_grasps)
            sample_order.append(sample_id)

            _save_prediction(fold, sample_id, preds)

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

    # Strip per_sample_correct from per-fold structures (it goes into the
    # top-level per_sample_correct vector, indexed by sample_order).
    per_fold_clean = []
    for f in fold_results:
        per_fold_clean.append(
            {k: (float(v) if isinstance(v, (int, float)) else v)
             for k, v in f.items() if k != "per_sample_correct"}
        )

    results = {
        "experiment_id": "EXP-02",
        "name": "full_cornell_eval",
        "spec_version": "2026-05-25",
        "seed": SEED,
        "dataset": "cornell-imagewise",
        "split": "5-fold-image-wise (folds 0..4, test partitions)",
        "n_samples": len(sample_order),
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
            "interpolation_note": (
                "EXP-01 cache uses bilinear interpolation on the depth model "
                "output. DepthEstimator (used on cache miss) uses bicubic. "
                "Heuristic thresholds the depth map at a percentile, so the "
                "ranking is preserved either way; impact considered negligible."
            ),
            "normalization": "min-max to [0, 1] applied at load time",
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
            "Full heuristic pipeline (DepthAnythingV2-Small + contour + 80px "
            "PCA tangent + ray-cast + CoG-boost rank) over all 885 Cornell "
            "images, image-wise 5-fold split. Headline vs. EXP-01 mono "
            "(16.95%) is the contribution magnitude of the contour pipeline."
        ),
    }

    with RESULTS_FILE.open("w") as fh:
        json.dump(results, fh, indent=2)

    print("\n=== EXP-02 summary ===")
    print(
        f"top1 = {agg['top1_mean']:.4f} ± {agg['top1_std']:.4f}  "
        f"top5 = {agg['top5_mean']:.4f} ± {agg['top5_std']:.4f}  "
        f"iou = {agg['iou_mean_mean']:.4f}  ang = {agg['angle_error_deg_mean_mean']:.2f}"
    )
    print(f"per-fold top1: {agg['top1_per_fold']}")
    print(
        f"cache: hits={cache_hits}, misses={cache_misses}; "
        f"failed_samples={len(failed_samples)}"
    )
    print(f"wallclock: {wallclock:.1f}s ({wallclock/60.0:.1f} min)")
    print(f"results -> {RESULTS_FILE}")


if __name__ == "__main__":
    run()
