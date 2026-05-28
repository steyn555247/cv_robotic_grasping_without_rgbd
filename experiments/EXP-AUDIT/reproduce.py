"""Reproduce the legacy 75.89% Top-1 from saved canonical predictions.

Strategy
--------
The legacy grid search (``cornell comparisson/grid_search_focused.py``)
applied a fixed manual crop (x=100..500, y=150..450) to every Cornell sample,
ran the detector on the 400x300 crop, and evaluated using the buggy
``GraspEvaluator`` from notebook cell 24. We already have predictions from
EXP-02b: the SAME detector (faithful refactor) run on the SAME crop, with
predictions mapped back to full-image coordinates.

We translate those predictions back to CROPPED coordinates (subtract crop
offset), keep the predicted ``width`` (= ray-cast length) and ``angle``,
and convert ``height`` to 20.0 to mirror the legacy detector's hardcoded
``h=20`` output. We then:

1. Reconstruct the legacy 50/50 train/test split (NumPy seed 42 on the
   sorted, 850-cap sample list).
2. For each TRAIN sample, adjust GT to cropped coordinates and drop GT
   outside the crop. If zero adjusted grasps survive, drop the sample.
3. Run ``legacy_evaluate_dataset`` on the resulting (pred, gt) pairs.

Reports the Top-1/Top-5/any% numbers the legacy evaluator would have
produced. Cross-checked against the canonical evaluator on the same subset.

Variants reported:
- A) Legacy eval, predictions as we have them (canonical wrapped angles).
- B) Legacy eval, predictions with angles intentionally un-wrapped to the
     half-plane closest to GT's first angle (proxy for the legacy detector's
     un-wrapped output, which we no longer have on disk).
- C) Canonical eval on the same subset (sanity check).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXP_DIR))

from legacy_eval import (  # type: ignore  # noqa: E402
    LegacyCandidate,
    legacy_evaluate_dataset,
)
from legacy_dataset import (  # type: ignore  # noqa: E402
    LEGACY_CROP_X,
    LEGACY_CROP_Y,
    LEGACY_CROP_W,
    LEGACY_CROP_H,
    adjust_gt_to_crop,
    legacy_load_samples,
    legacy_train_test_split,
)
from src.eval.cornell import GraspRect, evaluate_predictions  # noqa: E402


# Where canonical EXP-02b dumped per-sample predictions (full-image coords).
EXP02B_PRED_ROOT = REPO_ROOT / "experiments" / "EXP-02b_cropped_cornell_eval" / "predictions"


def _load_pred_for_sample(sample_id: str) -> list[dict]:
    """Search all EXP-02b fold dirs for the sample's prediction JSON."""
    for fold in range(5):
        path = EXP02B_PRED_ROOT / f"fold-{fold}" / f"{sample_id}.json"
        if path.exists():
            with path.open() as fh:
                return json.load(fh)["predictions"]
    return []


def _to_cropped_legacy_candidates(preds_full: list[dict], legacy_height: float = 20.0) -> list[LegacyCandidate]:
    """Translate FULL-image predictions into CROPPED-coord LegacyCandidate.

    Width is preserved (ray-cast length). Height is replaced with 20.0 to
    mirror the legacy ``Candidate(h=20.0)`` default. Angle is kept verbatim.
    """
    out = []
    for p in preds_full:
        out.append(
            LegacyCandidate(
                x=float(p["x"]) - LEGACY_CROP_X,
                y=float(p["y"]) - LEGACY_CROP_Y,
                angle=float(p["angle_rad"]),
                width=float(p["width"]),
                height=legacy_height,
            )
        )
    return out


def _unwrap_to_nearest_branch(angle_rad: float, ref_angle_rad: float) -> float:
    """Shift ``angle_rad`` by k*pi (k in -1, 0, +1) toward ``ref_angle_rad``.

    The canonical detector wraps angles to ``[-pi/2, pi/2]``; the legacy
    detector emits raw ``atan2`` in ``(-pi, pi]``. We can't recover the exact
    raw value from the saved wrapped value (it could be a or a+pi). Picking
    the branch closer to ``ref`` (a GT angle) maximises the chance that the
    legacy IoU+angle-diff check passes for THIS GT — a generous
    interpretation. We report variant B with this transformation only to
    demonstrate that ANY un-wrap choice can amplify the angle-bug effect.
    """
    best = angle_rad
    best_diff = abs(angle_rad - ref_angle_rad)
    for k in (-1, 1):
        cand = angle_rad + k * math.pi
        d = abs(cand - ref_angle_rad)
        if d < best_diff:
            best = cand
            best_diff = d
    return best


def _canonical_eval_subset(
    preds_legacy: list[list[LegacyCandidate]],
    gts_legacy: list[list[dict]],
    image_offset: tuple[float, float] = (0.0, 0.0),
) -> dict:
    """Convert legacy-format (cropped-coord) data to GraspRect and run canonical eval."""
    canon_preds: list[list[GraspRect]] = []
    canon_gts: list[list[GraspRect]] = []
    ox, oy = image_offset
    for preds, gts in zip(preds_legacy, gts_legacy):
        canon_preds.append(
            [
                GraspRect(
                    x=float(p.x) + ox,
                    y=float(p.y) + oy,
                    angle_rad=float(p.angle),
                    width=float(p.width),
                    height=20.0,  # match the legacy variant we're auditing
                )
                for p in preds
            ]
        )
        # Build GraspRect from GT corners using the SHORTER-side convention
        # (matches the canonical loader).
        canon_gt = []
        for g in gts:
            corners = g["corners"]
            side_a = corners[1] - corners[0]
            side_b = corners[2] - corners[1]
            la = float(np.linalg.norm(side_a))
            lb = float(np.linalg.norm(side_b))
            if la <= lb:
                width, height, opening = la, lb, side_a
            else:
                width, height, opening = lb, la, side_b
            cx = float(np.mean(corners[:, 0])) + ox
            cy = float(np.mean(corners[:, 1])) + oy
            angle = math.atan2(float(opening[1]), float(opening[0]))
            while angle > math.pi / 2:
                angle -= math.pi
            while angle < -math.pi / 2:
                angle += math.pi
            canon_gt.append(
                GraspRect(x=cx, y=cy, angle_rad=angle, width=width, height=height)
            )
        canon_gts.append(canon_gt)
    return evaluate_predictions(canon_preds, canon_gts)


def main() -> None:
    print("Loading legacy sample list (max 850, sorted)...")
    samples_all = legacy_load_samples(max_samples=850)
    print(f"  loaded {len(samples_all)} samples with non-empty GT")

    train_samples, test_samples = legacy_train_test_split(samples_all, 0.5, 42)
    print(f"  train: {len(train_samples)}  test: {len(test_samples)}")

    pairs_train_a: list[tuple[list[LegacyCandidate], list[dict]]] = []
    pairs_train_b: list[tuple[list[LegacyCandidate], list[dict]]] = []
    pairs_test_a: list[tuple[list[LegacyCandidate], list[dict]]] = []
    skipped_no_grasp = 0
    skipped_no_pred = 0

    for sample in train_samples:
        sid = sample["id"]
        preds_full = _load_pred_for_sample(sid)
        if not preds_full:
            skipped_no_pred += 1
            continue
        gts_cropped = adjust_gt_to_crop(sample["grasps"])
        if not gts_cropped:
            skipped_no_grasp += 1
            continue
        preds_a = _to_cropped_legacy_candidates(preds_full)
        # Variant B: unwrap each prediction's angle toward the FIRST gt angle.
        ref = gts_cropped[0]["angle"]
        preds_b = [
            LegacyCandidate(
                x=p.x, y=p.y,
                angle=_unwrap_to_nearest_branch(p.angle, ref),
                width=p.width, height=p.height,
            )
            for p in preds_a
        ]
        pairs_train_a.append((preds_a, gts_cropped))
        pairs_train_b.append((preds_b, gts_cropped))

    for sample in test_samples:
        sid = sample["id"]
        preds_full = _load_pred_for_sample(sid)
        if not preds_full:
            continue
        gts_cropped = adjust_gt_to_crop(sample["grasps"])
        if not gts_cropped:
            continue
        pairs_test_a.append((_to_cropped_legacy_candidates(preds_full), gts_cropped))

    print(f"  train pairs evaluable: {len(pairs_train_a)}  "
          f"(skipped: no-pred={skipped_no_pred}, no-cropped-gt={skipped_no_grasp})")
    print(f"  test  pairs evaluable: {len(pairs_test_a)}")

    print("\n=== Variant A: legacy eval, canonical-wrapped angles, TRAIN subset ===")
    res_a_train = legacy_evaluate_dataset(
        (p for p, _ in pairs_train_a), (g for _, g in pairs_train_a)
    )
    print(json.dumps(res_a_train, indent=2))

    print("\n=== Variant A: legacy eval, canonical-wrapped angles, TEST subset ===")
    res_a_test = legacy_evaluate_dataset(
        (p for p, _ in pairs_test_a), (g for _, g in pairs_test_a)
    )
    print(json.dumps(res_a_test, indent=2))

    print("\n=== Variant B: legacy eval, angle unwrapped to nearest GT-branch, TRAIN ===")
    res_b_train = legacy_evaluate_dataset(
        (p for p, _ in pairs_train_b), (g for _, g in pairs_train_b)
    )
    print(json.dumps(res_b_train, indent=2))

    print("\n=== Variant C: CANONICAL eval on the same TRAIN subset (height=20, sanity) ===")
    canon_res = _canonical_eval_subset(
        [p for p, _ in pairs_train_a], [g for _, g in pairs_train_a]
    )
    print(json.dumps({k: v for k, v in canon_res.items() if k != "per_sample_correct"}, indent=2))

    # Variant D: canonical eval with the heuristic's NATIVE height (jaw plate),
    # not the legacy h=20. This is what EXP-02b actually reports.
    print("\n=== Variant D: CANONICAL eval, native heuristic height, TRAIN subset ===")
    pairs_train_native: list[tuple[list[LegacyCandidate], list[dict]]] = []
    for sample in train_samples:
        sid = sample["id"]
        preds_full = _load_pred_for_sample(sid)
        if not preds_full:
            continue
        gts_cropped = adjust_gt_to_crop(sample["grasps"])
        if not gts_cropped:
            continue
        preds_native = [
            LegacyCandidate(
                x=float(p["x"]) - LEGACY_CROP_X,
                y=float(p["y"]) - LEGACY_CROP_Y,
                angle=float(p["angle_rad"]),
                width=float(p["width"]),
                height=float(p["height"]),  # use the actual canonical height
            )
            for p in preds_full
        ]
        pairs_train_native.append((preds_native, gts_cropped))
    canon_res_native = _canonical_eval_subset(
        [p for p, _ in pairs_train_native], [g for _, g in pairs_train_native]
    )
    # have to re-do without forcing height=20 inside helper — patch it inline
    canon_preds: list[list[GraspRect]] = []
    canon_gts: list[list[GraspRect]] = []
    for preds, gts in pairs_train_native:
        canon_preds.append(
            [
                GraspRect(
                    x=float(p.x),
                    y=float(p.y),
                    angle_rad=float(p.angle),
                    width=float(p.width),
                    height=float(p.height),  # native height
                )
                for p in preds
            ]
        )
        gt_list = []
        for g in gts:
            corners = g["corners"]
            side_a = corners[1] - corners[0]
            side_b = corners[2] - corners[1]
            la = float(np.linalg.norm(side_a))
            lb = float(np.linalg.norm(side_b))
            if la <= lb:
                width, height, opening = la, lb, side_a
            else:
                width, height, opening = lb, la, side_b
            cx = float(np.mean(corners[:, 0]))
            cy = float(np.mean(corners[:, 1]))
            angle = math.atan2(float(opening[1]), float(opening[0]))
            while angle > math.pi / 2:
                angle -= math.pi
            while angle < -math.pi / 2:
                angle += math.pi
            gt_list.append(GraspRect(x=cx, y=cy, angle_rad=angle, width=width, height=height))
        canon_gts.append(gt_list)
    res_d = evaluate_predictions(canon_preds, canon_gts)
    print(json.dumps({k: v for k, v in res_d.items() if k != "per_sample_correct"}, indent=2))

    # Save a machine-readable summary
    out = {
        "n_samples_in_train_with_pred_and_crop": len(pairs_train_a),
        "n_samples_in_test_with_pred_and_crop": len(pairs_test_a),
        "legacy_reported_top1": 75.89,
        "variant_A_legacy_eval_canonical_angles_train": res_a_train,
        "variant_A_legacy_eval_canonical_angles_test": res_a_test,
        "variant_B_legacy_eval_unwrapped_angles_train": res_b_train,
        "variant_C_canonical_eval_h20_train": {
            "top1": canon_res["top1"],
            "top5": canon_res["top5"],
            "iou_mean": canon_res["iou_mean"],
            "angle_error_deg_mean": canon_res["angle_error_deg_mean"],
            "n_samples": canon_res["n_samples"],
        },
        "variant_D_canonical_eval_native_height_train": {
            "top1": res_d["top1"],
            "top5": res_d["top5"],
            "iou_mean": res_d["iou_mean"],
            "angle_error_deg_mean": res_d["angle_error_deg_mean"],
            "n_samples": res_d["n_samples"],
        },
    }
    out_path = Path(__file__).resolve().parent / "reproduce_results.json"
    with out_path.open("w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nWrote summary: {out_path}")


if __name__ == "__main__":
    main()
