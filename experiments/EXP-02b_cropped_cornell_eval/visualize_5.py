"""EXP-02b qualitative figure generator.

Saves five PNGs (one per representative sample) showing, side by side:

1. Full 480x640 RGB with the crop region outlined in yellow.
2. The 300x400 cropped RGB (what the heuristic actually saw).
3. The full image with predicted top-1 grasp (red) and all GT grasps (green)
   overlaid in FULL-image coordinates.

Used as source for the paper's Fig. 3 if EXP-02b lands well.

All outputs in ``experiments/EXP-02b_*/qualitative/<sample_id>.png``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from src.data.cornell_loader import CornellDataset  # noqa: E402
from src.eval.cornell import GraspRect  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
PRED_ROOT = EXP_DIR / "predictions"
OUT_DIR = EXP_DIR / "qualitative"

CROP_X, CROP_Y, CROP_W, CROP_H = 100, 150, 400, 300

# Representative samples — chosen to span object types / outcomes. Drawn from
# the EXP-02 case IDs and a couple of common Cornell objects.
SAMPLE_IDS: List[str] = [
    "pcd0110",  # first sample in fold-0; EXP-02 was a false positive here
    "pcd0217",  # EXP-02 true positive
    "pcd0411",  # EXP-02 true positive
    "pcd0817",  # EXP-02 true positive
    "pcd0500",  # mid-dataset sample, diverse object
]


def _grasp_rect_from_dict(d: dict) -> GraspRect:
    return GraspRect(
        x=float(d["x"]),
        y=float(d["y"]),
        angle_rad=float(d["angle_rad"]),
        width=float(d["width"]),
        height=float(d["height"]),
    )


def _draw_grasp(ax, g: GraspRect, color: str, linewidth: float = 2.0,
                label: str | None = None) -> None:
    corners = g.get_corners()
    # Close the polygon.
    poly = np.vstack([corners, corners[0:1]])
    ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=linewidth,
            label=label)
    # Mark the gripper-opening axis (the two shorter sides) with thicker lines.
    # Corners are: TL, TR, BR, BL relative to opening axis.
    ax.plot(
        [corners[0, 0], corners[1, 0]], [corners[0, 1], corners[1, 1]],
        color=color, linewidth=linewidth + 1,
    )
    ax.plot(
        [corners[2, 0], corners[3, 0]], [corners[2, 1], corners[3, 1]],
        color=color, linewidth=linewidth + 1,
    )


def _find_fold_for_sample(sample_id: str) -> int | None:
    for fold in range(5):
        if (PRED_ROOT / f"fold-{fold}" / f"{sample_id}.json").exists():
            return fold
    return None


def _load_predictions_full(sample_id: str) -> tuple[int, List[GraspRect]]:
    fold = _find_fold_for_sample(sample_id)
    if fold is None:
        raise FileNotFoundError(
            f"No prediction file found for {sample_id} in any fold under {PRED_ROOT}"
        )
    with (PRED_ROOT / f"fold-{fold}" / f"{sample_id}.json").open() as fh:
        payload = json.load(fh)
    return fold, [_grasp_rect_from_dict(p) for p in payload["predictions"]]


def _load_sample(sample_id: str) -> dict:
    """Locate the sample by ID in any fold."""
    ds = CornellDataset(split="all")
    for i in range(len(ds)):
        s = ds[i]
        if s["sample_id"] == sample_id:
            return s
    raise KeyError(f"Sample {sample_id} not found in Cornell dataset")


def _render_one(sample_id: str) -> Path:
    sample = _load_sample(sample_id)
    img_full = sample["image"]
    gts: List[GraspRect] = sample["grasps_gt"]
    fold, preds_full = _load_predictions_full(sample_id)
    img_crop = img_full[CROP_Y:CROP_Y + CROP_H, CROP_X:CROP_X + CROP_W]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel 1: full image with crop region outlined ---
    ax = axes[0]
    ax.imshow(img_full)
    rect = mpatches.Rectangle(
        (CROP_X, CROP_Y), CROP_W, CROP_H,
        linewidth=2.0, edgecolor="yellow", facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)
    ax.set_title(f"{sample_id} — full 480x640 (crop in yellow)")
    ax.set_xlim(0, img_full.shape[1])
    ax.set_ylim(img_full.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Panel 2: cropped image (what the heuristic actually saw) ---
    ax = axes[1]
    ax.imshow(img_crop)
    ax.set_title(f"cropped 300x400 (heuristic input)")
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Panel 3: full image + predicted top-1 (red) + GT (green) ---
    ax = axes[2]
    ax.imshow(img_full)
    # GT rectangles in green.
    for j, g in enumerate(gts):
        _draw_grasp(
            ax, g, color="lime",
            linewidth=1.5,
            label="GT" if j == 0 else None,
        )
    # Predicted top-1 in red (if any).
    if preds_full:
        _draw_grasp(
            ax, preds_full[0], color="red",
            linewidth=2.2,
            label="pred top-1",
        )
    # Crop boundary for context.
    rect = mpatches.Rectangle(
        (CROP_X, CROP_Y), CROP_W, CROP_H,
        linewidth=1.0, edgecolor="yellow", facecolor="none",
        linestyle=":",
    )
    ax.add_patch(rect)
    ax.set_title(f"pred (red) vs GT (green) — fold {fold}")
    ax.set_xlim(0, img_full.shape[1])
    ax.set_ylim(img_full.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{sample_id}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    print(f"EXP-02b visualize_5: rendering {len(SAMPLE_IDS)} samples ...")
    for sid in SAMPLE_IDS:
        try:
            out_path = _render_one(sid)
            print(f"  OK {sid} -> {out_path}")
        except Exception as e:
            print(f"  FAIL {sid}: {type(e).__name__}: {e}")
    print(f"done. outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
