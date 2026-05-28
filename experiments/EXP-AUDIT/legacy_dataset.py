"""Reconstruct the legacy ``CornellGraspDataset`` sample list and split.

Mirrors the load + split that the prior grid-search script applied:
1. ``CornellGraspDataset.load_dataset(max_samples=850)`` (sorted by full path,
   keeps only samples with at least one valid GT grasp, stops at 850).
2. ``split_dataset(samples, train_ratio=0.5, random_seed=42)`` (NumPy
   permutation, first 50% becomes train).
3. The manual crop is applied; samples with zero adjusted grasps are dropped.

Returns just the list of sample IDs that the grid search actually evaluated.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

LEGACY_DATA_DIR = REPO_ROOT / "cornell comparisson" / "cornell_dataset"

# Legacy manual crop (matches GRID_SEARCH_README.md and cell 52).
LEGACY_CROP_X = 100
LEGACY_CROP_Y = 150
LEGACY_CROP_W = 400
LEGACY_CROP_H = 300


def _legacy_load_grasp_rectangles(filepath: str) -> list:
    """Verbatim from CornellGraspDataset._load_grasp_rectangles."""
    import cv2  # noqa: F401  (kept to match the original import context)
    grasps = []
    try:
        with open(filepath) as fh:
            lines = fh.readlines()
        for i in range(0, len(lines) - 3, 4):
            corners = []
            valid = True
            for j in range(4):
                try:
                    parts = lines[i + j].strip().split()
                    if len(parts) >= 2:
                        x, y = float(parts[0]), float(parts[1])
                        if not (np.isnan(x) or np.isnan(y)):
                            corners.append([x, y])
                        else:
                            valid = False
                            break
                    else:
                        valid = False
                        break
                except Exception:
                    valid = False
                    break
            if valid and len(corners) == 4:
                corners_np = np.array(corners)
                center = corners_np.mean(axis=0)
                dx = corners_np[1, 0] - corners_np[0, 0]
                dy = corners_np[1, 1] - corners_np[0, 1]
                angle = float(np.arctan2(dy, dx))
                width = float(np.linalg.norm(corners_np[1] - corners_np[0]))
                height = float(np.linalg.norm(corners_np[2] - corners_np[1]))
                grasps.append(
                    {
                        "corners": corners_np,
                        "center": center,
                        "angle": angle,
                        "angle_deg": float(np.degrees(angle)),
                        "width": width,
                        "height": height,
                    }
                )
    except Exception:
        pass
    return grasps


def legacy_load_samples(max_samples: int = 850, data_dir: Path = LEGACY_DATA_DIR) -> List[dict]:
    """Verbatim from CornellGraspDataset.load_dataset (cell 14)."""
    samples = []
    patterns = [
        os.path.join(str(data_dir), "**", "pcd*r.png"),
        os.path.join(str(data_dir), "**", "*r.png"),
        os.path.join(str(data_dir), "*", "pcd*r.png"),
    ]
    rgb_files = []
    for pattern in patterns:
        rgb_files.extend(glob.glob(pattern, recursive=True))
    rgb_files = list(set(rgb_files))
    for rgb_path in sorted(rgb_files):
        if not rgb_path.endswith("r.png"):
            continue
        base = rgb_path[:-5]
        grasp_path = base + "cpos.txt"
        if not os.path.exists(grasp_path):
            continue
        grasps = _legacy_load_grasp_rectangles(grasp_path)
        if not grasps:
            continue
        depth_path = base + "d.tiff"
        if not os.path.exists(depth_path):
            depth_path = None
        samples.append(
            {
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "grasps": grasps,
                "id": os.path.basename(base),
            }
        )
        if max_samples and len(samples) >= max_samples:
            break
    return samples


def legacy_train_test_split(
    samples: List[dict], train_ratio: float = 0.5, random_seed: int = 42
) -> tuple[List[dict], List[dict]]:
    """Verbatim from grid_search_focused.py.split_dataset (cell 55)."""
    np.random.seed(random_seed)
    indices = np.random.permutation(len(samples))
    split_idx = int(len(samples) * train_ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    train_samples = [samples[i] for i in train_indices]
    test_samples = [samples[i] for i in test_indices]
    return train_samples, test_samples


def adjust_gt_to_crop(grasps: List[dict], crop_bbox=(LEGACY_CROP_X, LEGACY_CROP_Y, LEGACY_CROP_W, LEGACY_CROP_H)) -> List[dict]:
    """Mirror grid_search_focused.adjust_grasp_coordinates (cell 55).

    Translates GT grasps by the crop offset and drops grasps whose centre is
    outside the crop. Returns grasps in CROPPED-image coordinates.
    """
    x_off, y_off, cw, ch = crop_bbox
    out = []
    for g in grasps:
        cx, cy = g["center"]
        ncx = cx - x_off
        ncy = cy - y_off
        if 0 <= ncx < cw and 0 <= ncy < ch:
            adj_corners = []
            for corner in g["corners"]:
                adj_corners.append((corner[0] - x_off, corner[1] - y_off))
            out.append(
                {
                    "center": (ncx, ncy),
                    "angle": g["angle"],
                    "angle_deg": g["angle_deg"],
                    "width": g["width"],
                    "height": g["height"],
                    "corners": np.array(adj_corners),
                }
            )
    return out
