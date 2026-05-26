"""Cornell Grasping Dataset loader.

Loader contract is documented in `.claude/agents/dataset-ops.md` and
`src/data/README.md`. Splits are read from `src/data/splits/cornell.json`
(immutable — never modify).

Cornell file layout (one folder per hundred IDs, plus a backgrounds folder
we ignore):

    cornell_dataset/
        01/  pcd0100r.png  pcd0100d.tiff  pcd0100cpos.txt  pcd0100cneg.txt  pcd0100.txt
            ...
        10/  ...
        backgrounds/   (ignored)

For each pcd ID we read:
- `pcdXXXXr.png`     — RGB image (PIL mode RGB, 640x480)
- `pcdXXXXd.tiff`    — depth, float32 in metres
- `pcdXXXXcpos.txt`  — positive grasp rectangles, 4 lines of "x y" floats per rect

This module deliberately re-uses the `GraspRect` defined in
`src.eval.cornell` so the loader and the evaluator agree on the grasp
representation. If that module does not yet exist, importing this loader
will fail — that is by design.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from src.eval.cornell import GraspRect  # canonical representation


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_ROOT = _REPO_ROOT / "cornell comparisson" / "cornell_dataset"
_SPLITS_FILE = Path(__file__).resolve().parent / "splits" / "cornell.json"


def _auto_root(root: Optional[str]) -> Path:
    if root is None or root == "<auto-detect>":
        return _DEFAULT_DATA_ROOT
    return Path(root)


def _pcd_subfolder(pcd_id: int) -> str:
    """Cornell groups pcd IDs in folders 01..10 by hundreds.

    pcd 100..199 -> 01, 200..299 -> 02, ..., 900..949 -> 09, 1000..1034 -> 10.
    """
    if 100 <= pcd_id <= 999:
        return f"{pcd_id // 100:02d}"
    if 1000 <= pcd_id <= 1099:
        return "10"
    raise ValueError(f"pcd ID {pcd_id} outside known Cornell range")


def _sample_paths(root: Path, pcd_id: int) -> dict:
    sub = _pcd_subfolder(pcd_id)
    base = root / sub / f"pcd{pcd_id:04d}"
    return {
        "rgb": base.with_name(f"pcd{pcd_id:04d}r.png"),
        "depth": base.with_name(f"pcd{pcd_id:04d}d.tiff"),
        "cpos": base.with_name(f"pcd{pcd_id:04d}cpos.txt"),
    }


# ---------------------------------------------------------------------------
# Grasp parsing
# ---------------------------------------------------------------------------

def _parse_cpos_file(path: Path) -> list[GraspRect]:
    """Parse a Cornell cpos.txt into a list of GraspRect.

    Each rectangle is 4 consecutive lines, "x y" floats. Cornell's corner
    order is consistent: lines 0..3 trace the rectangle in order. Sides
    (0->1) and (2->3) are parallel; sides (1->2) and (3->0) are parallel.
    Convention used here: gripper *width* (the opening direction along which
    the fingers close) is the SHORTER side, matching the convention used by
    Jiang et al. and downstream evaluators.

    Rectangles containing NaN coordinates (the canonical example is
    pcd0165) are silently dropped.
    """
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    grasps: list[GraspRect] = []
    for i in range(0, len(lines) - 3, 4):
        rect_lines = lines[i:i + 4]
        try:
            pts = np.array(
                [[float(tok) for tok in ln.split()] for ln in rect_lines],
                dtype=np.float64,
            )
        except ValueError:
            # Non-numeric (NaN strings get parsed by float(), but anything
            # else here is malformed). Skip the rectangle.
            continue
        if not np.all(np.isfinite(pts)):
            continue
        if pts.shape != (4, 2):
            continue
        grasps.append(_corners_to_grasp_rect(pts))
    return grasps


def _corners_to_grasp_rect(pts: np.ndarray) -> GraspRect:
    """Convert 4 ordered corner pixels to a GraspRect.

    Cornell encodes a grasp as a rectangle whose two pairs of parallel
    sides are: the *jaw* sides (length = gripper opening = `width`) and
    the *finger* sides (length = jaw thickness = `height`). The corner
    ordering is consistent across the dataset: side (pts[0]->pts[1]) is
    perpendicular to the gripper opening, i.e. it is the *finger* side.

    We follow the widely used convention (Lenz 2015 onward): assign
    `width` to the shorter of the two side lengths so that `angle_rad`
    represents the orientation of the gripper opening axis.
    """
    side_a = pts[1] - pts[0]  # one side (finger side per Cornell convention)
    side_b = pts[2] - pts[1]  # adjacent side (jaw side per Cornell convention)
    len_a = float(np.linalg.norm(side_a))
    len_b = float(np.linalg.norm(side_b))

    # `width` = gripper opening = shorter side (typical for Cornell).
    if len_a <= len_b:
        width = len_a
        height = len_b
        opening_vec = side_a
    else:
        width = len_b
        height = len_a
        opening_vec = side_b

    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    angle = math.atan2(float(opening_vec[1]), float(opening_vec[0]))
    # Wrap into [-pi/2, pi/2] — antipodal grasps are 180-symmetric.
    while angle > math.pi / 2:
        angle -= math.pi
    while angle < -math.pi / 2:
        angle += math.pi

    return GraspRect(x=cx, y=cy, angle_rad=angle, width=width, height=height)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def _load_splits() -> dict:
    if not _SPLITS_FILE.exists():
        raise FileNotFoundError(
            f"Cornell splits file not found at {_SPLITS_FILE}. "
            "Regenerate it from raw data (see scripts in this module)."
        )
    with _SPLITS_FILE.open() as fh:
        return json.load(fh)


def _select_ids(splits: dict, split_type: str, fold: int, partition: str) -> list[int]:
    if split_type == "image-wise":
        folds = splits["image_wise_folds"]
    elif split_type == "object-wise":
        folds = splits["object_wise_folds"]
    else:
        raise ValueError(f"Unknown split_type: {split_type}")

    if not (0 <= fold < len(folds)):
        raise ValueError(f"fold must be in 0..{len(folds) - 1}, got {fold}")

    test_ids = set(folds[str(fold)])
    if partition == "test":
        return sorted(test_ids)
    if partition == "train":
        return sorted(int(pid) for pid in splits["image_ids"] if pid not in test_ids)
    raise ValueError(f"partition must be 'train' or 'test', got {partition!r}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CornellDataset:
    """Cornell Grasping Dataset.

    Parameters
    ----------
    split : str, default "all"
        "all" returns every pcd ID in the dataset (folds ignored). Any
        other value is currently unused — keep for forward-compat with
        the loader contract.
    fold : int, default 0
        Fold index (0..4 for 5-fold CV).
    partition : str, default "train"
        "train" or "test". Ignored when split="all".
    root : str, default "<auto-detect>"
        Path to the directory containing folders 01..10. Auto-detects
        `<repo_root>/cornell comparisson/cornell_dataset/` if left as
        the sentinel.
    split_type : str, default "image-wise"
        "image-wise" or "object-wise".
    """

    def __init__(
        self,
        split: str = "all",
        fold: int = 0,
        partition: str = "train",
        root: str = "<auto-detect>",
        split_type: str = "image-wise",
    ) -> None:
        self.split = split
        self.fold = fold
        self.partition = partition
        self.split_type = split_type
        self.root = _auto_root(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Cornell root not found: {self.root}")

        splits = _load_splits()
        self._splits = splits
        self._object_map = {int(k): int(v) for k, v in splits["object_assignment"].items()}

        if split == "all":
            self.ids = sorted(int(p) for p in splits["image_ids"])
        else:
            self.ids = _select_ids(splits, split_type, fold, partition)

    # -- python protocol -----------------------------------------------------

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        if idx < 0:
            idx += len(self.ids)
        if not (0 <= idx < len(self.ids)):
            raise IndexError(idx)
        pcd_id = self.ids[idx]
        paths = _sample_paths(self.root, pcd_id)

        rgb = np.array(Image.open(paths["rgb"]).convert("RGB"), dtype=np.uint8)

        depth: Optional[np.ndarray] = None
        if paths["depth"].exists():
            depth_img = Image.open(paths["depth"])
            depth = np.array(depth_img, dtype=np.float32)

        grasps = _parse_cpos_file(paths["cpos"])

        return {
            "image": rgb,
            "depth_gt": depth,
            "grasps_gt": grasps,
            "object_id": self._object_map.get(pcd_id, -1),
            "sample_id": f"pcd{pcd_id:04d}",
        }


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds = CornellDataset(split="all")
    print(f"CornellDataset(split='all') size: {len(ds)}")
    sample = ds[0]
    img = sample["image"]
    depth = sample["depth_gt"]
    grasps = sample["grasps_gt"]
    print(f"  sample_id : {sample['sample_id']}")
    print(f"  object_id : {sample['object_id']}")
    print(f"  image     : shape={img.shape} dtype={img.dtype}")
    if depth is not None:
        print(
            f"  depth_gt  : shape={depth.shape} dtype={depth.dtype} "
            f"min={depth.min():.3f} max={depth.max():.3f}"
        )
    else:
        print("  depth_gt  : None")
    print(f"  grasps_gt : {len(grasps)} rectangles")
    if grasps:
        g = grasps[0]
        print(
            f"    first   : x={g.x:.1f} y={g.y:.1f} "
            f"angle={g.angle_rad:.3f}rad w={g.width:.1f} h={g.height:.1f}"
        )

    # Fold sanity
    train = CornellDataset(split="fold", fold=0, partition="train", split_type="image-wise")
    test = CornellDataset(split="fold", fold=0, partition="test", split_type="image-wise")
    print(f"image-wise fold 0: train={len(train)} test={len(test)} total={len(train) + len(test)}")

    train_obj = CornellDataset(split="fold", fold=0, partition="train", split_type="object-wise")
    test_obj = CornellDataset(split="fold", fold=0, partition="test", split_type="object-wise")
    print(
        f"object-wise fold 0: train={len(train_obj)} test={len(test_obj)} "
        f"total={len(train_obj) + len(test_obj)}"
    )
