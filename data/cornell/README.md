# Cornell Grasping Dataset — local copy

## Where the raw data lives

`cornell comparisson/cornell_dataset/` (repo-relative; ~12 GB).

The directory is **not** modified, copied, or symlinked into `data/cornell/raw/`
— we read it in place. The path is auto-detected by
`src/data/cornell_loader.py`.

Layout (one subfolder per hundred pcd IDs, plus an unused `backgrounds/`):

```
cornell_dataset/
    01/   pcd0100r.png   pcd0100d.tiff   pcd0100cpos.txt   pcd0100cneg.txt   pcd0100.txt
          pcd0101r.png   ...
    02/   ...
    ...
    10/   pcd1000r.png ... pcd1034r.png
    backgrounds/         (ignored — clutter-free table shots)
```

Per pcd ID:
- `pcdXXXXr.png`    — RGB, 640x480, PIL mode `RGB`.
- `pcdXXXXd.tiff`   — depth, float32 metres.
- `pcdXXXX.txt`     — point cloud (PCL ASCII), not used by the loader.
- `pcdXXXXcpos.txt` — positive grasps, 4 lines of "x y" per rectangle.
- `pcdXXXXcneg.txt` — negative grasps; we ignore them.

## Counts

| Thing | Value |
|---|---|
| Total `pcd*r.png` files | **885** |
| pcd ID range | 100 – 1034 |
| Missing IDs in range | 50 (the block 950–999 is absent) |
| pcd IDs with all four needed files (RGB, depth, cpos, pcl) | 885 / 885 |
| Total positive grasp rectangles (`cpos` lines / 4, NaN-filtered) | **5111** |
| Objects identified by gap-≥5 heuristic | **2** (see caveat below) |

## Splits — `src/data/splits/cornell.json`

Frozen at the values produced by seed=42. Once committed, this file is
immutable; if a different split is needed, create `cornell_v2.json` and
document why.

- `image_wise_folds` — 5-fold random partition of the 885 pcd IDs.
  Each fold has 177 samples. Standard image-wise CV.
- `object_wise_folds` — 5-fold partition of *objects*, then each pcd
  inherits its object's fold. Objects are derived by the
  "gap ≥ 5" heuristic over sorted pcd IDs.

`object_assignment` is a `pcd_id -> object_id` map embedded in the JSON.

SHA-256 of `src/data/splits/cornell.json`:
```
3089f8b9ab5f00e8276183cf4fe34a8b9276ec3da2d5def3c0a65875285f6a7a
```

## Data quirks discovered during loader build

1. **855 vs 885.** The Cornell paper canon is 855 images. This local
   copy contains **885** — 30 more samples than the canonical set.
   We use all 885; downstream comparisons to literature numbers should
   note the larger denominator.

2. **One large gap.** pcd IDs `950–999` (50 IDs) are absent. All other
   IDs in `[100, 1034]` are present and contiguous.

3. **Gap-≥5 heuristic yields only 2 objects.** Because the surviving
   pcd IDs are contiguous everywhere except the single 950–999 gap,
   the "consecutive-IDs-with-gap≥5-start-new-object" rule produces
   exactly two objects: `[100, 949]` and `[1000, 1034]`. The
   resulting object-wise folds are heavily imbalanced (fold 0: 35
   samples, fold 1: 850 samples, folds 2–4: empty). This is the
   correct mechanical output of the specified heuristic on this
   distribution of the data, not a bug in the loader.

   **Implication:** object-wise CV as currently encoded is not a
   useful generalisation test. Treat the image-wise folds as the
   primary evaluation protocol. If true object-wise CV is needed,
   commit a hand-curated or externally-sourced object-assignment
   file as `cornell_v2.json`.

4. **One file with NaN grasps.** `pcd0165cpos.txt` contains a partial
   rectangle whose first and fourth corners are `NaN NaN`. The
   loader silently drops any rectangle containing non-finite
   coordinates.

5. **Depth units.** `pcd*d.tiff` is float32 metres (typical range
   ~0.03–2.0). The loader returns it as `float32` without rescaling.

6. **RGB encoding.** PIL opens the `r.png` files in mode `RGB`. The
   loader returns `np.ndarray[H, W, 3]` in RGB order (not BGR).

## Reproducing the splits

The splits file is generated once and committed. To regenerate
identically (e.g. for audit), use seed=42 over the sorted set of
pcd IDs whose `r.png` is present. The exact procedure is encoded in
the build script embedded in this repo's git history at the commit
that first introduced `src/data/splits/cornell.json`.
