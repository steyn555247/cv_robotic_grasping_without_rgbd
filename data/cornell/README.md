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
| Canonical object count (per Jiang 2011) | **~240** (not derivable from local files — see caveat 3) |

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

1. **885 IS the canonical Cornell image count.** Verified 2026-05-25 by
   `lit-scout` against Jiang ICRA 2011 (origin), Redmon ICRA 2015
   (Sec. V), Morrison RSS 2018 (GG-CNN), and Pinto 2016. The "855"
   figure that appeared in some prior project notes was a transcription
   error. Our local 885 matches the canon exactly. See
   `paper/related_work.md` § "Dataset audit — Cornell canonical numbers".

2. **The 950-999 gap is a folder boundary, not withholding.** Folder
   `09/` contains pcd0900–pcd0949 (50 images); folder `10/` contains
   pcd1000–pcd1034 (35 images). The "missing" pcd0950–pcd0999 block
   simply doesn't exist in the original Cornell distribution and is
   undocumented (no published explanation found). Best interpretation:
   curation artifact — failed-QC images never released. Every prior
   Cornell-evaluating paper reports the full 885.

3. **No public `pcd_id → object_id` mapping.** Cornell contains
   ~240 unique objects (Jiang 2011), but the original distribution's
   `backgroundMapping.txt` is **not present in our local copy** and
   was never publicly released by Lenz/Redmon/Kumra. Standard practice
   (every Cornell paper since 2015): build a private partition.
   We defer this curation to the RA-L extension; the workshop will
   report image-wise 5-fold CV only (this is also what most prior
   papers report — Redmon 2015, GG-CNN 2018, most modern works).

   `object_assignment` in the current splits JSON is the mechanical
   gap-≥5 output and is degenerate (2 objects). **Do not use it
   for object-wise CV.** A future `cornell_v2.json` will fix this
   when an external mapping is sourced or a hand curation is done.

4. **Two files with NaN grasps.** `pcd0132cpos.txt` and `pcd0165cpos.txt`
   contain partial rectangles with `NaN NaN` corners. The loader
   silently drops any rectangle containing non-finite coordinates.
   (Also flagged by Nishida-Lab's published Cornell preprocessing notes.)

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
