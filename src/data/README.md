# `src/data/` — Dataset loaders + immutable splits

## Files

- `cornell_loader.py` — Cornell Grasping Dataset loader
- `jacquard_loader.py` — Jacquard loader
- `splits/cornell.json` — image-wise + object-wise 5-fold split assignments. **Sacred — never edit.**
- `splits/jacquard.json` — same, for Jacquard

## Loader contract

```python
class CornellDataset:
    def __init__(self, split: str, fold: int = 0, root: str = ...): ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx) -> dict:
        # {"image": HxWx3 uint8,
        #  "depth_gt": HxW float32 or None,
        #  "grasps_gt": list[GraspRect],
        #  "object_id": int,
        #  "sample_id": str}
```

## Splits

Splits are committed to git. They define reproducibility. If someone breaks them, every number in the paper becomes incomparable.

If a new split is genuinely needed: add `cornell_v2.json`, do NOT modify `cornell.json`.
