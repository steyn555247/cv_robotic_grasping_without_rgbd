# CV Robotic Grasping without RGBD

CIS 5810 Final Project — robotic grasp detection from RGB only (no depth sensor required).

## Layout

- **Heuristics approach/** — Contour-based heuristic grasp detector (`app_working.py`, `grasp_detection_*.py`).
- **cornell comparisson/** — Evaluation pipeline against the Cornell grasp dataset (notebooks + grid search).
- **Final Project Merged with Aanal's work/** — Final deliverables: report PDF, demo presentation, benchmark outputs.
- **Images/** — Reference images used by the report / demos.

## Excluded from the repo

The following are intentionally not tracked (see `.gitignore`):

- Python virtual environments (`venv/`, `.venv/`) — recreate with `pip install -r requirements.txt`.
- `cornell comparisson/cornell_dataset/` (~12 GB) — download separately from the [Cornell Grasping Dataset](https://www.kaggle.com/datasets/oneoneliu/cornell-grasp).
- Demo MP4s (>100 MB GitHub limit) — available out-of-band.

## Setup

```bash
cd "Heuristics approach"
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python app_working.py
```
