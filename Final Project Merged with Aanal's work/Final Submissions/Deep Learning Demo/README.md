# Heatmap Stability Demo

Interactive demo to visualize and tune heatmap stability parameters.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python server.py
```

Then open http://localhost:5000 in your browser.

## Required Files

1. **Mesh** (.obj) - 3D mesh of the object
2. **Color Image** (.png/.jpg) - RGB image
3. **Mask** (.npz/.npy/.png) - Segmentation mask
4. **Pose** (.npy) - 4x4 pose matrix (or array of poses)
5. **Intrinsics** (.yml/.yaml/.npy) - Camera intrinsics

Optional:
- **Mesh Scale** (.npy) - Scale factors [sx, sy, sz]

## Parameters

### Thresholds
- **Top-facing**: Vertices with normal·(-gravity) > threshold (default: 0.7)
- **Bottom-facing**: Vertices with normal·(-gravity) < threshold (default: -0.7)
- **Side-facing**: Vertices with |normal·(-gravity)| < threshold (default: 0.3)

### Weights (per surface type)
- **Height**: Preference for lower vertices
- **CoG**: Preference for vertices near center of gravity
- **Vertical**: Preference for vertices with normals aligned to gravity
- **Edge**: Preference for vertices near edges

### General
- **Num points**: How many top stable points to display
- **Gaussian sigma**: Smoothing for heatmap visualization
