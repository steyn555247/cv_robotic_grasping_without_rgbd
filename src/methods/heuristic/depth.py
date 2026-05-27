"""Depth estimator wrapper for the heuristic grasp detector.

Wraps HuggingFace's Depth-Anything-V2 (default: Small) as a stateless,
no-grad callable that returns a depth map normalised to ``[0, 1]`` at the
exact resolution of the input image.

The original Streamlit implementation also supported MiDaS variants and
larger Depth-Anything checkpoints (see ``DepthModelWrapper`` in
``Heuristics approach/grasp_detection_contour_80px.py``). The research
pipeline only needs Depth-Anything-V2-Small as the default, so this module
is intentionally minimal; swap ``model_name`` to use Base/Large.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DepthEstimator:
    """Frozen Depth-Anything-V2 wrapper.

    The model is loaded once at construction time and placed in ``.eval()``
    mode on the chosen device. Calling the estimator runs inference under
    ``torch.no_grad()`` and returns a normalised depth array.
    """

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str | torch.device | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_name = model_name

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        """Predict a normalised depth map.

        Parameters
        ----------
        image_rgb : np.ndarray
            HxWx3 uint8 RGB image.

        Returns
        -------
        np.ndarray
            HxW float32 depth map in [0, 1]. Same spatial size as input.
        """
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(
                f"image_rgb must be HxWx3, got shape {image_rgb.shape}"
            )
        if image_rgb.dtype != np.uint8:
            raise ValueError(
                f"image_rgb must be uint8, got dtype {image_rgb.dtype}"
            )

        h, w = image_rgb.shape[:2]
        pil = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        depth = self.model(**inputs).predicted_depth  # (1, h', w') or (h', w')
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        depth = F.interpolate(
            depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_np = depth.detach().cpu().numpy().astype(np.float32)
        # Min-max normalise to [0, 1] (matches the Streamlit pipeline).
        d_min = float(depth_np.min())
        d_max = float(depth_np.max())
        depth_np = (depth_np - d_min) / (d_max - d_min + 1e-8)
        return depth_np
