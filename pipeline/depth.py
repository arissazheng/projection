"""Monocular depth with Depth Anything V2 (Hugging Face transformers port).

Output convention used everywhere in this repo: a float32 *normalized disparity*
map in [0, 1], where 1 = nearest and 0 = farthest. The viewer turns it into metric
distance with the scene's nearDist / farDist (see PLAN.md, "Coordinate conventions").
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def estimate_disparity(rgb: np.ndarray, model: str = "small") -> np.ndarray:
    """Return a normalized disparity map (H, W) in [0, 1] for an RGB uint8 image."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    name = MODELS.get(model, model)
    device = _device()
    processor = AutoImageProcessor.from_pretrained(name)
    net = AutoModelForDepthEstimation.from_pretrained(name).to(device).eval()

    inputs = processor(images=Image.fromarray(rgb), return_tensors="pt").to(device)
    with torch.no_grad():
        raw = net(**inputs).predicted_depth  # relative models predict disparity-like values
    raw = raw.squeeze().float().cpu().numpy()

    h, w = rgb.shape[:2]
    raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_CUBIC)
    return normalize_disparity(raw)


def normalize_disparity(disp: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Robustly rescale to [0, 1] (percentile clip so a few outliers don't squash the range)."""
    lo, hi = np.percentile(disp, [lo_pct, hi_pct])
    out = (disp - lo) / max(hi - lo, 1e-6)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
