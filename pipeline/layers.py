"""Extend background surfaces behind depth edges in a single photo."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage


@dataclass
class Layers:
    bg_rgb: np.ndarray     # (H, W, 3) uint8
    bg_disp: np.ndarray    # (H, W) float32, normalized disparity
    edges: np.ndarray      # (H, W) bool, depth discontinuities
    hole: np.ndarray       # (H, W) bool, pixels where the background differs from the photo


def depth_edges(disp: np.ndarray, threshold: float, window: int = 5) -> np.ndarray:
    """Pixels whose local disparity range exceeds `threshold` (a jump, not a slope)."""
    local_range = ndimage.maximum_filter(disp, window) - ndimage.minimum_filter(disp, window)
    return local_range > threshold


def background_extension(rgb: np.ndarray, disp: np.ndarray, radius: int,
                         stride: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Find the color and disparity of the farthest sampled neighbor of each pixel."""
    best_disp = disp.copy()
    best_rgb = rgb.copy()
    padded_disp = np.pad(disp, radius, mode="edge")
    padded_rgb = np.pad(rgb, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    h, w = disp.shape
    for dy in range(-radius, radius + 1, stride):
        for dx in range(-radius, radius + 1, stride):
            if dx * dx + dy * dy > radius * radius:
                continue
            rows = slice(radius + dy, radius + dy + h)
            cols = slice(radius + dx, radius + dx + w)
            candidate = padded_disp[rows, cols]
            farther = candidate < best_disp
            best_disp[farther] = candidate[farther]
            best_rgb[farther] = padded_rgb[rows, cols][farther]
    return best_rgb, best_disp


def build_layers(rgb: np.ndarray, disp: np.ndarray, edge_threshold: float = 0.04,
                 radius_frac: float = 0.04) -> Layers:
    h, w = disp.shape
    radius = max(3, int(round(radius_frac * max(h, w))))

    edges = depth_edges(disp, edge_threshold)
    near_edges = ndimage.binary_dilation(edges, iterations=radius)

    ext_rgb, ext_disp = background_extension(rgb, disp, radius)
    # Fill the near side of each edge, where shifting the viewer can uncover the background.
    win = 2 * radius + 1
    mid = 0.5 * (ndimage.maximum_filter(disp, win) + ndimage.minimum_filter(disp, win))
    hole = near_edges & (disp > mid) & (disp - ext_disp > edge_threshold)

    bg_disp = np.where(hole, ext_disp, disp).astype(np.float32)
    # Median-smooth the filled depth so the background surface isn't jagged.
    bg_disp_smooth = ndimage.median_filter(bg_disp, size=5)
    bg_disp = np.where(hole, bg_disp_smooth, bg_disp).astype(np.float32)

    bg_rgb = fill_color(rgb, ext_rgb, hole)
    return Layers(bg_rgb=bg_rgb, bg_disp=bg_disp, edges=edges, hole=hole)


def fill_color(rgb: np.ndarray, ext_rgb: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Background extension colors, softened, then feathered into the original photo."""
    filled = np.where(hole[..., None], ext_rgb, rgb)
    blurred = cv2.GaussianBlur(filled, (0, 0), sigmaX=3)
    alpha = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), sigmaX=2)[..., None]
    out = filled * (1 - alpha) + blurred * alpha
    return np.clip(out, 0, 255).astype(np.uint8)
