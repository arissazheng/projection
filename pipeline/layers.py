"""Split one RGB-D image into a foreground layer and an inpainted background layer.

When the viewer moves, foreground objects slide relative to what is behind them and
uncover regions the photo never saw. We prepare for that offline:

1. Find depth discontinuities (edges where disparity jumps).
2. On the *near* side of each edge, within `radius` pixels, build a background
   layer by extending the farther surface inward: each pixel takes the depth and
   color of the farthest pixel in its neighborhood ("background extension").
3. Smooth the filled colors so the extension doesn't look streaky.

The viewer then renders the background mesh behind the foreground mesh, and drops
foreground triangles that straddle an edge, so the gap shows the background layer
instead of a rubber-sheet stretch.

This is the cheap, classical baseline. Upgrade path: replace `fill_color` with LaMa or
Stable Diffusion inpainting on the `hole` mask (and compare them in the writeup).
"""
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


def background_extension(rgb: np.ndarray, disp: np.ndarray, radius: int, stride: int = 2):
    """For every pixel, the color and disparity of the farthest pixel within `radius`."""
    best_disp = disp.copy()
    best_rgb = rgb.copy()
    pad = radius
    disp_p = np.pad(disp, pad, mode="edge")
    rgb_p = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    h, w = disp.shape
    for dy in range(-radius, radius + 1, stride):
        for dx in range(-radius, radius + 1, stride):
            if dx * dx + dy * dy > radius * radius:
                continue
            d = disp_p[pad + dy: pad + dy + h, pad + dx: pad + dx + w]
            farther = d < best_disp
            best_disp = np.where(farther, d, best_disp)
            best_rgb[farther] = rgb_p[pad + dy: pad + dy + h, pad + dx: pad + dx + w][farther]
    return best_rgb, best_disp


def build_layers(rgb: np.ndarray, disp: np.ndarray, edge_threshold: float = 0.04,
                 radius_frac: float = 0.04) -> Layers:
    h, w = disp.shape
    radius = max(3, int(round(radius_frac * max(h, w))))

    edges = depth_edges(disp, edge_threshold)
    near_edges = ndimage.binary_dilation(edges, iterations=radius)

    ext_rgb, ext_disp = background_extension(rgb, disp, radius)
    # Only the *near* side of an edge can hide anything. Receding walls/floors next to
    # an object also beat the threshold against the farthest neighbor, so require the
    # pixel to sit above the midpoint between the local near and far surfaces.
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
