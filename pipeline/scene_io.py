"""Write a scene folder the browser viewer can load.

viewer/scenes/<name>/
  image.jpg          full-res photo (foreground texture)
  bg.jpg             inpainted background texture
  disparity.f32      float32 little-endian, grid_h * grid_w, row-major, 1 = near
  bg_disparity.f32   same, for the background layer
  meta.json          sizes + default depth mapping
  debug_*.png        visualizations (not loaded by the viewer)
viewer/scenes/index.json lists all scenes.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SCENES_DIR = REPO / "viewer" / "scenes"


def _resize(img: np.ndarray, long_side: int, interp=cv2.INTER_AREA) -> np.ndarray:
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=interp)


def write_scene(name: str, rgb: np.ndarray, disp: np.ndarray, bg_rgb: np.ndarray,
                bg_disp: np.ndarray, *, near_dist: float, far_dist: float,
                edge_threshold: float, ref_eye_z: float | None = None,
                debug: dict[str, np.ndarray] | None = None, max_texture: int = 2048) -> Path:
    out = SCENES_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    tex = _resize(rgb, max_texture)
    cv2.imwrite(str(out / "image.jpg"), cv2.cvtColor(tex, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out / "bg.jpg"), cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])

    grid_h, grid_w = disp.shape
    disp.astype("<f4").tofile(out / "disparity.f32")
    bg_disp.astype("<f4").tofile(out / "bg_disparity.f32")

    meta = {
        "name": name,
        "imageWidth": int(tex.shape[1]),
        "imageHeight": int(tex.shape[0]),
        "gridWidth": int(grid_w),
        "gridHeight": int(grid_h),
        # Distances (meters) from the reference eye to the nearest / farthest point.
        "nearDist": near_dist,
        "farDist": far_dist,
        "edgeThreshold": edge_threshold,
    }
    if ref_eye_z is not None:
        meta["refEyeZ"] = ref_eye_z
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    for key, img in (debug or {}).items():
        cv2.imwrite(str(out / f"debug_{key}.png"), img)

    _update_index()
    return out


def disparity_vis(disp: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap((np.clip(disp, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def _update_index() -> None:
    names = sorted(p.name for p in SCENES_DIR.iterdir() if (p / "meta.json").exists())
    (SCENES_DIR / "index.json").write_text(json.dumps({"scenes": names}, indent=2))
