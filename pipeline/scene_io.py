"""Write textures, little-endian disparity grids, and metadata for the photo viewer."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SCENES_DIR = REPO / "viewer" / "scenes"


def resize_to_fit(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def write_scene(name: str, rgb: np.ndarray, disp: np.ndarray, bg_rgb: np.ndarray,
                bg_disp: np.ndarray, *, near_dist: float, far_dist: float,
                edge_threshold: float, debug: dict[str, np.ndarray] | None = None, max_texture: int = 2048) -> Path:
    out = SCENES_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    tex = resize_to_fit(rgb, max_texture)
    cv2.imwrite(str(out / "image.jpg"), cv2.cvtColor(tex, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out / "bg.jpg"), cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])

    grid_h, grid_w = disp.shape
    disp.astype("<f4").tofile(out / "disparity.f32")
    bg_disp.astype("<f4").tofile(out / "bg_disparity.f32")

    meta = {
        "name": name,
        "imageWidth": tex.shape[1],
        "imageHeight": tex.shape[0],
        "gridWidth": grid_w,
        "gridHeight": grid_h,
        # Distances (meters) from the reference eye to the nearest / farthest point.
        "nearDist": near_dist,
        "farDist": far_dist,
        "edgeThreshold": edge_threshold,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for key, img in (debug or {}).items():
        cv2.imwrite(str(out / f"debug_{key}.png"), img)

    names = sorted(path.parent.name for path in SCENES_DIR.glob("*/meta.json"))
    (SCENES_DIR / "index.json").write_text(json.dumps({"scenes": names}, indent=2), encoding="utf-8")
    return out


def disparity_vis(disp: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap((np.clip(disp, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
