"""Photo -> scene folder for the viewer.

    python -m pipeline.prepare_scene photos/our_room.jpg --name our_room

Then reload the viewer and pick the scene from the dropdown.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from .depth import estimate_disparity
from .layers import build_layers
from .scene_io import _resize, disparity_vis, write_scene


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo", type=Path)
    ap.add_argument("--name", help="scene name (default: photo filename)")
    ap.add_argument("--model", default="small", help="small | base | large | HF model id")
    ap.add_argument("--work-size", type=int, default=768,
                    help="long side (px) for depth/layers; also the mesh grid resolution")
    ap.add_argument("--near", type=float, default=1.0, help="meters from eye to nearest point")
    ap.add_argument("--far", type=float, default=6.0, help="meters from eye to farthest point")
    ap.add_argument("--edge-threshold", type=float, default=0.04,
                    help="normalized-disparity jump that counts as an object edge")
    args = ap.parse_args()

    bgr = cv2.imread(str(args.photo))
    if bgr is None:
        raise SystemExit(f"could not read {args.photo}")
    rgb_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = _resize(rgb_full, args.work_size)

    t = time.perf_counter()
    disp = estimate_disparity(rgb, args.model)
    print(f"depth ({args.model}): {time.perf_counter() - t:.1f}s  grid {disp.shape[1]}x{disp.shape[0]}")

    t = time.perf_counter()
    layers = build_layers(rgb, disp, edge_threshold=args.edge_threshold)
    print(f"layers: {time.perf_counter() - t:.1f}s  hole pixels {layers.hole.mean() * 100:.1f}%")

    hole_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    hole_vis[layers.hole] = (0.4 * hole_vis[layers.hole] + [0, 0, 150]).astype(np.uint8)
    hole_vis[layers.edges] = (0, 255, 255)

    out = write_scene(
        args.name or args.photo.stem, rgb_full, disp, layers.bg_rgb, layers.bg_disp,
        near_dist=args.near, far_dist=args.far, edge_threshold=args.edge_threshold,
        debug={"disparity": disparity_vis(disp), "holes": hole_vis,
               "bg_disparity": disparity_vis(layers.bg_disp)},
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
