"""Which viewing directions around the start camera actually contain scene data.

The gesture bridge uses this to stop the palm from turning the camera toward areas
the capture never saw, where the viewer shows only black.

From the start camera, every reasonably opaque splat gets a horizontal angle (yaw,
positive = to the right) and a vertical angle (pitch, positive = up) relative to the
start view direction. The camera may point in any direction, continuing outward from
straight ahead, as long as at least half of its field of view still contains data.
That lets it turn past narrow gaps (bare wall between two objects) but not into
empty space.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

BIN_DEGREES = 5
MIN_BIN_SHARE = 0.1  # a bin counts as covered with at least 10% of a typical filled bin's weight
MIN_OPACITY = 0.05  # ignore nearly invisible splats
MIN_VIEW_FILLED = 0.5  # share of the view that must show scene data


def read_positions_and_opacity(ply: Path) -> tuple[np.ndarray, np.ndarray]:
    with ply.open('rb') as f:
        names, count = [], 0
        while (line := f.readline().decode('ascii', 'replace').strip()) != 'end_header':
            parts = line.split()
            if parts[:2] == ['element', 'vertex']:
                count = int(parts[2])
            elif parts[:2] == ['property', 'float']:
                names.append(parts[2])
        data = np.fromfile(f, dtype=np.dtype([(n, '<f4') for n in names]), count=count)
    xyz = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float64)
    opacity = 1 / (1 + np.exp(-data['opacity'].astype(np.float64)))  # stored as a logit
    return xyz, opacity


def allowed_range(angles: np.ndarray, weights: np.ndarray, limit: float,
                  half_view: float) -> tuple[float, float]:
    """Range of view-center angles (degrees) around 0 whose view is mostly filled."""
    edges = np.arange(-limit, limit + BIN_DEGREES, BIN_DEGREES)
    centers = (edges[:-1] + edges[1:]) / 2
    counts, _ = np.histogram(angles, bins=edges, weights=weights)
    if counts.max() == 0:
        return 0.0, 0.0
    # Compare with a typical filled bin, not the fullest one, so a single dense
    # spot does not make the rest of the room look empty.
    filled = (counts >= MIN_BIN_SHARE * np.median(counts[counts > 0])).astype(float)
    window = max(1, int(round(half_view / BIN_DEGREES)))
    share = np.convolve(filled, np.ones(2 * window + 1) / (2 * window + 1), mode='same')
    ok = share >= MIN_VIEW_FILLED
    center = int(np.argmin(np.abs(centers)))
    if not ok[center]:
        return 0.0, 0.0
    lo = hi = center
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(ok) - 1 and ok[hi + 1]:
        hi += 1
    return float(min(0.0, centers[lo])), float(max(0.0, centers[hi]))


def compute_coverage(ply: Path, settings: Path) -> dict:
    camera = json.loads(settings.read_text(encoding='utf-8'))['cameras'][0]['initial']
    position = np.array(camera['position'])
    forward = np.array(camera['target']) - position
    fov = float(camera['fov'])

    xyz, opacity = read_positions_and_opacity(ply)
    # SuperSplat flips the PLY's x and y axes when it loads the scene.
    keep = opacity > MIN_OPACITY
    xyz, weight = xyz[keep] * np.array([-1.0, -1.0, 1.0]), opacity[keep]
    up = np.array([0.0, 1.0, 0.0])
    flat = forward - up * forward.dot(up)
    flat /= np.linalg.norm(flat)
    right = np.cross(flat, up)

    v = xyz - position
    yaw = np.degrees(np.arctan2(v @ right, v @ flat))
    pitch = np.degrees(np.arctan2(v @ up, np.hypot(v @ right, v @ flat)))
    start_pitch = math.degrees(math.atan2(forward.dot(up), np.linalg.norm(flat * np.linalg.norm(forward))))

    # The viewer's fov is vertical; on a landscape display the horizontal view is wider,
    # so the vertical value is the conservative choice for both.
    yaw_lo, yaw_hi = allowed_range(yaw, weight, 180, fov / 2)
    ahead = (yaw > yaw_lo - fov / 2) & (yaw < yaw_hi + fov / 2)  # pitch within reachable yaw
    pitch_lo, pitch_hi = allowed_range(pitch[ahead] - start_pitch, weight[ahead], 90, fov / 2)
    # Degrees relative to the start view: positive yaw = right, positive pitch = up.
    return {'yaw': [yaw_lo, yaw_hi], 'pitch': [pitch_lo, pitch_hi], 'fov': fov}


def cached_coverage(result: Path) -> dict:
    cache = result / 'coverage.json'
    if cache.is_file() and cache.stat().st_mtime >= (result / 'scene.ply').stat().st_mtime:
        return json.loads(cache.read_text(encoding='utf-8'))
    coverage = compute_coverage(result / 'scene.ply', result / 'settings.json')
    cache.write_text(json.dumps(coverage), encoding='utf-8')
    return coverage
