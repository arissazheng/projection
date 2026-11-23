"""Colab runner for Room Window: the repo's own frame extraction + COLMAP alignment,
with Brush's training step replaced by gsplat (CUDA). Produces the same scene folder
(scene.ply + settings.json) the repo's walkthrough server serves.

Usage: python colab_room.py VIDEO --repo /content/projection --out /content/scene [--steps 7000]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

LOG_FILE = Path('/content/progress.log') if Path('/content').exists() else Path('progress.log')


def log(msg: str) -> None:
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with LOG_FILE.open('a') as f:
        f.write(line + '\n')


# ---------------------------------------------------------------- data loading

def load_dataset(training: Path):
    """Read the undistorted, oriented COLMAP model written by the repo pipeline."""
    import pycolmap
    from PIL import Image

    model = pycolmap.Reconstruction(training / 'sparse')
    images, viewmats, Ks = [], [], []
    for image_id in sorted(model.reg_image_ids(), key=lambda i: model.image(i).name):
        image = model.image(image_id)
        camera = model.camera(image.camera_id)
        path = training / 'images' / image.name
        if not path.is_file():
            continue
        with Image.open(path) as source:
            rgb = np.asarray(source.convert('RGB'), dtype=np.uint8)
        # Undistorted images may be resized relative to the camera; rescale K to match.
        K = np.array(camera.calibration_matrix(), dtype=np.float64)
        sx, sy = rgb.shape[1] / camera.width, rgb.shape[0] / camera.height
        K[0] *= sx
        K[1] *= sy
        w2c = np.eye(4)
        w2c[:3, :4] = np.array(image.cam_from_world().matrix())
        images.append(rgb)
        viewmats.append(w2c)
        Ks.append(K)

    points = np.array([p.xyz for p in model.points3D.values()], dtype=np.float32)
    colors = np.array([p.color for p in model.points3D.values()], dtype=np.float32) / 255.0
    if len(images) < 3 or len(points) < 10:
        raise ValueError(f'Too little data to train: {len(images)} images, {len(points)} points.')
    return images, np.array(viewmats, np.float32), np.array(Ks, np.float32), points, colors


def knn_scale(points: np.ndarray, k: int = 3) -> np.ndarray:
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(points).query(points, k=k + 1)
    mean = np.sqrt((dist[:, 1:] ** 2).mean(axis=1))
    return np.clip(mean, 1e-4, None)


# ---------------------------------------------------------------- PLY export

SH_C0 = 0.28209479177387814


def write_ply(path: Path, means, scales, quats, opacities, sh0, shN) -> None:
    """Standard 3DGS PLY (what Brush/INRIA/SuperSplat use)."""
    n = means.shape[0]
    f_dc = sh0.reshape(n, 3)
    f_rest = shN.transpose(0, 2, 1).reshape(n, -1)  # channel-major, like INRIA
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    fields = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    fields += [f'f_dc_{i}' for i in range(3)] + [f'f_rest_{i}' for i in range(f_rest.shape[1])]
    fields += ['opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
    data = np.concatenate([means, np.zeros_like(means), f_dc, f_rest, opacities.reshape(n, 1), scales, quats],
                          axis=1).astype('<f4')
    header = 'ply\nformat binary_little_endian 1.0\n' + f'element vertex {n}\n'
    header += ''.join(f'property float {name}\n' for name in fields) + 'end_header\n'
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(np.ascontiguousarray(data).tobytes())


# ---------------------------------------------------------------- training

def ssim(a, b):
    import torch
    import torch.nn.functional as F

    window = 11
    coords = torch.arange(window, dtype=a.dtype, device=a.device) - window // 2
    g = torch.exp(-coords ** 2 / (2 * 1.5 ** 2))
    g = (g / g.sum())
    kernel = (g[:, None] @ g[None, :]).expand(a.shape[1], 1, window, window).contiguous()
    conv = lambda x: F.conv2d(x, kernel, padding=window // 2, groups=a.shape[1])
    mu_a, mu_b = conv(a), conv(b)
    s_a = conv(a * a) - mu_a ** 2
    s_b = conv(b * b) - mu_b ** 2
    s_ab = conv(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    value = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_a + s_b + c2))
    return value.mean()


def train(training: Path, output: Path, steps: int, sh_degree: int = 3, max_splats: int = 1_500_000,
          device: str = 'cuda') -> int:
    import torch
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy

    device = torch.device(device)
    images, viewmats, Ks, points, colors = load_dataset(training)
    log(f'Training on {len(images)} images, {len(points)} starting points, {steps} steps')

    cam_centers = np.linalg.inv(viewmats)[:, :3, 3]
    scene_scale = float(np.linalg.norm(cam_centers - cam_centers.mean(0), axis=1).max()) * 1.1
    scene_scale = max(scene_scale, 1e-3)

    gt = [torch.from_numpy(img).to(device).float().div_(255.0) for img in images]
    viewmats_t = torch.from_numpy(viewmats).to(device)
    Ks_t = torch.from_numpy(Ks).to(device)

    n = len(points)
    num_sh = (sh_degree + 1) ** 2
    rgb = torch.from_numpy(colors)
    params = {
        'means': torch.from_numpy(points),
        'scales': torch.log(torch.from_numpy(knn_scale(points)).float()).unsqueeze(1).repeat(1, 3),
        'quats': torch.nn.functional.normalize(torch.rand(n, 4), dim=-1),
        'opacities': torch.logit(torch.full((n,), 0.1)),
        'sh0': ((rgb - 0.5) / SH_C0).unsqueeze(1),
        'shN': torch.zeros(n, num_sh - 1, 3),
    }
    splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v.float().to(device)) for k, v in params.items()})
    lrs = {'means': 1.6e-4 * scene_scale, 'scales': 5e-3, 'quats': 1e-3, 'opacities': 5e-2,
           'sh0': 2.5e-3, 'shN': 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{'params': splats[k], 'lr': lr, 'name': k}], eps=1e-15)
                  for k, lr in lrs.items()}
    means_sched = torch.optim.lr_scheduler.ExponentialLR(optimizers['means'], gamma=0.01 ** (1.0 / steps))

    strategy = DefaultStrategy(refine_start_iter=500, refine_stop_iter=int(steps * 0.8), refine_every=100,
                               reset_every=3000, verbose=False)
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    rng = np.random.default_rng(0)
    order = []
    start = time.time()
    for step in range(steps):
        if not order:
            order = list(rng.permutation(len(gt)))
        i = order.pop()
        target = gt[i]
        h, w = target.shape[:2]
        renders, _, info = rasterization(
            means=splats['means'], quats=splats['quats'], scales=torch.exp(splats['scales']),
            opacities=torch.sigmoid(splats['opacities']),
            colors=torch.cat([splats['sh0'], splats['shN']], 1),
            viewmats=viewmats_t[i:i + 1], Ks=Ks_t[i:i + 1], width=w, height=h,
            sh_degree=min(step // 1000, sh_degree), packed=False,
        )
        render = renders[0]
        strategy.step_pre_backward(splats, optimizers, state, step, info)
        l1 = (render - target).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1 - ssim(render.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]))
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_sched.step()
        # Stop growing once the splat budget is used up (keeps T4 memory safe).
        if len(splats['means']) > max_splats and strategy.refine_stop_iter > step:
            strategy.refine_stop_iter = step
            log(f'Reached {max_splats} splats; stopping densification at step {step}')
        strategy.step_post_backward(splats, optimizers, state, step, info, packed=False)

        if step % 500 == 0 or step == steps - 1:
            psnr = -10 * math.log10(max(((render - target) ** 2).mean().item(), 1e-10))
            elapsed = time.time() - start
            eta = elapsed / (step + 1) * (steps - step - 1)
            log(f'step {step + 1}/{steps}  loss {loss.item():.4f}  psnr {psnr:.1f}  '
                f'splats {len(splats["means"]):,}  eta {eta / 60:.1f} min')

    output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        write_ply(output / 'scene.ply', *(splats[k].detach().cpu().numpy() for k in
                                         ('means', 'scales', 'quats', 'opacities', 'sh0', 'shN')))
    return len(splats['means'])


# ---------------------------------------------------------------- full pipeline

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', help='One video, or 12-500 photos')
    parser.add_argument('--repo', default='/content/projection')
    parser.add_argument('--out', default='/content/scene')
    parser.add_argument('--work', default='/content/work')
    parser.add_argument('--steps', type=int, default=7000)
    parser.add_argument('--align-only', action='store_true')
    args = parser.parse_args()

    sys.path.insert(0, args.repo)
    from pipeline import reconstruct_room as rr

    paths = [Path(p) for p in args.inputs]
    work, out = Path(args.work), Path(args.out)
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    work.mkdir(parents=True)
    dataset = work / 'dataset'
    dataset.mkdir()

    t0 = time.time()
    kind = rr.input_kind(paths)
    report = lambda state, message: log(f'{state}: {message}')
    report('preparing', 'Extracting video frames…' if kind == 'video' else 'Preparing photos…')
    count = rr.prepare_images(paths, kind, dataset / 'images')
    log(f'{count} frames ready')
    model = rr.align_images(dataset, dataset / 'images', kind, count, report)
    log(f'COLMAP aligned {model.num_reg_images()} of {count} images, {model.num_points3D()} points')
    report('preparing', 'Correcting lens distortion in the aligned images…')
    training, model = rr.prepare_training_data(model, dataset / 'images', dataset)
    registered = model.num_reg_images()
    camera = rr.orient_model(model)
    model.write(training / 'sparse')
    log(f'Alignment finished in {(time.time() - t0) / 60:.1f} min')
    (work / 'camera.json').write_text(json.dumps({'camera': camera, 'registered': registered, 'count': count, 'training': str(training)}))
    if args.align_only:
        return

    t1 = time.time()
    n = train(training, out, args.steps)
    log(f'Training finished in {(time.time() - t1) / 60:.1f} min, {n:,} splats')

    # Same settings.json the repo's walkthrough server writes for SuperSplat.
    from pipeline import walkthrough as wt
    scene, base = wt.create_scene(paths[0].stem if kind == 'video' else 'Room photos')
    shutil.copytree(out, scene / 'result_tmp')
    wt.publish_scene(scene, base, scene / 'result_tmp', camera, registered, count)
    (out / 'settings.json').write_text((scene / 'result' / 'settings.json').read_text())
    (out / 'scene_id.txt').write_text(scene.name)
    log(f'DONE scene_id={scene.name}')


if __name__ == '__main__':
    main()
