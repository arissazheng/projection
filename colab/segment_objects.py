"""Find interactable objects in a trained room scene.

SAM 3 segments + tracks text-prompted objects across the aligned video frames, then
depth-checked voting assigns each Gaussian splat to an object (or background).

Usage: python segment_objects.py --work /content/work --scene /content/scene/scene.ply --out /content/objects
Outputs: objects.json, labels.npy, object_XX_<label>.ply, scene_without_objects.ply, preview.jpg
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from colab_room import load_dataset, log  # noqa: E402

DEFAULT_PROMPTS = ('mug, cup, bottle, bowl, plate, book, laptop, keyboard, phone, remote control, lamp, '
                   'pillow, blanket, chair, stool, backpack, bag, box, basket, plant, vase, picture frame, '
                   'clock, speaker, shoe, toy')

PLY_TYPES = {'float': 'f4', 'float32': 'f4', 'double': 'f8', 'uchar': 'u1', 'uint8': 'u1', 'char': 'i1',
             'short': 'i2', 'ushort': 'u2', 'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4'}


# ---------------------------------------------------------------- PLY I/O (keeps every property)

def read_ply(path: Path):
    data = path.read_bytes()
    end = data.index(b'end_header\n') + len(b'end_header\n')
    lines = data[:end].decode('ascii').splitlines()
    if 'format binary_little_endian 1.0' not in lines:
        raise ValueError(f'{path.name}: only binary little-endian PLY is supported')
    count, fields, in_vertex = 0, [], False
    for line in lines:
        parts = line.split()
        if parts[:2] == ['element', 'vertex']:
            count, in_vertex = int(parts[2]), True
        elif parts[:1] == ['element']:
            in_vertex = False
        elif parts[:1] == ['property'] and in_vertex:
            fields.append((parts[2], '<' + PLY_TYPES[parts[1]]))
    rows = np.frombuffer(data, dtype=np.dtype(fields), count=count, offset=end)
    return rows


def write_ply(path: Path, rows: np.ndarray) -> None:
    inverse = {v: k for k, v in PLY_TYPES.items() if k in ('float', 'double', 'uchar', 'char', 'short', 'ushort', 'int', 'uint')}
    header = ['ply', 'format binary_little_endian 1.0', f'element vertex {len(rows)}']
    header += [f'property {inverse[rows.dtype[name].str[1:]]} {name}' for name in rows.dtype.names]
    with open(path, 'wb') as f:
        f.write(('\n'.join(header) + '\nend_header\n').encode('ascii'))
        f.write(np.ascontiguousarray(rows).tobytes())


# ---------------------------------------------------------------- SAM 3

def run_sam3(frames: list[np.ndarray], prompts: list[str], dtype_name: str, window: int = 40, overlap: int = 8):
    """Segment in overlapping windows (flat GPU memory). Returns per-frame int32 label maps (-1 = none),
    per-track info, and 2D-overlap merge pairs (same object seen twice, within or across windows)."""
    import torch
    from transformers import Sam3VideoModel, Sam3VideoProcessor

    dtype = {'float16': torch.float16, 'float32': torch.float32}[dtype_name]
    model = Sam3VideoModel.from_pretrained('facebook/sam3').to('cuda', dtype=dtype).eval()
    processor = Sam3VideoProcessor.from_pretrained('facebook/sam3')

    n = len(frames)
    h, w = frames[0].shape[:2]
    label_maps = [None] * n
    tracks: dict[int, dict] = {}
    pair_iou: dict[tuple[int, int], float] = {}
    stride = max(window - overlap, 1)
    starts = list(range(0, max(n - overlap, 1), stride))

    for wi, start in enumerate(starts):
        end = min(start + window, n)
        base = wi * 100000  # keeps track ids unique across windows
        session = processor.init_video_session(video=np.stack(frames[start:end]), inference_device='cuda',
                                               processing_device='cpu', video_storage_device='cpu', dtype=dtype)
        session = processor.add_text_prompt(inference_session=session, text=prompts)
        local: dict[int, np.ndarray] = {}
        with torch.inference_mode():
            for out in model.propagate_in_video_iterator(inference_session=session):
                res = processor.postprocess_outputs(session, out)
                i = start + out.frame_idx
                ids = [base + int(x) for x in res['object_ids'].tolist()]
                scores = res['scores'].float()
                if not torch.isfinite(scores).all():
                    raise FloatingPointError('non-finite SAM 3 scores')
                masks = res['masks']
                if masks.ndim == 4:
                    masks = masks[:, 0]
                masks = masks.bool()
                if masks.shape[-2:] != (h, w) and len(ids):
                    masks = torch.nn.functional.interpolate(masks[:, None].float(), size=(h, w), mode='nearest')[:, 0].bool()
                prompt_of = {base + int(oid): text for text, oids in res.get('prompt_to_obj_ids', {}).items() for oid in oids}

                labels = torch.full((h, w), -1, dtype=torch.int32, device=masks.device)
                areas = masks.flatten(1).sum(1) if len(ids) else torch.zeros(0)
                for k in torch.argsort(areas, descending=True).tolist():  # smaller (nested) objects win overlaps
                    if areas[k] == 0:
                        continue
                    labels[masks[k]] = ids[k]
                    t = tracks.setdefault(ids[k], {'prompt': prompt_of.get(ids[k], '?'), 'score': 0.0, 'frames': 0,
                                                   'best_frame': i, 'best_area': 0, 'max_area': 0})
                    t['score'] = max(t['score'], float(scores[k]))
                    t['frames'] += 1
                    t['max_area'] = max(t['max_area'], int(areas[k]))
                    if areas[k] > t['best_area']:
                        t['best_frame'], t['best_area'] = i, int(areas[k])
                # Same physical object found by two prompts (e.g. cup + mug) -> high mask overlap
                if len(ids) > 1:
                    flat = masks.flatten(1).float()
                    inter = flat @ flat.T
                    union = areas[:, None] + areas[None, :] - inter
                    iou = (inter / union.clamp(min=1)).cpu().numpy()
                    for a in range(len(ids)):
                        for b in range(a + 1, len(ids)):
                            if iou[a, b] > 0.5:
                                key = (min(ids[a], ids[b]), max(ids[a], ids[b]))
                                pair_iou[key] = max(pair_iou.get(key, 0.0), float(iou[a, b]))
                local[i] = labels.cpu().numpy()
                if i % 20 == 0:
                    log(f'SAM 3 frame {i + 1}/{n} (window {wi + 1}/{len(starts)}): {len(ids)} objects visible, {len(tracks)} tracks so far')
        del session
        torch.cuda.empty_cache()

        for i, lab in local.items():
            prev = label_maps[i]
            if prev is None:
                label_maps[i] = lab
                continue
            # Overlap frame: link tracks across windows by mask IoU; keep the earlier window's map.
            link_windows(prev, lab, pair_iou)

    del model
    torch.cuda.empty_cache()
    for i, m in enumerate(label_maps):
        if m is None:
            label_maps[i] = np.full((h, w), -1, np.int32)
    return label_maps, tracks, pair_iou


def link_windows(prev: np.ndarray, new: np.ndarray, pair_iou: dict, thresh: float = 0.5) -> None:
    """Record (prev_id, new_id) pairs whose masks overlap with IoU > thresh in the same frame."""
    pids = [p for p in np.unique(prev) if p >= 0]
    nids = [q for q in np.unique(new) if q >= 0]
    for p in pids:
        pm = prev == p
        pa = pm.sum()
        for q in nids:
            qm = new == q
            inter = np.count_nonzero(pm & qm)
            if inter == 0:
                continue
            iou = inter / (pa + qm.sum() - inter)
            if iou > thresh:
                key = (int(min(p, q)), int(max(p, q)))
                pair_iou[key] = max(pair_iou.get(key, 0.0), float(iou))


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(b)] = self.find(a)


# ---------------------------------------------------------------- 3D voting

def splat_params(rows):
    def stack(prefix, n):
        return np.stack([rows[f'{prefix}{i}'] for i in range(n)], 1).astype(np.float32)
    means = np.stack([rows['x'], rows['y'], rows['z']], 1).astype(np.float32)
    return means, stack('scale_', 3), stack('rot_', 4), rows['opacity'].astype(np.float32)


def render_depth(gs, viewmat, K, w, h):
    import torch
    from gsplat import rasterization
    means, log_scales, quats, opacity_logit = gs
    depth, alpha, _ = rasterization(
        means=means, quats=torch.nn.functional.normalize(quats, dim=-1), scales=torch.exp(log_scales),
        opacities=torch.sigmoid(opacity_logit), colors=torch.zeros_like(means),
        viewmats=viewmat[None], Ks=K[None], width=w, height=h, render_mode='ED', packed=False)
    d = depth[0, ..., 0]
    return torch.where(alpha[0, ..., 0] > 0.5, d, torch.full_like(d, float('inf')))


def vote(means_np, gs_np, viewmats, Ks, label_maps, track_index, k, depth_tol):
    """Each splat votes in every frame where its center is the visible surface. Returns votes + views."""
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    means = torch.from_numpy(means_np).to(dev)
    gs = tuple(torch.from_numpy(a).to(dev) for a in gs_np) if gs_np is not None else None
    n = len(means)
    seen = torch.zeros(n, dtype=torch.int32, device=dev)
    keys = []
    max_id = max(int(m.max()) for m in label_maps) if label_maps else 0
    lut = torch.full((max(max_id, max(track_index, default=0)) + 2,), -1, dtype=torch.int64, device=dev)
    for tid, idx in track_index.items():
        lut[tid] = idx
    ones = torch.ones(n, 1, device=dev)
    for f, labels_np in enumerate(label_maps):
        h, w = labels_np.shape
        V = torch.from_numpy(viewmats[f]).to(dev)
        K = torch.from_numpy(Ks[f]).to(dev)
        cam = torch.cat([means, ones], 1) @ V.T
        z = cam[:, 2]
        uv = cam[:, :3] @ K.T
        u = (uv[:, 0] / z.clamp(min=1e-6)).long()
        v = (uv[:, 1] / z.clamp(min=1e-6)).long()
        ok = (z > 1e-3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        depth = render_depth(gs, V, K, w, h)
        idx = torch.nonzero(ok, as_tuple=True)[0]
        surf = depth[v[idx], u[idx]]
        visible = idx[torch.isfinite(surf) & ((z[idx] - surf).abs() <= depth_tol * surf + 1e-3)]
        seen[visible] += 1
        lab = torch.from_numpy(labels_np.astype(np.int64)).to(dev)
        tid = lab[v[visible], u[visible]]
        hit = tid >= 0
        obj = lut[tid[hit]]
        keep = obj >= 0
        keys.append(visible[hit][keep] * k + obj[keep])
    keys = torch.cat(keys) if keys else torch.zeros(0, dtype=torch.int64, device=dev)
    uniq, counts = torch.unique(keys, return_counts=True)
    return uniq.cpu().numpy(), counts.cpu().numpy(), seen.cpu().numpy()


def assign(uniq, counts, seen, n, k, min_views, min_fraction):
    splat = uniq // k
    obj = uniq % k
    order = np.lexsort((-counts, splat))  # per splat, highest vote first
    splat, obj, counts = splat[order], obj[order], counts[order]
    first = np.ones(len(splat), bool)
    first[1:] = splat[1:] != splat[:-1]
    s, o, c = splat[first], obj[first], counts[first]
    good = (c >= min_views) & (c >= min_fraction * seen[s])
    labels = np.full(n, -1, np.int32)
    labels[s[good]] = o[good]
    return labels


def robust_box(points):
    lo, hi = np.percentile(points, [2, 98], axis=0)
    return lo, hi


def box_overlap(a, b):
    """(IoU, intersection / smaller box) of two axis-aligned boxes."""
    lo, hi = np.maximum(a[0], b[0]), np.minimum(a[1], b[1])
    inter = np.prod(np.clip(hi - lo, 0, None))
    vol = lambda box: np.prod(np.clip(box[1] - box[0], 0, None))
    return inter / max(vol(a) + vol(b) - inter, 1e-12), inter / max(min(vol(a), vol(b)), 1e-12)


# ---------------------------------------------------------------- preview

def contact_sheet(path, objects, frames, label_maps, track_lists, cell=256, columns=5):
    from PIL import Image, ImageDraw
    if not objects:
        return
    rows = (len(objects) + columns - 1) // columns
    sheet = Image.new('RGB', (columns * cell, rows * (cell + 22)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for n, ob in enumerate(objects):
        f = ob['best_frame_index']
        mask = np.isin(label_maps[f], track_lists[ob['id']])
        img = frames[f].copy()
        img[~mask] = (img[~mask] * 0.35).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        if len(xs):
            pad = 20
            x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
            y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
            img = img[y0:y1, x0:x1]
        tile = Image.fromarray(img)
        tile.thumbnail((cell, cell))
        x, y = (n % columns) * cell, (n // columns) * (cell + 22)
        sheet.paste(tile, (x + (cell - tile.width) // 2, y + (cell - tile.height) // 2))
        draw.text((x + 6, y + cell + 4), f"{ob['id']:02d} {ob['label']} ({ob['score']:.2f})", fill=(230, 230, 230))
    sheet.save(path, quality=88)


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', default='/content/work')
    p.add_argument('--scene', default='/content/scene/scene.ply')
    p.add_argument('--out', default='/content/objects')
    p.add_argument('--prompts', default=DEFAULT_PROMPTS)
    p.add_argument('--min-score', type=float, default=0.5)
    p.add_argument('--min-frames', type=int, default=3)
    p.add_argument('--min-views', type=int, default=3)
    p.add_argument('--min-fraction', type=float, default=0.5)
    p.add_argument('--min-splats', type=int, default=100)
    p.add_argument('--depth-tol', type=float, default=0.05)
    p.add_argument('--max-area-frac', type=float, default=0.35, help='drop tracks whose mask ever covers more of the frame than this')
    p.add_argument('--contain', type=float, default=0.7, help='merge groups when box overlap / smaller box exceeds this')
    p.add_argument('--no-cache', action='store_true')
    p.add_argument('--window', type=int, default=40)
    p.add_argument('--overlap', type=int, default=8)
    p.add_argument('--precision', default='float16', choices=['float16', 'float32'])
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob('*'):
        old.unlink()
    training = Path(json.loads((Path(args.work) / 'camera.json').read_text())['training'])
    prompts = [t.strip() for t in args.prompts.split(',') if t.strip()]
    frames, viewmats, Ks, _, _ = load_dataset(training)
    log(f'Segmenting {len(frames)} frames with {len(prompts)} prompts')

    t0 = time.time()
    cache = Path(args.work) / 'sam3_cache.pkl'
    key = {'prompts': prompts, 'window': args.window, 'overlap': args.overlap, 'n': len(frames)}
    loaded = None
    if cache.exists() and not args.no_cache:
        saved = pickle.loads(cache.read_bytes())
        if saved['key'] == key:
            loaded = saved['data']
            log('Reusing cached SAM 3 results (use --no-cache to redo)')
    if loaded is None:
        try:
            loaded = run_sam3(frames, prompts, args.precision, args.window, args.overlap)
        except FloatingPointError:
            log('float16 gave invalid scores; rerunning SAM 3 in float32')
            loaded = run_sam3(frames, prompts, 'float32', args.window, args.overlap)
        cache.write_bytes(pickle.dumps({'key': key, 'data': loaded}))
    label_maps, tracks, pair_iou = loaded
    log(f'SAM 3 found {len(tracks)} tracks in {(time.time() - t0) / 60:.1f} min')

    # Keep confident, persistent tracks; merge tracks that are the same object in 2D.
    frame_px = label_maps[0].size
    too_big = {t for t, info in tracks.items() if info.get('max_area', info['best_area']) > args.max_area_frac * frame_px}
    keep = {t for t, info in tracks.items() if info['score'] >= args.min_score and info['frames'] >= args.min_frames
            and t not in too_big}
    log(f'Dropped {len(too_big)} tracks covering >{args.max_area_frac:.0%} of the frame')
    uf = UnionFind()
    for t in keep:
        uf.find(t)
    for (a, b), iou in pair_iou.items():
        if a in keep and b in keep:
            uf.union(a, b)
    groups: dict[int, list[int]] = {}
    for t in keep:
        groups.setdefault(uf.find(t), []).append(t)
    group_list = list(groups.values())
    track_index = {t: g for g, members in enumerate(group_list) for t in members}

    rows = read_ply(Path(args.scene))
    gs = splat_params(rows)
    t1 = time.time()
    k = max(len(group_list), 1)
    uniq, counts, seen = vote(gs[0], gs, viewmats, Ks, label_maps, track_index, k, args.depth_tol)
    labels = assign(uniq, counts, seen, len(rows), k, args.min_views, args.min_fraction)
    log(f'Voting done in {(time.time() - t1):.0f} s')

    # Merge same-prompt groups that occupy the same 3D box (track lost and re-found).
    means = gs[0]
    boxes = {g: robust_box(means[labels == g]) for g in range(len(group_list)) if (labels == g).sum() >= args.min_splats}
    uf3 = UnionFind()
    names = {g: {tracks[t]['prompt'] for t in group_list[g]} for g in boxes}
    gids = sorted(boxes)
    for i, a in enumerate(gids):
        uf3.find(a)
        for b in gids[i + 1:]:
            iou3, inside = box_overlap(boxes[a], boxes[b])
            if names[a] & names[b] and (iou3 > 0.5 or inside > args.contain):
                uf3.union(a, b)
    final = sorted({uf3.find(g) for g in gids})
    remap = np.full(len(group_list) + 1, -1, np.int32)
    for new, root in enumerate(final):
        for g in gids:
            if uf3.find(g) == root:
                remap[g] = new
    labels = np.where(labels >= 0, remap[labels], -1).astype(np.int32)

    objects, track_lists = [], {}
    for new, root in enumerate(final):
        members = [t for g in gids if uf3.find(g) == root for t in group_list[g]]
        pts = means[labels == new]
        lo, hi = robust_box(pts)
        best = max(members, key=lambda t: tracks[t]['best_area'])
        prompt_scores = {}
        for t in members:
            prompt_scores[tracks[t]['prompt']] = max(prompt_scores.get(tracks[t]['prompt'], 0), tracks[t]['score'])
        label = max(prompt_scores, key=prompt_scores.get)
        slug = re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')
        ply_name = f'object_{new:02d}_{slug}.ply'
        write_ply(out / ply_name, rows[labels == new])
        track_lists[new] = members
        objects.append({
            'id': new, 'label': label, 'prompts': sorted(prompt_scores), 'score': round(max(prompt_scores.values()), 3),
            'splats': int(len(pts)), 'center': [round(float(x), 4) for x in pts.mean(0)],
            'box_min': [round(float(x), 4) for x in lo], 'box_max': [round(float(x), 4) for x in hi],
            'frames_seen': int(sum(tracks[t]['frames'] for t in members)),
            'best_frame_index': int(tracks[best]['best_frame']), 'ply': ply_name, 'sam3_track_ids': members,
        })

    np.save(out / 'labels.npy', labels)
    write_ply(out / 'scene_without_objects.ply', rows[labels < 0])
    (out / 'objects.json').write_text(json.dumps({
        'scene': str(args.scene), 'units': 'scene units (camera path ~3 units; not meters)',
        'prompts': prompts, 'settings': {k: v for k, v in vars(args).items() if k not in ('prompts',)},
        'objects': objects}, indent=1))
    contact_sheet(out / 'preview.jpg', objects, frames, label_maps, track_lists)
    labeled = int((labels >= 0).sum())
    log(f'DONE {len(objects)} objects, {labeled:,} of {len(rows):,} splats labeled '
        f'({(time.time() - t0) / 60:.1f} min total)')


if __name__ == '__main__':
    main()
