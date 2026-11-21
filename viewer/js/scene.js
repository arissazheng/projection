// Turn (photo, disparity) into a 3D mesh behind the screen.
//
// Each grid vertex is a pixel of the photo. It sits on the ray from the *reference
// eye* (0, 0, E) through that pixel's spot on the screen plane, at distance D:
//
//   screen point S = (xs, ys, 0)       photo "cover"-fit to the screen rectangle
//   1/D = 1/far + d * (1/near - 1/far) d = normalized disparity (1 = near)
//   P   = (xs * D/E, ys * D/E, E - D)
//
// The reference eye sees the source photo crop. Moving the eye produces parallax.

import * as THREE from 'three';

const MAX_GRID_COLS = 420; // ~420 x 315 ≈ 130k vertices per layer

export async function loadSceneData(name) {
  const base = `scenes/${encodeURIComponent(name)}/`;
  const get = async (f) => {
    const r = await fetch(base + f);
    if (!r.ok) throw new Error(`${base + f}: ${r.status}`);
    return r;
  };
  const meta = await (await get('meta.json')).json();
  const [fgDisp, bgDisp] = await Promise.all(
    ['disparity.f32', 'bg_disparity.f32'].map(async (f) => new Float32Array(await (await get(f)).arrayBuffer())),
  );
  const loader = new THREE.TextureLoader();
  const [fgTex, bgTex] = await Promise.all(['image.jpg', 'bg.jpg'].map((f) => loader.loadAsync(base + f)));
  for (const t of [fgTex, bgTex]) {
    t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = 4;
  }
  return { meta, fgDisp, bgDisp, fgTex, bgTex };
}

export class LayerMesh {
  /**
   * @param {Float32Array} disp full-resolution disparity (meta.gridWidth x meta.gridHeight)
   * @param {object} meta scene meta.json
   * @param {THREE.Texture} texture
   * @param {boolean} dropEdges remove triangles that straddle depth discontinuities
   */
  constructor(disp, meta, texture, { dropEdges, isBackground = false }) {
    const step = Math.max(1, Math.ceil(meta.gridWidth / MAX_GRID_COLS));
    const cols = Math.floor((meta.gridWidth - 1) / step) + 1;
    const rows = Math.floor((meta.gridHeight - 1) / step) + 1;
    this.cols = cols;
    this.rows = rows;
    this.dropEdges = dropEdges;

    this.d = new Float32Array(cols * rows);
    const uv = new Float32Array(cols * rows * 2);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const i = r * cols + c;
        const px = Math.min(c * step, meta.gridWidth - 1);
        const py = Math.min(r * step, meta.gridHeight - 1);
        this.d[i] = disp[py * meta.gridWidth + px];
        uv[2 * i] = px / (meta.gridWidth - 1);
        uv[2 * i + 1] = 1 - py / (meta.gridHeight - 1);
      }
    }

    this.geometry = new THREE.BufferGeometry();
    this.positions = new Float32Array(cols * rows * 3);
    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    this.geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2));

    const material = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
    if (isBackground) {
      // Coincides with the foreground wherever there is no hole; push it back a hair.
      material.polygonOffset = true;
      material.polygonOffsetFactor = 2;
      material.polygonOffsetUnits = 2;
    } else {
      material.depthFunc = THREE.LessEqualDepth;
    }
    this.mesh = new THREE.Mesh(this.geometry, material);
    this.mesh.renderOrder = isBackground ? 0 : 1;
    this.mesh.frustumCulled = false;
    this._lastEdgeThreshold = null;
  }

  /** Recompute vertex positions (and edge-dropped triangles) for new parameters. */
  update({ planeW, planeH, refEyeZ, nearDist, farDist, edgeThreshold }) {
    const { cols, rows, d, positions } = this;
    const invFar = 1 / farDist;
    const invRange = 1 / nearDist - invFar;
    for (let r = 0; r < rows; r++) {
      const ys = (0.5 - r / (rows - 1)) * planeH;
      for (let c = 0; c < cols; c++) {
        const i = r * cols + c;
        const xs = (c / (cols - 1) - 0.5) * planeW;
        const D = 1 / (invFar + d[i] * invRange);
        const s = D / refEyeZ;
        positions[3 * i] = xs * s;
        positions[3 * i + 1] = ys * s;
        positions[3 * i + 2] = refEyeZ - D;
      }
    }
    this.geometry.attributes.position.needsUpdate = true;
    if (this._lastEdgeThreshold !== edgeThreshold) {
      this._buildIndex(edgeThreshold);
      this._lastEdgeThreshold = edgeThreshold;
    }
  }

  _buildIndex(edgeThreshold) {
    const { cols, rows, d } = this;
    const idx = new Uint32Array((cols - 1) * (rows - 1) * 6);
    let n = 0;
    const ok = (a, b, c) => {
      if (!this.dropEdges) return true;
      const hi = Math.max(d[a], d[b], d[c]);
      const lo = Math.min(d[a], d[b], d[c]);
      return hi - lo <= edgeThreshold;
    };
    for (let r = 0; r < rows - 1; r++) {
      for (let c = 0; c < cols - 1; c++) {
        const a = r * cols + c, b = a + 1, e = a + cols, f = e + 1;
        if (ok(a, e, b)) { idx[n++] = a; idx[n++] = e; idx[n++] = b; }
        if (ok(b, e, f)) { idx[n++] = b; idx[n++] = e; idx[n++] = f; }
      }
    }
    this.geometry.setIndex(new THREE.BufferAttribute(idx.slice(0, n), 1));
  }

  dispose() {
    this.geometry.dispose();
    this.mesh.material.map.dispose();
    this.mesh.material.dispose();
  }
}
