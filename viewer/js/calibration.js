// Map the viewport rectangle to four measured corners with a homography.

const STORAGE_KEY = 'roomwindow.corners.v1';
const DEFAULT_CORNERS = [[0, 0], [1, 0], [1, 1], [0, 1]]; // Top left, clockwise, normalized.

/** Solve for H (3x3, h22 = 1) with H · src_i ∝ dst_i for 4 point pairs. */
export function solveHomography(src, dst) {
  const A = [], b = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i], [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
  }
  // Gaussian elimination with partial pivoting.
  const n = 8;
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
    [A[col], A[piv]] = [A[piv], A[col]];
    [b[col], b[piv]] = [b[piv], b[col]];
    for (let r = col + 1; r < n; r++) {
      const f = A[r][col] / A[col][col];
      for (let c = col; c < n; c++) A[r][c] -= f * A[col][c];
      b[r] -= f * b[col];
    }
  }
  const h = new Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let c = r + 1; c < n; c++) s -= A[r][c] * h[c];
    h[r] = s / A[r][r];
  }
  return [...h, 1];
}

/** 3x3 homography → CSS matrix3d (column-major 4x4, z passed through). */
export function homographyToCss(h) {
  const [a, b, c, d, e, f, g, i] = h;
  return `matrix3d(${[a, d, 0, g, b, e, 0, i, 0, 0, 1, 0, c, f, 0, 1].join(',')})`;
}

export class CornerPin {
  constructor(stage, gridCanvas) {
    this.stage = stage;
    this.grid = gridCanvas;
    this.active = false;
    this.selected = 0;
    this.corners = this._load();
    this.handles = this.corners.map((_, i) => {
      const el = document.createElement('div');
      el.className = 'handle';
      el.addEventListener('pointerdown', (ev) => this._drag(ev, i));
      document.body.appendChild(el);
      return el;
    });
    window.addEventListener('resize', () => this.apply());
    this.apply();
  }

  _load() {
    try {
      const c = JSON.parse(localStorage.getItem(STORAGE_KEY));
      if (Array.isArray(c) && c.length === 4) return c;
    } catch { /* storage unavailable */ }
    return DEFAULT_CORNERS.map((point) => [...point]);
  }

  _save() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(this.corners)); } catch { /* ignore */ }
  }

  setActive(on) {
    this.active = on;
    this.handles.forEach((h) => (h.style.display = on ? 'block' : 'none'));
    this.grid.style.display = on ? 'block' : 'none';
    this.apply();
  }

  reset() {
    this.corners = DEFAULT_CORNERS.map((point) => [...point]);
    this._save();
    this.apply();
  }

  nudge(dx, dy) {
    const c = this.corners[this.selected];
    c[0] += dx / window.innerWidth;
    c[1] += dy / window.innerHeight;
    this._save();
    this.apply();
  }

  _drag(ev, i) {
    ev.preventDefault();
    this.selected = i;
    const move = (e) => {
      this.corners[i] = [e.clientX / window.innerWidth, e.clientY / window.innerHeight];
      this.apply();
    };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      this._save();
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    this.apply();
  }

  apply() {
    const W = window.innerWidth, H = window.innerHeight;
    const src = [[0, 0], [W, 0], [W, H], [0, H]];
    const dst = this.corners.map(([x, y]) => [x * W, y * H]);
    this.stage.style.transform = homographyToCss(solveHomography(src, dst));
    this.handles.forEach((el, i) => {
      el.style.left = `${dst[i][0]}px`;
      el.style.top = `${dst[i][1]}px`;
      el.classList.toggle('selected', i === this.selected);
    });
    if (this.active) this._drawGrid(W, H);
  }

  _drawGrid(W, H) {
    const g = this.grid;
    g.width = W; g.height = H;
    const ctx = g.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = 'rgba(255, 204, 77, 0.9)';
    ctx.lineWidth = 2;
    ctx.strokeRect(1, 1, W - 2, H - 2);
    ctx.lineWidth = 1;
    for (let k = 1; k < 8; k++) {
      ctx.beginPath(); ctx.moveTo((W * k) / 8, 0); ctx.lineTo((W * k) / 8, H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, (H * k) / 8); ctx.lineTo(W, (H * k) / 8); ctx.stroke();
    }
  }
}
