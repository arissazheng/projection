// Gesture navigation: receives hand gestures from `python -m gestures.server` over a
// WebSocket and turns them into movement through the scene.
//
//   palm  (4 fingers up)       LOOK     hand position steers the viewpoint
//   point (index up)           FORWARD  walk into the room
//   peace (index + middle up)  BACK     walk back out
//   three (index+middle+ring)  RESET    glide back to the original photo view
//   fist / no hand             HOLD     freeze (reposition your hand freely)

export const GESTURE_HELP = [
  ['palm', '✋', 'Open palm', 'Look around: move your hand to steer'],
  ['point', '☝️', 'Point', 'Walk forward into the room'],
  ['peace', '✌️', 'Peace', 'Walk back'],
  ['three', '🤟', 'Three fingers', 'Reset to the photo view'],
  ['fist', '✊', 'Fist', 'Hold: freeze the view'],
];

export class GestureClient {
  constructor(url = 'ws://localhost:8765') {
    this.url = url;
    this.ws = null;
    this.connected = false;
    this.last = { hand: false, gesture: 'none', x: 0, y: 0 };
    this.lastAt = 0;
    this._wanted = false;
  }

  start() {
    this._wanted = true;
    this._connect();
  }

  stop() {
    this._wanted = false;
    this.ws?.close();
  }

  _connect() {
    if (!this._wanted) return;
    try {
      this.ws = new WebSocket(this.url);
    } catch {
      setTimeout(() => this._connect(), 1000);
      return;
    }
    this.ws.onopen = () => (this.connected = true);
    this.ws.onmessage = (e) => {
      try {
        this.last = JSON.parse(e.data);
        this.lastAt = performance.now();
      } catch { /* ignore malformed */ }
    };
    this.ws.onclose = () => {
      this.connected = false;
      if (this._wanted) setTimeout(() => this._connect(), 1000);
    };
    this.ws.onerror = () => this.ws.close();
  }

  /** Latest gesture; stale (>300 ms) data counts as no hand, so the view holds. */
  get state() {
    if (!this.connected || performance.now() - this.lastAt > 300) return { hand: false, gesture: 'none', x: 0, y: 0 };
    return this.last;
  }
}

/** Integrates gestures into a navigation state { lookX, lookY, dolly }. */
export function stepNavigation(nav, g, dt, { walkSpeed, dollyMin, dollyMax }) {
  const ease = (cur, target, rate) => cur + (target - cur) * (1 - Math.exp(-rate * dt));
  switch (g.gesture) {
    case 'palm':
      nav.lookX = ease(nav.lookX, g.x, 10);
      nav.lookY = ease(nav.lookY, g.y, 10);
      break;
    case 'point':
      nav.dolly = Math.min(dollyMax, nav.dolly + walkSpeed * dt);
      break;
    case 'peace':
      nav.dolly = Math.max(dollyMin, nav.dolly - walkSpeed * dt);
      break;
    case 'three':
      nav.lookX = ease(nav.lookX, 0, 4);
      nav.lookY = ease(nav.lookY, 0, 4);
      nav.dolly = ease(nav.dolly, 0, 4);
      break;
    default: // fist, none: hold
      break;
  }
  return nav;
}
