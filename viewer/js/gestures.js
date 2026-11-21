// Apply hand gestures received from the local WebSocket server.

export const GESTURE_HELP = [
  ['palm', '✋', 'Open palm', 'Look around: move your hand to steer'],
  ['point', '☝️', 'Point', 'Walk forward into the room'],
  ['peace', '✌️', 'Peace', 'Walk back'],
  ['three', '🤟', 'Three fingers', 'Reset to the photo view'],
  ['fist', '✊', 'Fist', 'Hold: freeze the view'],
];

const NO_GESTURE = { hand: false, gesture: 'none', x: 0, y: 0 };

export class GestureClient {
  constructor(url = 'ws://localhost:8765') {
    this.url = url;
    this.ws = null;
    this.last = NO_GESTURE;
    this.lastAt = -Infinity;
    this._wanted = false;
    this._retry = null;
  }

  start() {
    if (this._wanted) return;
    this._wanted = true;
    this._connect();
  }

  stop() {
    this._wanted = false;
    clearTimeout(this._retry);
    const socket = this.ws;
    this.ws = null;
    this.lastAt = -Infinity;
    socket?.close();
  }

  _connect() {
    if (!this._wanted) return;
    let socket;
    try {
      socket = new WebSocket(this.url);
    } catch {
      this._retry = setTimeout(() => this._connect(), 1000);
      return;
    }
    this.ws = socket;
    socket.onmessage = (event) => {
      if (socket !== this.ws) return;
      try {
        this.last = JSON.parse(event.data);
        this.lastAt = performance.now();
      } catch { /* ignore malformed */ }
    };
    socket.onclose = () => {
      if (socket !== this.ws) return;
      this.ws = null;
      this.lastAt = -Infinity;
      this._retry = setTimeout(() => this._connect(), 1000);
    };
    socket.onerror = () => socket.close();
  }

  get connected() {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /** Latest gesture; stale (>300 ms) data counts as no hand, so the view holds. */
  get state() {
    if (!this.connected || performance.now() - this.lastAt > 300) return NO_GESTURE;
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
