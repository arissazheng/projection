// Drive the SuperSplat walkthrough viewer with hand gestures from `python -m gestures.server`.
//
// The viewer has no public camera API, so the bridge presses the same keys a person
// would, on the viewer's own canvas:
//
//   point  (index up)              hold ↑: move forward
//   peace  (index + middle up)     hold ↓: move back
//   palm   (four fingers up)       hand left/right of center: slide left/right (← →)
//                                  hand above/below center: rise/sink (E / Q)
//   three  (index + middle + ring) press R: reset to the starting view
//   fist / no hand                 release everything
//
// Why keys only: the arrow keys and Q/E move the camera like W/A/S/D, but W/A/S/D also
// switch the viewer to "gaming controls", which need pointer lock, and a script cannot
// take pointer lock. Simulated mouse drags for turning the camera were not reliable
// either, so the palm slides the camera instead. Sliding sideways in front of the room
// is also the parallax motion the projection is about.
//
// Movement stops once the camera is MAX_DISTANCE from where it started, because fly
// mode otherwise passes through walls and leaves the captured room. Add ?nogestures to
// the viewer URL to disable the bridge, or ?gestures=ws://host:port for another stream.

import { GestureClient } from '../js/gestures.js';

const params = new URL(location.href).searchParams;
const DEAD_ZONE = 0.25; // palm offsets smaller than this (in [-1, 1]) hold still
const MAX_DISTANCE = 4; // scene units (the importer scales the camera path to about 3)
const COAST_SECONDS = 0.3; // fly mode keeps gliding briefly after a key is released
const SETTLE_SECONDS = 0.4; // ignore direction while momentum from the last move fades
// Fly mode moves 4 units/s, fast for hand control. Holding Ctrl is the viewer's own
// "slow" modifier (a quarter of the speed).
const SLOW_KEY = 'ControlLeft';
const WALK_KEYS = { point: 'ArrowUp', peace: 'ArrowDown' };
const LABELS = {
  palm: '✋ Slide', point: '☝️ Forward', peace: '✌️ Back',
  three: '🤟 Reset', fist: '✊ Hold', none: 'No hand',
};
const KEY_NAMES = { ArrowUp: 'ArrowUp', ArrowDown: 'ArrowDown', ArrowLeft: 'ArrowLeft',
  ArrowRight: 'ArrowRight', ControlLeft: 'Control' };

const client = new GestureClient(params.get('gestures') ?? 'ws://localhost:8765');
let canvas = null;
let heldKey = null;
let blockedKey = null; // a key stopped at MAX_DISTANCE, until the gesture changes
let lastGesture = 'none';
let start = null; // camera position when the scene loaded or was reset
let lastTime = performance.now();
let lastDistance = 0;
let heldFor = 0; // seconds the current key has been held

function keyEvent(type, code) {
  const key = KEY_NAMES[code] ?? (code.startsWith('Key') ? code.slice(3).toLowerCase() : code.replace('Digit', ''));
  canvas.dispatchEvent(new KeyboardEvent(type, { code, key, bubbles: true, cancelable: true }));
}

function tapKey(code) {
  keyEvent('keydown', code);
  setTimeout(() => keyEvent('keyup', code), 50);
}

function holdKey(code) {
  if (heldKey === code) return;
  releaseKey();
  if (!code) return;
  keyEvent('keydown', SLOW_KEY);
  keyEvent('keydown', code);
  heldKey = code;
  heldFor = 0;
}

function releaseKey() {
  if (!heldKey) return;
  keyEvent('keyup', heldKey);
  keyEvent('keyup', SLOW_KEY);
  heldKey = null;
}

function cameraPosition() {
  return window.app?.root?.findComponents?.('camera')?.[0]?.entity?.getPosition().clone() ?? null;
}

function distanceFromStart() {
  const p = cameraPosition();
  if (!p) return 0;
  if (!start) start = p;
  return p.distance(start);
}

/** Palm: the hand's offset from the frame center picks one direction to slide. */
function slideKey(g) {
  const ax = Math.abs(g.x), ay = Math.abs(g.y);
  if (Math.max(ax, ay) < DEAD_ZONE) return null;
  if (ax >= ay) return g.x > 0 ? 'ArrowRight' : 'ArrowLeft';
  return g.y > 0 ? 'KeyE' : 'KeyQ';
}

/** Hold `code`, but stop before the camera glides past MAX_DISTANCE from the start. */
function move(code, dt) {
  if (!code) {
    releaseKey();
    lastDistance = distanceFromStart();
    return;
  }
  if (code === blockedKey) return releaseKey();
  const distance = distanceFromStart();
  const outward = (distance - lastDistance) / Math.max(dt, 1e-3); // + = moving away from start
  lastDistance = distance;
  if (heldKey === code) {
    heldFor += dt;
    // Only block motion away from the start, so moving back toward it still works.
    if (heldFor > SETTLE_SECONDS && outward > 0 && distance + outward * COAST_SECONDS > MAX_DISTANCE) {
      blockedKey = code;
      return releaseKey();
    }
  }
  holdKey(code);
}

function update(now) {
  requestAnimationFrame(update);
  const dt = Math.min((now - lastTime) / 1000, 0.1);
  lastTime = now;
  if (!canvas) return;

  const g = client.state;
  if (g.gesture !== lastGesture) {
    blockedKey = null;
    if (g.gesture === 'three') {
      releaseKey();
      tapKey('KeyR');
      setTimeout(() => { start = cameraPosition(); }, 300);
    }
    lastGesture = g.gesture;
  }

  move(g.gesture === 'palm' ? slideKey(g) : WALK_KEYS[g.gesture] ?? null, dt);

  badge.textContent = !client.connected
    ? 'Gestures: not connected. Run python -m gestures.server'
    : `Gestures: ${LABELS[g.gesture] ?? g.gesture}${blockedKey ? ' (edge of room)' : ''}`;
}

const badge = document.createElement('div');
badge.style.cssText = [
  'position:fixed', 'left:12px', 'bottom:12px', 'z-index:1000', 'padding:6px 10px',
  'border-radius:8px', 'background:rgba(0,0,0,.6)', 'color:#fff', 'pointer-events:none',
  'font:13px -apple-system,BlinkMacSystemFont,sans-serif',
].join(';');

function waitForViewer() {
  // Input only counts when it targets the engine's render canvas.
  canvas = window.app?.graphicsDevice?.canvas ?? null;
  if (!canvas) {
    setTimeout(waitForViewer, 250);
    return;
  }
  document.body.appendChild(badge);
  // Fly mode lets the camera move freely through the captured room.
  setTimeout(() => {
    tapKey('Digit2');
    setTimeout(() => { start = cameraPosition(); }, 300);
  }, 500);
}

if (!params.has('nogestures')) {
  client.start();
  waitForViewer();
  requestAnimationFrame(update);
  window.addEventListener('blur', releaseKey);
}
