// Drive the SuperSplat walkthrough viewer with hand gestures from `python -m gestures.server`.
//
// The viewer has no public camera API, so the bridge presses the same keys a person
// would, on the viewer's own canvas:
//
//   point  (index up)              hold ↑: move forward
//   peace  (index + middle up)     hold ↓: move back
//   palm   (four fingers up)       joystick turn: hand left/right of center turns,
//                                  above/below looks up/down
//   three  (index + middle + ring) reset to the starting position and direction
//   fist / no hand                 release everything
//
// Moving uses the arrow keys, not W/A/S/D: W/A/S/D also switch the viewer to "gaming
// controls", which need pointer lock, and a script cannot take pointer lock.
//
// Turning: simulated mouse drags were not reliable, so the bridge turns the camera
// through the viewer's own camera-state hook. The viewer only puts getCameraState /
// setCameraState on window while its debug panel (Ctrl+Shift+D) is open, so the bridge
// opens the panel once, keeps the two functions, and closes it again.
//
// Movement stops once the camera is MAX_DISTANCE from where it started, because fly
// mode otherwise passes through walls and leaves the captured room. Add ?nogestures to
// the viewer URL to disable the bridge, or ?gestures=ws://host:port for another stream.

import { GestureClient } from '../js/gestures.js';

const params = new URL(location.href).searchParams;
const DEAD_ZONE = 0.25; // palm offsets smaller than this (in [-1, 1]) hold still
const TURN_SPEED = 70; // degrees per second at full hand offset
const MAX_PITCH = 75; // degrees above or below level
const MAX_DISTANCE = 4; // scene units (the importer scales the camera path to about 3)
const COAST_SECONDS = 0.3; // fly mode keeps gliding briefly after a key is released
const SETTLE_SECONDS = 0.4; // ignore direction while momentum from the last move fades
// Fly mode moves 4 units/s, fast for hand control. Holding Ctrl is the viewer's own
// "slow" modifier (a quarter of the speed).
const SLOW_KEY = 'ControlLeft';
const WALK_KEYS = { point: 'ArrowUp', peace: 'ArrowDown' };
const LABELS = {
  palm: '✋ Turn', point: '☝️ Forward', peace: '✌️ Back',
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
let cameraState = null; // { get, set } from the viewer's debug panel hook
let startView = null; // camera state to restore on reset

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

/** Palm joystick: turn speed grows with the hand's distance from the frame center. */
function turn(g, dt) {
  if (!cameraState) return;
  const axis = (v) => {
    const m = Math.abs(v);
    return m < DEAD_ZONE ? 0 : Math.sign(v) * (m - DEAD_ZONE) / (1 - DEAD_ZONE);
  };
  const yaw = axis(g.x), pitch = axis(g.y);
  if (!yaw && !pitch) return;
  const s = cameraState.get();
  if (s.mode !== 'fly') return;
  s.angles[1] -= yaw * TURN_SPEED * dt; // hand right = turn right
  s.angles[0] = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, s.angles[0] + pitch * TURN_SPEED * dt));
  cameraState.set(s);
}

/** Open the debug panel just long enough to keep its camera get/set functions. */
function captureCameraHook() {
  const toggle = () => {
    const mods = { ctrlKey: true, shiftKey: true };
    canvas.dispatchEvent(new KeyboardEvent('keydown', { code: 'KeyD', key: 'D', bubbles: true, ...mods }));
    canvas.dispatchEvent(new KeyboardEvent('keyup', { code: 'KeyD', key: 'D', bubbles: true, ...mods }));
  };
  if (!window.getCameraState) toggle();
  const get = window.getCameraState, set = window.setCameraState;
  toggle(); // close the panel; the kept functions stay valid
  if (typeof get === 'function' && typeof set === 'function') {
    cameraState = { get, set };
    startView = get();
  }
  else console.warn('gesture bridge: camera hook unavailable, palm turning disabled');
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
      if (startView) {
        // Position and direction. Apply again once any glide from the last move has stopped.
        cameraState.set(structuredClone(startView));
        setTimeout(() => cameraState.set(structuredClone(startView)), 400);
      } else {
        tapKey('KeyR');
      }
      setTimeout(() => { start = cameraPosition(); }, 500);
    }
    lastGesture = g.gesture;
  }

  move(WALK_KEYS[g.gesture] ?? null, dt);
  if (g.gesture === 'palm') turn(g, dt);

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
    setTimeout(() => {
      start = cameraPosition();
      captureCameraHook();
    }, 300);
  }, 500);
}

if (!params.has('nogestures')) {
  client.start();
  waitForViewer();
  requestAnimationFrame(update);
  window.addEventListener('blur', releaseKey);
}
