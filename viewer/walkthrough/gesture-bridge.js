// Drive the SuperSplat walkthrough viewer with hand gestures from `python -m gestures.server`.
//
// Hand movement maps directly to camera movement: how far you move is how far the
// camera goes, and how fast you move is how fast it goes.
//
//   palm   (four fingers up)       turn: moving your hand left/right/up/down from where
//                                  you opened it turns the view by a matching angle.
//                                  Turning stops at the edge of the captured data.
//   point  (index up)              push: moving your hand toward the camera moves the
//                                  camera forward, pulling it back moves backward.
//   peace  (index + middle up)     move back steadily
//   three  (index + middle + ring) return to the starting position and direction
//   fist / no hand                 hold, so you can reposition your hand
//
// The viewer has no public camera API. Its debug panel (Ctrl+Shift+D) puts
// getCameraState / setCameraState on window while it is open, so the bridge opens it
// once, keeps the two functions, and closes it again.
//
// Movement stops MAX_DISTANCE from the start so the camera stays inside the captured
// room. Add ?nogestures to the viewer URL to disable the bridge, or
// ?gestures=ws://host:port to use another gesture stream.

import { GestureClient } from '../js/gestures.js';

const params = new URL(location.href).searchParams;
const TURN_PER_HAND = 60; // degrees of turn per half-frame of hand movement
const PUSH_PER_DOUBLING = 1.5; // scene units moved when the hand's apparent size doubles
const SIZE_SMOOTHING = 12; // 1/s; filters frame-to-frame jitter in the hand size
const FOLLOW = 14; // 1/s; how quickly the camera catches up with the hand
const MAX_DISTANCE = 4; // scene units (the importer scales the camera path to about 3)
const DEFAULT_COVERAGE = { yaw: [-30, 30], pitch: [-20, 20] }; // if the scene has none
const BACK_SPEED = 0.8; // scene units per second for the "peace" gesture
const LABELS = {
  palm: '✋ Turn', point: '☝️ Push to move', peace: '✌️ Back',
  three: '🤟 Reset', fist: '✊ Hold', none: 'No hand',
};

const client = new GestureClient(params.get('gestures') ?? 'ws://localhost:8765');
let canvas = null;
let camera = null; // { get, set } from the viewer's debug panel hook
let startView = null; // camera state at load, restored by the reset gesture
let coverage = DEFAULT_COVERAGE;
let lastGesture = 'none';
let lastTime = performance.now();
let edge = ''; // why movement stopped, for the badge

// Palm turning: angles relative to the start view (yaw + = right, pitch + = up).
let turn = { yaw: 0, pitch: 0 };
let turnTarget = { yaw: 0, pitch: 0 };
let turnAnchor = null; // { x, y, yaw, pitch } when the palm opened

// Pointing: smoothed log of the hand's apparent size.
let logSize = null;
let pushPending = 0; // scene units still to move toward the hand's target

// ---------------------------------------------------------------- viewer input

function tapKey(code, extra = {}) {
  const key = code.startsWith('Key') ? code.slice(3) : code.replace('Digit', '');
  for (const type of ['keydown', 'keyup']) {
    canvas.dispatchEvent(new KeyboardEvent(type, { code, key, bubbles: true, cancelable: true, ...extra }));
  }
}

/** Open the debug panel just long enough to keep its camera get/set functions. */
function captureCameraHook() {
  const toggle = () => tapKey('KeyD', { ctrlKey: true, shiftKey: true });
  if (!window.getCameraState) toggle();
  const get = window.getCameraState, set = window.setCameraState;
  toggle(); // close the panel; the kept functions stay valid
  if (typeof get !== 'function' || typeof set !== 'function') {
    console.warn('gesture bridge: camera hook unavailable; gestures disabled');
    return;
  }
  camera = { get, set };
  startView = get();
}

// ---------------------------------------------------------------- camera math

const RAD = Math.PI / 180;
const clamp = (v, [lo, hi]) => Math.max(lo, Math.min(hi, v));
const ease = (rate, dt) => 1 - Math.exp(-rate * dt);

/** Unit view direction for viewer angles (degrees): x = pitch, y = yaw. */
function direction(angles) {
  const pitch = angles[0] * RAD, yaw = angles[1] * RAD;
  return [-Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), -Math.cos(yaw) * Math.cos(pitch)];
}

function distanceFromStart(position) {
  const s = startView.position;
  return Math.hypot(position[0] - s[0], position[1] - s[1], position[2] - s[2]);
}

/** Turn: ease toward the target angles, which are clamped to the covered range. */
function applyTurn(state, dt) {
  const k = ease(FOLLOW, dt);
  turn.yaw += (turnTarget.yaw - turn.yaw) * k;
  turn.pitch += (turnTarget.pitch - turn.pitch) * k;
  state.angles[1] = startView.angles[1] - turn.yaw; // viewer yaw grows to the left
  state.angles[0] = startView.angles[0] + turn.pitch;
}

/** Push: move along the view direction, but not past MAX_DISTANCE from the start. */
function applyPush(state, dt) {
  if (Math.abs(pushPending) < 1e-4) return;
  const step = pushPending * ease(FOLLOW, dt);
  pushPending -= step;
  const d = direction(state.angles);
  const next = state.position.map((p, i) => p + d[i] * step);
  const outward = distanceFromStart(next) > distanceFromStart(state.position);
  if (outward && distanceFromStart(next) > MAX_DISTANCE) {
    pushPending = 0;
    edge = 'edge of room';
    return;
  }
  state.position = next;
}

// ---------------------------------------------------------------- gestures

function onGestureStart(g) {
  edge = '';
  turnAnchor = null;
  logSize = null;
  pushPending = 0;
  if (g.gesture === 'three' && startView) {
    camera.set(structuredClone(startView));
    turn = { yaw: 0, pitch: 0 };
    turnTarget = { yaw: 0, pitch: 0 };
  }
}

function palm(g) {
  if (!turnAnchor) turnAnchor = { x: g.x, y: g.y, yaw: turn.yaw, pitch: turn.pitch };
  const yaw = turnAnchor.yaw + (g.x - turnAnchor.x) * TURN_PER_HAND; // hand right = turn right
  const pitch = turnAnchor.pitch + (g.y - turnAnchor.y) * TURN_PER_HAND; // hand up = look up
  turnTarget = { yaw: clamp(yaw, coverage.yaw), pitch: clamp(pitch, coverage.pitch) };
  edge = turnTarget.yaw !== yaw || turnTarget.pitch !== pitch ? 'edge of captured view' : '';
}

function point(g, dt) {
  if (!(g.size > 0)) return;
  const raw = Math.log(g.size);
  if (logSize === null) {
    logSize = raw;
    return;
  }
  const smoothed = logSize + (raw - logSize) * ease(SIZE_SMOOTHING, dt);
  // Doubling the hand's apparent size (about half the distance to the camera) moves
  // PUSH_PER_DOUBLING units, so a faster push moves the camera faster.
  pushPending += ((smoothed - logSize) / Math.LN2) * PUSH_PER_DOUBLING;
  logSize = smoothed;
}

// ---------------------------------------------------------------- main loop

function update(now) {
  requestAnimationFrame(update);
  const dt = Math.min((now - lastTime) / 1000, 0.1);
  lastTime = now;
  if (!canvas) return;

  const g = client.state;
  if (g.gesture !== lastGesture) {
    lastGesture = g.gesture;
    onGestureStart(g);
  }

  if (camera) {
    if (g.gesture === 'palm') palm(g);
    if (g.gesture === 'point') point(g, dt);
    if (g.gesture === 'peace') pushPending = -BACK_SPEED / FOLLOW; // steady glide backward
    const state = camera.get();
    if (state.mode === 'fly') {
      const before = JSON.stringify([state.position, state.angles]);
      applyTurn(state, dt);
      applyPush(state, dt);
      if (JSON.stringify([state.position, state.angles]) !== before) camera.set(state);
    }
  }

  badge.textContent = !client.connected
    ? 'Gestures: not connected. Run python -m gestures.server'
    : `Gestures: ${LABELS[g.gesture] ?? g.gesture}${edge ? ` (${edge})` : ''}`;
}

const badge = document.createElement('div');
badge.style.cssText = [
  'position:fixed', 'left:12px', 'bottom:12px', 'z-index:1000', 'padding:6px 10px',
  'border-radius:8px', 'background:rgba(0,0,0,.6)', 'color:#fff', 'pointer-events:none',
  'font:13px -apple-system,BlinkMacSystemFont,sans-serif',
].join(';');

async function loadCoverage() {
  // The viewer URL names the scene file (/room/<id>/scene.ply); coverage sits beside it.
  const content = params.get('content');
  if (!content) return;
  try {
    const response = await fetch(content.replace(/scene\.ply$/, 'coverage.json'));
    if (response.ok) coverage = await response.json();
  } catch { /* keep the default range */ }
}

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
    setTimeout(captureCameraHook, 300);
  }, 500);
}

if (!params.has('nogestures')) {
  client.start();
  loadCoverage();
  waitForViewer();
  requestAnimationFrame(update);
}
