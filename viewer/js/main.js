import * as THREE from 'three';
import { setOffAxisCamera } from './projection.js';
import { loadSceneData, LayerMesh } from './scene.js';
import { FaceTracker } from './tracking.js';
import { CornerPin } from './calibration.js';
import { GestureClient, GESTURE_HELP, stepNavigation } from './gestures.js';

// ---------- settings (persisted per browser; wrapped because storage can throw) ----------
const RIG_DEFAULTS = {
  screenWidth: 0.30,  // m, physical width of the display / projected image
  refEyeZ: 0.6,       // m, eye distance the photo is "correct" from
  mouseRange: 0.35,   // mouse/gesture look: eye travel as a fraction of the screen half-size
  walkSpeed: 0.8,     // gesture mode: m/s when walking forward/back
  gain: 1.0,          // webcam: exaggerate head motion (1 = physically correct)
  hfovDeg: 60,        // webcam horizontal field of view
  camAboveTop: 0.01,  // m, webcam lens above the top edge of the screen
  camZ: 0.0,          // m, webcam in front of (+) / behind (-) the screen plane
  minCutoff: 1.0,     // One-Euro smoothing: lower = smoother when still
  beta: 0.05,         // One-Euro: higher = less lag when moving fast
};
const SLIDERS = {
  scene: [
    ['nearDist', 'Near distance (m)', 0.2, 5, 0.01],
    ['farDist', 'Far distance (m)', 1, 30, 0.1],
    ['edgeThreshold', 'Edge threshold', 0.005, 0.3, 0.005],
  ],
  rig: [
    ['screenWidth', 'Screen width (m)', 0.1, 5, 0.01],
    ['refEyeZ', 'Reference eye distance (m)', 0.2, 5, 0.01],
    ['mouseRange', 'Look range (mouse/gesture)', 0.05, 2, 0.05],
    ['walkSpeed', 'Walk speed (m/s, gestures)', 0.1, 3, 0.05],
  ],
  webcam: [
    ['gain', 'Motion gain', 0.5, 4, 0.05],
    ['hfovDeg', 'Webcam FOV (°)', 40, 100, 1],
    ['camAboveTop', 'Webcam above screen top (m)', -0.5, 0.5, 0.005],
    ['camZ', 'Webcam z offset (m)', -0.5, 0.5, 0.005],
    ['minCutoff', 'Smoothing min cutoff', 0.05, 5, 0.05],
    ['beta', 'Smoothing beta', 0, 1, 0.01],
  ],
};

const store = {
  get(key, fallback) {
    try { const v = JSON.parse(localStorage.getItem(key)); return v ?? fallback; } catch { return fallback; }
  },
  set(key, v) { try { localStorage.setItem(key, JSON.stringify(v)); } catch { /* ignore */ } },
};

const rig = { ...RIG_DEFAULTS, ...store.get('roomwindow.rig.v1', {}) };
let sceneParams = {};
let sceneName = null;

// ---------- three.js setup ----------
const stage = document.getElementById('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x000000);
stage.prepend(renderer.domElement);
const camera = new THREE.Camera();
const world = new THREE.Scene();
let layers = null; // { fg: LayerMesh, bg: LayerMesh, meta }

const msg = document.getElementById('msg');
const hud = document.getElementById('hud');
const panel = document.getElementById('panel');
const corner = new CornerPin(stage, document.getElementById('calib-grid'));
const tracker = new FaceTracker(document.getElementById('webcam-preview'));
const gestures = new GestureClient(new URLSearchParams(location.search).get('ws') ?? 'ws://localhost:8765');

let mode = 'mouse';
let layerView = 'both';
const mouse = { x: 0, y: 0, zScale: 1 };
const nav = { lookX: 0, lookY: 0, dolly: 0 }; // gesture navigation; dolly moves the room toward you (m)

function screenSize() {
  const W = rig.screenWidth;
  return { W, H: (W * window.innerHeight) / window.innerWidth };
}

function resize() {
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  updateGeometry();
}
window.addEventListener('resize', resize);

// ---------- scene loading ----------
async function listScenes() {
  try {
    const r = await fetch('scenes/index.json', { cache: 'no-store' });
    return (await r.json()).scenes ?? [];
  } catch {
    return [];
  }
}

async function loadScene(name) {
  msg.textContent = `Loading ${name}…`;
  const data = await loadSceneData(name);
  if (layers) {
    world.remove(layers.fg.mesh, layers.bg.mesh);
    layers.fg.dispose(); layers.bg.dispose();
  }
  const { meta } = data;
  layers = {
    meta,
    fg: new LayerMesh(data.fgDisp, meta, data.fgTex, { dropEdges: true }),
    bg: new LayerMesh(data.bgDisp, meta, data.bgTex, { dropEdges: false, isBackground: true }),
  };
  world.add(layers.bg.mesh, layers.fg.mesh);
  sceneName = name;
  sceneParams = {
    nearDist: meta.nearDist, farDist: meta.farDist, edgeThreshold: meta.edgeThreshold,
    ...store.get(`roomwindow.scene.${name}`, {}),
  };
  setLayerView(layerView);
  buildSliders();
  updateGeometry();
  msg.textContent = '';
  const url = new URL(location.href);
  url.searchParams.set('scene', name);
  history.replaceState(null, '', url);
}

function updateGeometry() {
  if (!layers) return;
  const { W, H } = screenSize();
  // "Cover" fit the photo to the screen rectangle (crop instead of letterbox).
  const photoAspect = layers.meta.imageWidth / layers.meta.imageHeight;
  const planeW = Math.max(W, H * photoAspect);
  const planeH = planeW / photoAspect;
  const p = { planeW, planeH, refEyeZ: rig.refEyeZ, ...sceneParams };
  p.farDist = Math.max(p.farDist, p.nearDist + 0.01);
  layers.fg.update(p);
  layers.bg.update(p);
}

function setLayerView(v) {
  layerView = v;
  if (layers) {
    layers.fg.mesh.visible = v !== 'bg';
    layers.bg.mesh.visible = v !== 'fg';
  }
  document.querySelectorAll('[data-layers]').forEach((b) => b.classList.toggle('on', b.dataset.layers === v));
}

// ---------- UI ----------
function buildSliders() {
  const root = document.getElementById('sliders');
  root.innerHTML = '';
  const groups = [['Scene depth', 'scene', sceneParams], ['Display', 'rig', rig], ['Webcam tracking', 'webcam', rig]];
  for (const [title, key, target] of groups) {
    const h = document.createElement('h2');
    h.textContent = title;
    root.appendChild(h);
    for (const [prop, label, min, max, step] of SLIDERS[key]) {
      const row = document.createElement('div');
      row.className = 'row';
      const id = `s-${prop}`;
      row.innerHTML = `<label for="${id}">${label}</label><span class="val"></span>
        <input id="${id}" type="range" min="${min}" max="${max}" step="${step}">`;
      const input = row.querySelector('input');
      const val = row.querySelector('.val');
      input.value = target[prop];
      val.textContent = (+target[prop]).toFixed(step < 0.01 ? 3 : 2);
      input.addEventListener('input', () => {
        target[prop] = +input.value;
        val.textContent = (+input.value).toFixed(step < 0.01 ? 3 : 2);
        if (key === 'scene') store.set(`roomwindow.scene.${sceneName}`, sceneParams);
        else store.set('roomwindow.rig.v1', rig);
        updateGeometry();
      });
      root.appendChild(row);
    }
  }
}

async function setMode(m) {
  if (m === 'webcam' && !tracker.landmarker) {
    msg.textContent = 'Starting webcam + face model…';
    try {
      await tracker.start();
      msg.textContent = '';
    } catch (err) {
      console.error(err);
      msg.textContent = `Webcam failed: ${err.message}. Staying in mouse mode.`;
      setTimeout(() => (msg.textContent = ''), 4000);
      m = 'mouse';
    }
  }
  if (m === 'gesture') gestures.start(); else gestures.stop();
  mode = m;
  document.getElementById('mode-mouse').classList.toggle('on', m === 'mouse');
  document.getElementById('mode-webcam').classList.toggle('on', m === 'webcam');
  document.getElementById('mode-gesture').classList.toggle('on', m === 'gesture');
  document.getElementById('webcam-preview').style.display = m === 'webcam' ? 'block' : 'none';
  document.getElementById('gesture-help').style.display = m === 'gesture' ? 'block' : 'none';
}

const gestureHelp = document.getElementById('gesture-help');
gestureHelp.innerHTML = '<div class="status"></div>' + GESTURE_HELP.map(([id, icon, name, what]) =>
  `<div class="g" data-g="${id}"><span>${icon}</span><b>${name}</b><small>${what}</small></div>`).join('');
let shownGesture = null;
function updateGestureHelp(g) {
  gestureHelp.querySelector('.status').textContent = gestures.connected
    ? (g.hand ? `Seeing: ${g.gesture}` : 'Connected · no hand in view')
    : 'Not connected. Run: python -m gestures.server';
  if (g.gesture !== shownGesture) {
    shownGesture = g.gesture;
    gestureHelp.querySelectorAll('.g').forEach((el) => el.classList.toggle('on', el.dataset.g === g.gesture));
  }
}

function toggleCalibration() {
  corner.setActive(!corner.active);
  document.getElementById('calib').classList.toggle('on', corner.active);
}

document.getElementById('mode-mouse').onclick = () => setMode('mouse');
document.getElementById('mode-webcam').onclick = () => setMode('webcam');
document.getElementById('mode-gesture').onclick = () => setMode('gesture');
document.getElementById('calib').onclick = toggleCalibration;
document.querySelectorAll('[data-layers]').forEach((b) => (b.onclick = () => setLayerView(b.dataset.layers)));
document.getElementById('scene').onchange = (e) => loadScene(e.target.value);

window.addEventListener('pointermove', (e) => {
  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -((e.clientY / window.innerHeight) * 2 - 1);
});
window.addEventListener('wheel', (e) => {
  if (e.target.closest('#panel')) return;
  mouse.zScale = Math.min(4, Math.max(0.25, mouse.zScale * Math.exp(e.deltaY * 0.001)));
}, { passive: true });

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  const k = e.key.toLowerCase();
  if (k === 'h') { panel.classList.toggle('hidden'); hud.classList.toggle('hidden'); }
  else if (k === 'm') setMode({ mouse: 'webcam', webcam: 'gesture', gesture: 'mouse' }[mode]);
  else if (k === 'c') toggleCalibration();
  else if (k === 'f') document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
  else if (k === '0') { mouse.x = 0; mouse.y = 0; mouse.zScale = 1; Object.assign(nav, { lookX: 0, lookY: 0, dolly: 0 }); }
  else if (k === '1') setLayerView('both');
  else if (k === '2') setLayerView('fg');
  else if (k === '3') setLayerView('bg');
  else if (corner.active && k === 'r') corner.reset();
  else if (corner.active && e.key.startsWith('Arrow')) {
    const s = e.shiftKey ? 10 : 1;
    corner.nudge({ ArrowLeft: -s, ArrowRight: s }[e.key] ?? 0, { ArrowUp: -s, ArrowDown: s }[e.key] ?? 0);
    e.preventDefault();
  } else if (corner.active && ['tab'].includes(k)) {
    corner.selected = (corner.selected + 1) % 4; corner.apply(); e.preventDefault();
  }
});

// ---------- render loop ----------
let frames = 0, fps = 0, fpsT = performance.now(), lastFrame = performance.now();

function currentEye(dt) {
  const { W, H } = screenSize();
  if (mode === 'gesture') {
    const g = gestures.state;
    // Walk at most halfway to the far wall. Never back up past where the photo was taken:
    // behind that point you only see the photo's black borders.
    const far = sceneParams.farDist ?? 6;
    stepNavigation(nav, g, dt, { walkSpeed: rig.walkSpeed, dollyMin: 0, dollyMax: 0.5 * far });
    updateGestureHelp(g);
    return { x: nav.lookX * W * 0.5 * rig.mouseRange, y: nav.lookY * H * 0.5 * rig.mouseRange, z: rig.refEyeZ };
  }
  if (mode === 'webcam') {
    const e = tracker.update({ ...rig, camX: 0, camY: H / 2 + rig.camAboveTop, flipX: false });
    if (e) return { x: e.x * rig.gain, y: e.y * rig.gain, z: e.z };
  }
  return {
    x: mouse.x * W * 0.5 * rig.mouseRange,
    y: mouse.y * H * 0.5 * rig.mouseRange,
    z: rig.refEyeZ * mouse.zScale,
  };
}

function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min((now - lastFrame) / 1000, 0.1);
  lastFrame = now;
  const eye = currentEye(dt);
  world.position.z = nav.dolly; // walking = moving the room toward the viewer
  const { W, H } = screenSize();
  setOffAxisCamera(camera, eye, W, H);
  renderer.render(world, camera);

  frames++;
  if (now - fpsT > 500) { fps = (frames * 1000) / (now - fpsT); frames = 0; fpsT = now; }
  if (!hud.classList.contains('hidden')) {
    const g = gestures.state;
    const face = mode === 'webcam' ? `  face ${tracker.faceVisible ? 'yes' : 'NO'}  detect ${tracker.lastDetectMs.toFixed(1)} ms`
      : mode === 'gesture' ? `  ${gestures.connected ? `gesture ${g.gesture}  cam ${g.fps ?? '-'} fps` : 'gesture server not connected'}  walk ${nav.dolly.toFixed(2)} m` : '';
    hud.textContent = `${mode}  ${fps.toFixed(0)} fps${face}\neye  x ${eye.x.toFixed(3)}  y ${eye.y.toFixed(3)}  z ${eye.z.toFixed(3)} m`;
  }
}

// ---------- boot ----------
(async () => {
  resize();
  buildSliders();
  setMode('mouse');
  setLayerView('both');
  const scenes = await listScenes();
  const sel = document.getElementById('scene');
  sel.innerHTML = scenes.map((s) => `<option>${s}</option>`).join('');
  if (!scenes.length) {
    msg.innerHTML = 'No scenes yet.<br>Run <code>python -m pipeline.prepare_scene photos/hotel_room.jpg --name hotel_room</code>, then reload.';
  } else {
    const want = new URLSearchParams(location.search).get('scene');
    const name = scenes.includes(want) ? want : scenes[0];
    sel.value = name;
    await loadScene(name).catch((err) => { console.error(err); msg.textContent = `Failed to load ${name}: ${err.message}`; });
  }
  requestAnimationFrame(frame);
})();
