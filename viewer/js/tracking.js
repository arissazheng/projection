// Webcam head tracking → eye position in screen coordinates (meters).
//
// MediaPipe FaceLandmarker gives 478 landmarks including both iris centers (468, 473).
// Pinhole camera model:
//   focal length (px)  f = (videoWidth / 2) / tan(hfov / 2)
//   distance           Z = f * IPD / ipd_px              (IPD ≈ 63 mm for adults)
//   lateral offset     X = (cx - videoWidth / 2) * Z / f   (same for Y)
// The webcam faces the viewer, so the viewer's right is the image's left: x flips.
// Turning the head shrinks the measured pupil separation and overestimates distance.

const MP_VERSION = '0.10.14';
const MP_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/vision_bundle.mjs`;
const MP_WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;
const FACE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const IPD_M = 0.063;

/** One-Euro filter (Casiez et al. 2012): smooth when still, responsive when moving. */
export class OneEuro {
  constructor(minCutoff = 1.0, beta = 0.05, dCutoff = 1.0) {
    Object.assign(this, { minCutoff, beta, dCutoff });
    this.x = null;
    this.dx = 0;
    this.t = null;
  }
  static alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }
  filter(x, tSec) {
    if (this.x === null) { this.x = x; this.t = tSec; return x; }
    const dt = Math.max(tSec - this.t, 1e-3);
    this.t = tSec;
    const dxRaw = (x - this.x) / dt;
    this.dx += OneEuro.alpha(this.dCutoff, dt) * (dxRaw - this.dx);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    this.x += OneEuro.alpha(cutoff, dt) * (x - this.x);
    return this.x;
  }
}

export class FaceTracker {
  constructor(videoEl) {
    this.video = videoEl;
    this.landmarker = null;
    this.startPromise = null;
    this._resetTracking();
  }

  _resetTracking() {
    this.lastVideoTime = -1;
    this.lastDetectMs = 0;
    this.filters = [new OneEuro(), new OneEuro(), new OneEuro()];
    this.lastEye = null;
    this.faceVisible = false;
  }

  start() {
    this.startPromise ??= this._openCamera().finally(() => { this.startPromise = null; });
    return this.startPromise;
  }

  async _openCamera() {
    const vision = await import(MP_URL);
    const fileset = await vision.FilesetResolver.forVisionTasks(MP_WASM);
    const landmarker = await vision.FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: FACE_MODEL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numFaces: 1,
    });
    try {
      this.video.srcObject = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 60 }, facingMode: 'user' },
        audio: false,
      });
      await this.video.play();
      this.landmarker = landmarker;
    } catch (error) {
      this.video.srcObject?.getTracks().forEach((track) => track.stop());
      this.video.srcObject = null;
      landmarker.close();
      throw error;
    }
  }

  stop() {
    if (!this.landmarker) return;
    this.video.srcObject?.getTracks().forEach((track) => track.stop());
    this.video.srcObject = null;
    this.landmarker.close();
    this.landmarker = null;
    this._resetTracking();
  }

  /**
   * @param {object} cfg { hfovDeg, camX, camY, camZ, flipX, minCutoff, beta }
   * @returns {{x:number,y:number,z:number}|null} smoothed eye position, or the last one if no new frame
   */
  update(cfg) {
    if (!this.landmarker || this.video.readyState < 2) return this.lastEye;
    if (this.video.currentTime === this.lastVideoTime) return this.lastEye;
    this.lastVideoTime = this.video.currentTime;

    const t0 = performance.now();
    const res = this.landmarker.detectForVideo(this.video, t0);
    this.lastDetectMs = performance.now() - t0;
    const lm = res.faceLandmarks?.[0];
    this.faceVisible = !!lm;
    if (!lm || lm.length < 474) return this.lastEye;

    const vw = this.video.videoWidth, vh = this.video.videoHeight;
    const a = lm[468], b = lm[473];
    const f = (vw / 2) / Math.tan((cfg.hfovDeg * Math.PI) / 360);
    const ipdPx = Math.hypot((a.x - b.x) * vw, (a.y - b.y) * vh);
    const Z = (f * IPD_M) / Math.max(ipdPx, 1);
    const cx = ((a.x + b.x) / 2) * vw, cy = ((a.y + b.y) / 2) * vh;
    const Xc = ((cx - vw / 2) * Z) / f;
    const Yc = ((cy - vh / 2) * Z) / f;

    const raw = [
      (cfg.flipX ? Xc : -Xc) + cfg.camX,
      -Yc + cfg.camY,
      Z + cfg.camZ,
    ];
    const tSec = t0 / 1000;
    const out = raw.map((v, i) => {
      const fl = this.filters[i];
      fl.minCutoff = cfg.minCutoff;
      fl.beta = cfg.beta;
      return fl.filter(v, tSec);
    });
    this.lastEye = { x: out[0], y: out[1], z: out[2] };
    return this.lastEye;
  }
}
