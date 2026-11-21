"""Track hand gestures and stream them to the single-photo viewer over WebSocket."""
from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
import urllib.request
from concurrent.futures import Future
from pathlib import Path

import cv2
import numpy as np
from websockets.asyncio.server import broadcast, serve

from .classify import DESCRIPTIONS, PALM_CENTER, Debouncer, classify, fingers_extended

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
                    (18, 19), (19, 20), (0, 17)]


class Broadcaster:
    """Run WebSocket I/O on a background thread so OpenCV can own the main thread."""

    def __init__(self, host: str, port: int):
        self.host, self.port = host, port
        self.clients: set = set()
        self.loop = asyncio.new_event_loop()
        self.ready: Future[None] = Future()
        threading.Thread(target=self._run, daemon=True).start()
        self.ready.result(timeout=5)

    def _run(self):
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._serve())
        except Exception as error:
            if self.ready.done():
                raise
            self.ready.set_exception(error)
        finally:
            self.loop.close()

    async def _serve(self):
        async with serve(self._handler, self.host, self.port):
            self.ready.set_result(None)
            await asyncio.Future()

    async def _handler(self, ws):
        self.clients.add(ws)
        print(f"viewer connected ({len(self.clients)})")
        try:
            await ws.wait_closed()
        finally:
            self.clients.discard(ws)

    def send(self, msg: dict):
        data = json.dumps(msg)
        self.loop.call_soon_threadsafe(lambda: broadcast(self.clients.copy(), data))


def load_landmarker():
    from mediapipe.tasks.python import BaseOptions, vision

    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading hand model → {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(opts)


def draw(frame: np.ndarray, lm_px: np.ndarray | None, gesture: str, raw: str, fps: float) -> None:
    """Preview overlay. `frame` is already mirrored; lm_px are mirrored pixel coords."""
    h, w = frame.shape[:2]
    if lm_px is not None:
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, tuple(lm_px[a]), tuple(lm_px[b]), (255, 255, 255), 2)
        for p in lm_px:
            cv2.circle(frame, tuple(p), 4, (0, 200, 255), -1)
    cv2.rectangle(frame, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.putText(frame, f"{gesture.upper()}: {DESCRIPTIONS[gesture]}", (12, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"raw {raw}  {fps:4.1f} fps", (w - 210, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    legend = ["palm  (4 up)   look", "point (index)  forward", "peace (2 up)   back",
              "three (3 up)   reset", "fist           hold"]
    for i, line in enumerate(legend):
        active = line.startswith(gesture)
        cv2.putText(frame, line, (12, h - 16 - 22 * (len(legend) - 1 - i)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (80, 220, 255) if active else (230, 230, 230), 2 if active else 1, cv2.LINE_AA)


def run_camera(args, out: Broadcaster):
    landmarker = load_landmarker()
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera} (check camera permission for your terminal)")

    import mediapipe as mp
    deb = Debouncer(window=args.debounce)
    fps, t_prev, t0 = 0.0, time.perf_counter(), time.perf_counter()
    print(f"streaming on ws://{args.host}:{args.port}  (q / Esc to quit)")
    first_frame, fail_since = True, None
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            # macOS cameras often return empty frames while warming up or after a hiccup.
            fail_since = fail_since or time.perf_counter()
            if time.perf_counter() - fail_since > 5:
                raise SystemExit("camera opened but delivered no frames for 5 s "
                                 "(another app using it? permission revoked?)")
            time.sleep(0.05)
            continue
        fail_since = None
        if first_frame:
            print(f"camera ok: {frame.shape[1]}x{frame.shape[0]}. Hold up a hand, palm facing the camera.")
            first_frame = False
        frame = cv2.flip(frame, 1)  # mirror: your right hand moves right on screen
        h, w = frame.shape[:2]

        t_det = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                          int((t_det - t0) * 1000))
        det_ms = (time.perf_counter() - t_det) * 1000

        lm = None
        if res.hand_landmarks:
            lm = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]], dtype=np.float32)
        raw = classify(lm)
        gesture = deb.update(raw)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
        t_prev = now

        msg = {"hand": lm is not None, "gesture": gesture, "raw": raw, "fps": round(fps, 1),
               "ms": round(det_ms, 1), "x": 0.0, "y": 0.0, "fingers": [0, 0, 0, 0]}
        if lm is not None:
            # Frame is mirrored, so image x already matches the user's left/right.
            msg["x"] = round(float(lm[PALM_CENTER, 0] * 2 - 1), 4)
            msg["y"] = round(float(1 - lm[PALM_CENTER, 1] * 2), 4)
            msg["fingers"] = [int(f) for f in fingers_extended(lm)]
        out.send(msg)

        if not args.no_window:
            lm_px = None if lm is None else (lm[:, :2] * [w, h]).astype(int)
            draw(frame, lm_px, gesture, raw, fps)
            cv2.imshow("gestures (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    cap.release()
    cv2.destroyAllWindows()


def run_demo(out: Broadcaster):
    """No camera: cycle every gesture so the viewer integration can be tested."""
    script = [("palm", 3.0), ("point", 2.0), ("fist", 1.0), ("peace", 2.0), ("three", 1.0), ("none", 1.0)]
    print(f"demo mode on ws://{out.host}:{out.port}, cycling gestures (Ctrl-C to stop)")
    t0 = time.perf_counter()
    while True:
        for gesture, dur in script:
            end = time.perf_counter() + dur
            print(f"  {gesture:6s} {DESCRIPTIONS[gesture]}")
            while time.perf_counter() < end:
                t = time.perf_counter() - t0
                x, y = (0.7 * np.sin(t * 1.5), 0.4 * np.sin(t * 0.9)) if gesture == "palm" else (0.0, 0.0)
                out.send({"hand": gesture != "none", "gesture": gesture, "raw": gesture, "x": round(x, 4),
                          "y": round(y, 4), "fingers": [0, 0, 0, 0], "fps": 30.0, "ms": 0.0, "demo": True})
                time.sleep(1 / 30)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index (0 = built-in)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--debounce", type=int, default=5, help="frames of majority vote before a gesture switches")
    ap.add_argument("--no-window", action="store_true", help="don't show the preview window")
    ap.add_argument("--demo", action="store_true", help="no camera; cycle through gestures")
    args = ap.parse_args()
    if args.debounce < 1:
        ap.error("--debounce must be positive")

    try:
        out = Broadcaster(args.host, args.port)
    except (OSError, TimeoutError) as error:
        ap.error(f"Could not start the gesture server on {args.host}:{args.port}. {error}")
    try:
        run_demo(out) if args.demo else run_camera(args, out)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
