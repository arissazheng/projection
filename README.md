# Room Window

CS 180 final project: turn a single photo of a room into a head-tracked wall projection you can "look into." See **[PLAN.md](PLAN.md)** for the build plan, milestones and ownership.

```
photo ──► Depth Anything V2 ──► foreground + inpainted background layers ──► browser viewer
                                                                              ├─ off-axis projection from eye position
                                                                              ├─ eye = mouse  or  webcam (MediaPipe)
                                                                              └─ 4-corner pin for the projector
```

## Setup

```bash
uv venv --python 3.12 .venv          # or: python3 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

The first depth run downloads the model (~100 MB for `small`).

## Make a scene

From a photo (our first test image is `photos/hotel_room.jpg`):

```bash
.venv/bin/python -m pipeline.prepare_scene photos/hotel_room.jpg --name hotel_room --model base --near 1 --far 15
```

Options: `--model small|base|large`, `--near`/`--far` (meters from the eye to the nearest/farthest point), `--edge-threshold`. Each scene folder also gets `debug_disparity.png` and `debug_holes.png` (yellow = depth edges, red = regions the background layer fills in), so check those first when something looks wrong.

## Run the viewer

```bash
python3 -m http.server 8180 -d viewer
```

Open http://localhost:8180. The webcam needs `localhost` or https.

## Navigate with hand gestures (laptop camera)

In a second terminal:

```bash
.venv/bin/python -m gestures.server
```

This opens a mirrored camera preview showing the detected gesture, and streams it to the viewer at `ws://localhost:8765`. In the viewer, click **Gestures** (or press `M` until the HUD says `gesture`). On macOS, the first run asks for camera permission for your terminal app.

Hold your hand up, palm facing the camera, about 40–80 cm away:

| Gesture | Fingers up | What it does |
|---|---|---|
| ✋ **Open palm** | index, middle, ring, pinky | **Look around:** move your hand left/right/up/down to steer the viewpoint |
| ☝️ **Point** | index only | **Walk forward** into the room (Walk speed slider) |
| ✌️ **Peace** | index + middle | **Walk back** (stops at the spot the photo was taken from) |
| 🤟 **Three** | index + middle + ring | **Reset** to the original photo view |
| ✊ **Fist** | none | **Hold:** freeze the view so you can move your hand without moving the view |
| no hand / anything else | | Hold |

How it works ([gestures/classify.py](gestures/classify.py)): MediaPipe finds 21 hand landmarks. A finger counts as "up" when its tip is ≥1.15× farther from the wrist than its middle joint. The thumb is ignored because it's unreliable when the hand rotates. A gesture only switches after it wins a majority of the last 5 frames, so brief misreads while you change poses don't trigger a walk or reset.

No camera handy? `python -m gestures.server --demo` cycles through every gesture so you can test the viewer side.

## Viewer controls

| Key | |
|---|---|
| mouse / scroll | move the virtual eye / change its distance |
| `M` | cycle mouse → head tracking (webcam, in-browser) → gestures |
| `0` | snap the eye to the reference position (shows the original photo exactly) |
| `1` `2` `3` | both layers / foreground only / background only |
| `C` | calibrate the projector: drag the corners onto the wall rectangle, arrows nudge, `Tab` next corner, `R` reset |
| `H` | hide UI · `F` fullscreen |

### Getting physically correct parallax
1. Set **Screen width** to the real width of the display or projected image, in meters.
2. Set **Reference eye distance** to where the viewer usually stands. For the least distortion it should also match the photo's field of view: `refEyeZ ≈ screenWidth / (2·tan(photoHFOV/2))`.
3. Webcam: set **Webcam FOV** (most laptop cams are 60–78°) and **Webcam above screen top** to where the camera really sits. Then stand still at a measured spot and check that the HUD's eye position matches.
4. Tune **Near/Far distance** until the room's proportions look right. Relative depth has no scale, so a tape-measured distance to the back wall helps.

## Layout

```
pipeline/   offline Python: depth.py, layers.py, scene_io.py, prepare_scene.py
gestures/   OpenCV + MediaPipe hand tracker: classify.py (gesture rules), server.py (camera → WebSocket)
viewer/     static web app: js/projection.js (off-axis math), js/scene.js (meshes),
            js/tracking.js (MediaPipe + One-Euro), js/calibration.js (homography corner pin)
photos/     input photos (gitignored; `git add -f` the ones you want to keep)
```
