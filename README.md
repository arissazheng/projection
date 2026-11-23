# Room Window

Room Window is a CS 180 project for exploring a captured room in a browser. The multi-view walkthrough builds a 3D scene from a video or photo set. The single-photo viewer uses estimated depth to create parallax for a display or projection.

## Multi-view walkthrough

On Apple Silicon macOS, install `uv` and Node.js with `npm`. Run setup and start the server from the repository root.

```bash
bash scripts/setup_walkthrough.sh
.walkthrough-venv/bin/python -m pipeline.walkthrough
```

Open [Room Walkthrough](http://127.0.0.1:8181/). Select one video or 12–500 overlapping photos of the same room, then click **Build walkthrough**. Leave the server running while it aligns the views and trains the scene. When the scene is ready, click **Open walkthrough**.

| Control | Action |
| --- | --- |
| `2` | Enter Fly mode |
| `W` `A` `S` `D` | Move the camera |
| Drag or press `G` | Look around with the mouse |
| `Q` `E` | Change height with mouse look enabled |
| `R` / `Esc` | Reset the camera / release the mouse |

**Hand gestures.** With `.venv/bin/python -m gestures.server` running, the walkthrough viewer follows the same gestures as the photo viewer (see below). Point to move forward, show two fingers to move back, open your palm and move it off center to turn and look up or down, show three fingers to return to the starting view, and make a fist to stop. The bridge holds the arrow keys and Ctrl (slow) for you, and stops movement about 4 units from the start so you stay inside the captured room. Add `?nogestures` to the viewer URL to turn it off.

Move slowly through the room while recording. Keep nearby views overlapping and film toward the angles you want to explore. Sparsely captured angles can smear, and Fly mode allows the camera to pass through surfaces.

Processing runs locally. The app saves finished scenes in `.walkthrough-data/` so they reopen without training. Use `--steps 10000` when starting the server to train new imports longer, or `--steps 2000` for a quick preview of a new capture.

For video, the importer decodes three candidate frames for each one it keeps and keeps the sharpest. Camera alignment tries COLMAP's global mapper (GLOMAP) first and falls back to incremental mapping if global places fewer than 70% of the frames. Set `WALKTHROUGH_MAPPER=incremental` before starting the server to skip the global mapper. The server log prints a `timing` line for each stage.

## Single-photo projection

Run setup and prepare a scene from the repository root. Substitute your own image path.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pipeline.prepare_scene path/to/room.jpg --name room
python3 -m http.server 8180 -d viewer
```

Open [Room Window](http://localhost:8180/) and select the scene. The first preparation run downloads the depth model. Use `--model small|base|large`, `--near`, `--far`, and `--edge-threshold` to tune the result. The generated folder in `viewer/scenes/` contains disparity and layer diagnostics.

| Control | Action |
| --- | --- |
| Mouse / scroll | Move the virtual eye / adjust distance |
| `M` | Cycle through Mouse, Head, and Gestures |
| `0` | Reset the mouse and gesture viewpoint |
| `1` `2` `3` | Show both layers, foreground only, or background only |
| `C` | Start or finish projector calibration |
| `H` / `F` | Toggle the controls / toggle fullscreen |

### Head tracking and gestures

Use **Head** to track the viewer through the browser camera. For hand gestures, run `.venv/bin/python -m gestures.server` in a second terminal and choose **Gestures**. Hold your palm about 40–80 cm from the camera and face it toward the lens. Add `--demo` to cycle through gestures without a camera.

| Gesture | Action |
| --- | --- |
| Open palm | Move your hand to steer the view |
| Index finger | Move forward |
| Index and middle fingers | Move back |
| Index, middle, and ring fingers | Reset the view |
| Fist or no hand | Hold the view |

### Projector calibration

1. Measure the displayed image and set **Screen width** in meters. Set **Reference eye distance** to the usual viewing distance. To match the photo's horizontal field of view, use `screenWidth / (2 * tan(photoHFOV / 2))`.
2. In **Head** mode, set **Webcam FOV**, **Webcam above screen top**, and **Webcam z offset**. Check the eye position in the HUD from a measured spot.
3. Tune **Near distance** and **Far distance** to match the room's proportions. The depth estimate has relative scale, so a measured wall distance helps.
4. Press `C` and drag the corners onto the physical display rectangle. Use `Tab` to change corners, arrow keys to nudge, and `R` to reset the corners.

## Repository layout

| Path | Contents |
| --- | --- |
| `pipeline/` | Capture reconstruction, scene preparation, and the local walkthrough server |
| `viewer/walkthrough/` | Walkthrough importer and scene library |
| `viewer/index.html`, `viewer/style.css`, `viewer/js/` | Single-photo viewer, tracking, and calibration |
| `gestures/` | Hand classification, camera preview, and WebSocket server |
| `scripts/setup_walkthrough.sh` | Walkthrough dependency setup |

See the [project plan](PLAN.md) for architecture and evaluation.
