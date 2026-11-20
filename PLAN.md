# Build Plan: Single-Photo Walk-In Room

**Goal.** Turn one photo of one room into a wall projection that shifts its perspective as a visitor walks around, so the wall acts like a window into the room. Visitors grabbing and moving objects is the stretch goal.

**Scope decision (per GSI feedback).** We build around **one real room we can access and photograph ourselves**. Kaggle and NYU rooms are only for generalization checks and depth evaluation. Interactive objects are **2–4 pre-selected objects** processed offline.

---

## Architecture

```
            OFFLINE (Python, once per photo)                         RUNTIME (browser, real time)
 ┌──────────────────────────────────────────────────┐     ┌──────────────────────────────────────────────┐
 │ photo.jpg                                        │     │ viewer/index.html  (Three.js + MediaPipe)    │
 │   └─ Depth Anything V2 ──► disparity             │     │                                              │
 │        └─ depth edges ──► foreground mask        │     │  head pose ◄── webcam (FaceLandmarker)       │
 │             └─ dilate far depth + inpaint color  │ ──► │      │        or mouse (debug)               │
 │                  ──► background layer            │     │      ▼                                       │
 │ (stretch) SAM 2 objects ─► per-object RGBA +     │     │  off-axis projection (screen = window)       │
 │           inpainted hole + image-to-3D mesh      │     │      ▼                                       │
 │                                                  │     │  render FG mesh + BG mesh                    │
 │ writes viewer/scenes/<name>/…                    │     │      ▼                                       │
 └──────────────────────────────────────────────────┘     │  corner-pin homography (projector keystone)  │
                                                          └──────────────────────────────────────────────┘
```

**Why the split.** Everything slow (depth, segmentation, inpainting, 3D generation) runs once, offline. At runtime the work is only head tracking and drawing two textured meshes, which keeps latency low. The runtime is a web page, so the same code runs on a laptop screen and full-screen on a projector, and MediaPipe runs in the browser without a Python↔GPU bridge.

### Coordinate conventions (everyone uses these)
- Units are **meters**. The **screen (or projected image) is the plane z = 0**, centered at the origin. +x is right, +y is up, and the viewer stands at +z.
- The room lives behind the screen (z < 0).
- The photo is back-projected from a **reference eye position** `(0, 0, refEyeZ)`. A viewer standing exactly there sees the original photo. Moving away from it produces parallax.
- Depth model output is relative disparity. We map it to metric distance with two knobs, `nearDist` and `farDist` (meters from the reference eye to the nearest and farthest points). Both live in `meta.json` and can be tuned live in the viewer.

---

## Milestones

### M0: Setup (day 1–2)
- [x] Repo, venv, `requirements.txt`
- [x] First test photo: `photos/hotel_room.jpg` → scene `hotel_room`
- [ ] Everyone runs the viewer locally (see README)

### M1: MVP on screen, mouse-driven (week 1) ← **built in this commit**
- [x] `pipeline/prepare_scene.py`: photo → Depth Anything V2 → disparity → FG/BG layers → scene folder
- [x] Background layer: far depth dilated into near regions at depth edges, color hole filled with OpenCV inpainting
- [x] Viewer: displaced-grid meshes, edge triangles dropped from the FG mesh, off-axis projection
- [x] Mouse moves the virtual eye; scroll changes distance
- [ ] Photograph **our room**: one hero photo at eye height plus 6–10 photos from taped floor marks for evaluation
- [ ] Tune `nearDist` / `farDist` / edge threshold for our room

### M1.5: Gesture navigation for testing ← **built**
- [x] `gestures/server.py`: OpenCV camera → MediaPipe Hands → rule-based gestures (palm = look, point = forward, peace = back, three = reset, fist = hold) → WebSocket
- [x] Viewer "Gestures" mode turns them into look + walk
- [ ] Try on each teammate's laptop. Tune the finger ratio (1.15) and debounce (5 frames) if gestures misfire
- [ ] Collect labelled clips, then train the MLP classifier (M4) and compare it against these rules

### M2: Webcam head tracking (week 2) ← **first version built**
- [x] MediaPipe FaceLandmarker → eye midpoint → metric eye position (from interpupillary distance + webcam FOV)
- [x] One-Euro smoothing filter
- [ ] Calibrate webcam offset relative to the screen center, and the webcam FOV
- [ ] Measure motion-to-photon latency (240 fps phone video showing head and screen together)

### M3: Projector (week 3) ← **corner-pin built**
- [x] 4-corner pin: drag the corners to the projected wall rectangle, apply as a homography (CSS `matrix3d`)
- [ ] Measure the physical projected width/height → `screenWidth` / `screenHeight`
- [ ] Mount the webcam at the projection wall, measure its offset, re-calibrate
- [ ] (Nice) automatic calibration: project a checkerboard, detect it with OpenCV, solve the homography

### M4: Stretch — interactive objects (week 3–4)
- [ ] SAM 2 masks for 2–4 chosen objects (offline, clicked prompts)
- [ ] Per-object hole fill: **compare** classical (OpenCV/Poisson) vs. LaMa on the same holes → writeup figure
- [ ] Fit a floor plane (RANSAC on back-projected points); objects slide along it, so perspective scaling comes for free
- [ ] MediaPipe Hands → small MLP gesture classifier (grab / release / rotate) trained on our own labelled clips
- [ ] Rotation: small angles with the depth mesh; full rotation by swapping in an image-to-3D mesh (TripoSR / Stable Fast 3D)

### M5: Evaluation + writeup (final week)
- [ ] **Novel-view accuracy:** render from each taped floor mark and compare against the real photo taken there (PSNR / SSIM / LPIPS). Compare inpainting variants.
- [ ] **Latency:** motion-to-photon ms (mean, p95)
- [ ] **Depth (optional):** Depth Anything V2 relative vs. metric-indoor vs. our NYU fine-tune on NYU test (AbsRel, δ1)
- [ ] **(Optional upper bound)** nerfstudio `splatfacto` from ~50 photos of the same room, scored with the same held-out-view metric
- [ ] **User study:** 5–10 people rate realism 1–5, single photo vs. flat image

**MVP = M1 + M2 + M3.** If M4 slips, we still have a convincing walkable projection.

---

## Suggested ownership (4 people)

| Area | Owns | Key files |
|---|---|---|
| Depth + layers | depth tuning, inpainting comparison, NYU eval | `pipeline/` |
| Rendering | mesh quality, edge handling, off-axis math | `viewer/js/scene.js`, `viewer/js/projection.js` |
| Tracking + latency | webcam calibration, smoothing, latency measurement | `viewer/js/tracking.js` |
| Projector + interaction | calibration, SAM 2 objects, gesture classifier | `viewer/js/calibration.js`, `pipeline/objects/` (TBD) |

## Course-concept tie-ins (for the writeup)
- **Homographies:** projector corner-pin / keystone correction
- **Perspective projection:** back-projection of the photo; off-axis frustum per eye position; perspective scaling of moved objects
- **Image blending / hole filling:** background layer, object holes (classical vs. learned)

## Known limitations to state up front
- Head-coupled perspective is correct for **one viewer** at a time.
- A single photo has no information behind objects. Large head movements reveal inpainted regions, so the comfortable range is about ±0.5 m.
- Relative depth has unknown scale, so near/far are tuned by hand per room (or measured with a tape measure).
