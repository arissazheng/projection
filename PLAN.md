# Room Window project plan

## Project scope

Use one accessible room to evaluate two ways to explore a capture. The multi-view walkthrough is the first milestone. The single-photo viewer supports the head-tracked projection experiment. The projector can show the walkthrough as a second display. Object interaction is a later milestone.

## Architecture

### Multi-view walkthrough

1. Sample video frames with FFmpeg or normalize the uploaded photos.
2. Use COLMAP to match overlapping images, recover camera poses, and correct lens distortion.
3. Train a Gaussian splat with Brush. Publish it with the starting camera and load it in SuperSplat.

The local server queues builds one at a time and saves finished scenes. The browser renders the saved model as the visitor moves the camera. Capture coverage determines which positions and directions render clearly.

### Single-photo projection

1. Estimate relative disparity with Depth Anything V2.
2. Extend the far surface behind depth edges and soften the filled colors. Store the foreground and background textures and disparity grids.
3. Build two textured meshes in Three.js. Drop foreground triangles across depth edges so that the background layer shows through.
4. Use the mouse, MediaPipe head tracking, or the hand gesture stream to update the viewpoint. Apply a four-corner homography to align the image with the projection surface.

The screen is centered at `z = 0`. The positive x-axis points right, the positive y-axis points up, and the viewer stands at positive z. The reference eye position `(0, 0, refEyeZ)` reproduces the source photo crop. The near and far distances map normalized disparity to the display geometry. The projection is calibrated for one viewer, and larger movements reveal more of the filled background.

## Evaluation

- For the walkthrough, check where the recovered cameras cover the room and inspect continuous motion between captured views. Record the number of aligned images and the preprocessing and training times.
- For novel-view evaluation, take reference photos from measured floor marks. Render from the same positions and compare visual quality and image metrics such as PSNR, SSIM, or LPIPS.
- For projection, measure the displayed width and camera offset. Record the viewer and display together to estimate motion-to-photon latency. Report the mean and 95th percentile.
- If collecting viewer feedback, compare the single-photo parallax view with a flat photo. Use external datasets for depth evaluation or generalization checks.

## Object interaction

Start with two to four preselected objects. Segment them with SAM 2, reconstruct or inpaint the surfaces behind them, and fit a floor plane for translation. Compare the rule-based hand tracker with a small classifier for grab, release, and rotation. Use a separate 3D asset when an object needs to rotate far enough to reveal an obscured side.

See the [README](README.md) for setup and controls.
