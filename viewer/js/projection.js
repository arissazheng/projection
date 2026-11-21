// Build an off-axis frustum from the eye through the edges of the screen at z = 0.
// See Kooima, "Generalized Perspective Projection" (2008).

export function setOffAxisCamera(camera, eye, screenW, screenH, near = 0.02, far = 200) {
  const ez = Math.max(eye.z, 0.05);
  const k = near / ez; // similar triangles: screen edge offset scaled onto the near plane
  const left = (-screenW / 2 - eye.x) * k;
  const right = (screenW / 2 - eye.x) * k;
  const bottom = (-screenH / 2 - eye.y) * k;
  const top = (screenH / 2 - eye.y) * k;

  camera.position.set(eye.x, eye.y, ez);
  camera.quaternion.identity(); // always look straight down -z (perpendicular to screen)
  camera.updateMatrixWorld(true);
  camera.projectionMatrix.makePerspective(left, right, top, bottom, near, far);
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
}
