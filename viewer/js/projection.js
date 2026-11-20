// Head-coupled ("off-axis") perspective.
//
// The physical screen is the rectangle [-W/2, W/2] x [-H/2, H/2] on the plane z = 0
// (meters). The viewer's eye is at `eye` (z > 0, in front of the screen). We build an
// asymmetric frustum from the eye through the four screen edges, so the screen acts as
// a window into the virtual room behind it. Moving the eye changes which part of the
// room you see, which is the whole parallax effect.
//
// Reference: Kooima, "Generalized Perspective Projection" (2008). With the screen
// axis-aligned at z = 0, it reduces to the simple formulas below.

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
