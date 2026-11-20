"""Rule-based hand-gesture classifier on MediaPipe's 21 hand landmarks.

GESTURES (palm facing the camera, either hand):

  name    fingers up                 meaning in the viewer
  ------  -------------------------  ------------------------------------------------
  palm    index+middle+ring+pinky    LOOK: hand position steers your viewpoint
  point   index only                 WALK FORWARD into the room
  peace   index + middle             WALK BACK out of the room
  three   index + middle + ring      RESET to the original photo view
  fist    none                       HOLD: freeze where you are (move your hand freely)
  none    no hand / anything else    HOLD

The thumb is ignored on purpose: whether it reads as "up" depends on hand rotation,
so it makes classification flaky.

A finger counts as extended when its tip is clearly farther from the wrist than its
middle (PIP) joint. That ratio doesn't change with hand size, distance to the camera,
or in-plane rotation.

This is the hand-written baseline. PLAN.md M4 replaces it with a small MLP trained
on our own labelled clips, and the baseline gives that model something to beat.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

WRIST = 0
# (pip, tip) for index, middle, ring, pinky
FINGERS = ((6, 8), (10, 12), (14, 16), (18, 20))
PALM_CENTER = 9  # middle-finger MCP: stable point for "where is the hand"

GESTURES = {
    (True, True, True, True): "palm",
    (True, False, False, False): "point",
    (True, True, False, False): "peace",
    (True, True, True, False): "three",
    (False, False, False, False): "fist",
}

DESCRIPTIONS = {
    "palm": "LOOK - move hand to steer view",
    "point": "WALK FORWARD",
    "peace": "WALK BACK",
    "three": "RESET view",
    "fist": "HOLD (freeze)",
    "none": "HOLD (no gesture)",
}


def fingers_extended(lm: np.ndarray, ratio: float = 1.15) -> tuple[bool, bool, bool, bool]:
    """lm: (21, 2|3) landmarks. Returns extended flags for index, middle, ring, pinky."""
    wrist = lm[WRIST, :2]
    out = []
    for pip, tip in FINGERS:
        d_tip = np.linalg.norm(lm[tip, :2] - wrist)
        d_pip = np.linalg.norm(lm[pip, :2] - wrist)
        out.append(bool(d_tip > ratio * d_pip))
    return tuple(out)


def classify(lm: np.ndarray | None) -> str:
    if lm is None:
        return "none"
    return GESTURES.get(fingers_extended(lm), "none")


@dataclass
class Debouncer:
    """Only switch gesture after it wins a majority of the last `window` frames.

    Stops a single misread frame (common while the hand moves between poses) from
    triggering a walk or a reset.
    """
    window: int = 5
    current: str = "none"

    def __post_init__(self):
        self._hist: deque[str] = deque(maxlen=self.window)

    def update(self, raw: str) -> str:
        self._hist.append(raw)
        label, count = Counter(self._hist).most_common(1)[0]
        if count > self.window // 2:
            self.current = label
        return self.current
