"""Classify hand gestures using MediaPipe landmarks and a rolling majority vote."""
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
    """Compare fingertip and middle-joint distances from the wrist. Ignore the thumb."""
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
    """Switch gestures after a majority of the last `window` frames agrees."""
    window: int = 5
    current: str = "none"

    def __post_init__(self) -> None:
        self._hist: deque[str] = deque(maxlen=self.window)

    def update(self, raw: str) -> str:
        self._hist.append(raw)
        label, count = Counter(self._hist).most_common(1)[0]
        if count > self.window // 2:
            self.current = label
        return self.current
