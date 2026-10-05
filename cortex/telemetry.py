"""Live numbers about the current run, for the app window: what Cortex sees, what it just did, and how fast.

Runners update the module-level ``LIVE`` object; the UI polls ``LIVE.snapshot()``.
Everything is cheap and thread-safe, so it's fine to call from the hot loop.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

THUMB_WIDTH = 360


class Telemetry:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "_lock", threading.Lock()):
            self.brain = ""            # "Gemini", "Claude", "Taught skill", "Fast 2D"
            self.reaction_ms = None    # local loop: screen -> keypress
            self.think_ms = None       # AI round-trip
            self.actions = 0
            self.busy_pct = None       # share of the time keys/mouse were doing something (pipelining)
            self.thought = ""          # the AI's latest one-line plan
            self.log: deque[str] = deque(maxlen=12)
            self._frame: np.ndarray | None = None
            self.frame_id = 0
            self._last_frame_t = 0.0

    def action(self, text: str) -> None:
        with self._lock:
            self.actions += 1
            self.log.appendleft(f"{time.strftime('%H:%M:%S')}  {text}")

    def frame(self, frame: np.ndarray, every_s: float = 0.4) -> None:
        """Keep a small copy of the latest frame (at most every ``every_s`` seconds)."""
        now = time.monotonic()
        if now - self._last_frame_t < every_s or frame is None or frame.ndim != 3:
            return
        self._last_frame_t = now
        step = max(1, frame.shape[1] // THUMB_WIDTH)
        small = np.ascontiguousarray(frame[::step, ::step, :3])
        with self._lock:
            self._frame = small
            self.frame_id += 1

    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "brain": self.brain,
                "reaction_ms": None if self.reaction_ms is None else round(self.reaction_ms),
                "think_ms": None if self.think_ms is None else round(self.think_ms),
                "actions": self.actions,
                "busy_pct": None if self.busy_pct is None else round(self.busy_pct),
                "thought": self.thought,
                "log": list(self.log),
                "frame_id": self.frame_id,
            }


LIVE = Telemetry()
