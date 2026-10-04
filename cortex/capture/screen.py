"""Game window capture on macOS.

Uses Quartz to find the game window and mss to grab its pixels. This needs the
Screen Recording permission (System Settings → Privacy & Security → Screen
Recording). mss returns physical pixels, so on Retina displays frames are 2x
the window's size in points. ``to_screen`` converts back to points for mouse input.

A ScreenCaptureKit streaming backend could cut capture latency further. It can
replace this class as long as it keeps the same interface.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class WindowInfo:
    x: float
    y: float
    width: float
    height: float
    window_id: int


def find_window(owner: str) -> WindowInfo | None:
    import Quartz

    opts = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    best = None
    for w in Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID) or []:
        if w.get("kCGWindowOwnerName") != owner or w.get("kCGWindowLayer", 0) != 0:
            continue
        b = w["kCGWindowBounds"]
        info = WindowInfo(b["X"], b["Y"], b["Width"], b["Height"], int(w["kCGWindowNumber"]))
        if best is None or info.width * info.height > best.width * best.height:
            best = info
    return best


class WindowCapture:
    def __init__(self, owner: str, refresh_s: float = 1.0):
        import mss

        self.owner = owner
        self.refresh_s = refresh_s
        self._sct = mss.mss()
        self._win: WindowInfo | None = None
        self._last_lookup = 0.0
        self.scale = 1.0

    @property
    def window(self) -> WindowInfo:
        now = time.monotonic()
        if self._win is None or now - self._last_lookup > self.refresh_s:
            self._win = find_window(self.owner)
            self._last_lookup = now
            if self._win is None:
                raise RuntimeError(f"no on-screen window owned by {self.owner!r}; is the game running?")
        return self._win

    def grab(self) -> np.ndarray:
        """Return the game window as an (H, W, 3) RGB array. This is a view, so no copy is made."""
        w = self.window
        shot = self._sct.grab({"left": int(w.x), "top": int(w.y), "width": int(w.width), "height": int(w.height)})
        bgra = np.frombuffer(shot.raw, np.uint8).reshape(shot.height, shot.width, 4)
        self.scale = shot.width / max(w.width, 1)
        return bgra[..., 2::-1]  # BGRA -> RGB view

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        w = self.window
        return w.x + x / self.scale, w.y + y / self.scale
