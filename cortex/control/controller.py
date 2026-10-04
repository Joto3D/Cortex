"""Turns planner actions into key and mouse events.

Movement is *held* (key down until the direction changes), the way a human
plays. Clicks use a short randomised press duration so they read as human input.
"""
from __future__ import annotations

import random
import time
from typing import Protocol

from cortex.config import Profile
from cortex.planner.actions import Action, Done, Interact, Move, Stop, UseTool, Wait


class InputBackend(Protocol):
    def key(self, name: str, down: bool) -> None: ...
    def mouse_move(self, x: float, y: float) -> None: ...
    def mouse_button(self, x: float, y: float, button: str, down: bool) -> None: ...


class Controller:
    def __init__(self, profile: Profile, backend: InputBackend, to_screen=lambda x, y: (x, y), sleep=time.sleep):
        """``to_screen`` maps frame pixels to global screen points (window offset and Retina scale)."""
        self.profile = profile
        self.c = profile.controls
        self.io = backend
        self.to_screen = to_screen
        self.sleep = sleep
        self.held: str | None = None   # movement key currently held down
        self.slot: str | None = None   # last selected hotbar key

    def _jitter(self) -> None:
        lo, hi = self.c.jitter_ms
        self.sleep(random.uniform(lo, hi) / 1000.0)

    def _tap(self, key: str) -> None:
        self.io.key(key, True)
        self._jitter()
        self.io.key(key, False)

    def release_all(self) -> None:
        if self.held:
            self.io.key(self.held, False)
            self.held = None

    def apply(self, action: Action) -> None:
        if isinstance(action, Move):
            key = getattr(self.c, action.direction)
            if key != self.held:
                self.release_all()
                self.io.key(key, True)
                self.held = key
        elif isinstance(action, (Stop, Wait, Done)):
            self.release_all()
        elif isinstance(action, UseTool):
            self.release_all()
            slot = self.profile.tools[action.tool]
            if slot != self.slot:
                self._tap(slot)
                self.slot = slot
            self._click(action.screen_px, "left")
        elif isinstance(action, Interact):
            self.release_all()
            self._click(action.screen_px, "right")
        else:  # pragma: no cover
            raise TypeError(f"unknown action {action!r}")

    def _click(self, frame_px: tuple[float, float], button: str) -> None:
        x, y = self.to_screen(*frame_px)
        self.io.mouse_move(x, y)
        self._jitter()
        self.io.mouse_button(x, y, button, True)
        self._jitter()
        self.io.mouse_button(x, y, button, False)
