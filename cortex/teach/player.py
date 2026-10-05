"""Play a learned skill: look at the screen, find the most similar moment you
recorded, do what you did next for half a second, and repeat."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from cortex.control.input_mac import KEYCODES

from .skill import FPS, NearestMomentPolicy, Skill, Tick, shrink

log = logging.getLogger(__name__)
MOUSE = {"mouse_left": "left", "mouse_right": "right"}


@dataclass
class PlayResult:
    reason: str      # "stopped" | "lost" | "time_limit"
    summary: str
    seconds: float


class TickPlayer:
    """Applies recorded Ticks to the keyboard/mouse, keeping held keys in sync."""

    def __init__(self, io, to_screen: Callable[[float, float], tuple[float, float]]):
        self.io = io
        self.to_screen = to_screen
        self.held: set[str] = set()

    def _set(self, key: str, down: bool) -> None:
        if key in MOUSE:
            x, y = self.to_screen(0.5, 0.5)
            self.io.mouse_button(x, y, MOUSE[key], down)
        else:
            self.io.key(key, down)

    def apply(self, tick: Tick) -> None:
        want = {k for k in tick.held if k in KEYCODES or k in MOUSE}
        for k in sorted(self.held - want):
            self._set(k, False)
        for k in sorted(want - self.held):
            self._set(k, True)
        self.held = want
        if tick.dx or tick.dy:
            self.io.mouse_delta(tick.dx, tick.dy)
        for c in tick.clicks:
            x, y = self.to_screen(c["x"], c["y"])
            self.io.mouse_move(x, y)
            self.io.mouse_button(x, y, c["button"], True)
            self.io.mouse_button(x, y, c["button"], False)

    def release_all(self) -> None:
        for k in sorted(self.held):
            self._set(k, False)
        self.held = set()


def play_skill(
    skill: Skill,
    grab: Callable[[], np.ndarray],
    encoder,
    io,
    to_screen: Callable[[float, float], tuple[float, float]],
    should_stop: Callable[[], bool] = lambda: False,
    is_paused: Callable[[], bool] = lambda: False,
    on_status: Callable[[str], None] = lambda s: None,
    min_similarity: float = 0.5,
    lost_after_s: float = 4.0,
    max_minutes: float = 60.0,
    width: int = 256,
    chunk: int = FPS // 2,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> PlayResult:
    policy = NearestMomentPolicy(skill, chunk=chunk)
    player = TickPlayer(io, to_screen)
    start = clock()
    lost_since: float | None = None
    period = 1.0 / FPS
    on_status(f"Playing “{skill.name}”")
    try:
        while not should_stop():
            now = clock()
            if now - start > max_minutes * 60:
                return PlayResult("time_limit", f"Played “{skill.name}” for {max_minutes:.0f} minutes.", now - start)
            if is_paused():
                player.release_all()
                policy.reset()
                sleep(0.2)
                continue

            emb = encoder.encode_images(shrink(grab(), width)[None])[0]
            i, sim = policy.choose(emb)
            if sim < min_similarity:
                player.release_all()
                lost_since = lost_since or now
                on_status(f"I don't recognise this screen ({sim:.2f})…")
                if now - lost_since >= lost_after_s:
                    return PlayResult(
                        "lost",
                        f"Stopped: this screen doesn't look like anything in “{skill.name}”. "
                        "Record another example that includes it.",
                        now - start,
                    )
                sleep(period)
                continue
            lost_since = None

            for tick in policy.actions(i):
                if should_stop() or is_paused():
                    break
                t0 = clock()
                player.apply(tick)
                sleep(max(0.0, period - (clock() - t0)))
        return PlayResult("stopped", f"Stopped “{skill.name}”.", clock() - start)
    finally:
        player.release_all()
