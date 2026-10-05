"""Record a demonstration: screen frames plus what you pressed, 10 times a second.

Only input that happens while the game window is in front is recorded, so nothing
you type in other apps ends up in a recording. The kill-switch / pause keys and
anything combined with ⌘ are never recorded.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np

from .skill import FPS, Skill, Tick, shrink

log = logging.getLogger(__name__)

MAX_SECONDS = 20 * 60  # safety cap on one recording


class InputLog:
    """Thread-safe accumulator of input events, sampled into Ticks.

    It always tracks key state (so a key pressed just after switching to the game
    counts), but the Recorder only *keeps* ticks sampled while the game is in front.
    """

    def __init__(self, ignore: set[str] | None = None):
        self._lock = threading.Lock()
        self.ignore = ignore or set()
        self.held: set[str] = set()
        self._pressed_this_tick: set[str] = set()
        self._dx = 0.0
        self._dy = 0.0
        self._clicks: list[dict] = []
        self._down_at: dict[str, tuple[float, float]] = {}

    def key(self, name: str | None, down: bool) -> None:
        if not name or name in self.ignore:
            return
        with self._lock:
            if "cmd" in self.held and name != "cmd":
                return  # ⌘-shortcuts are for the Mac, not the game
            if down:
                self.held.add(name)
                self._pressed_this_tick.add(name)
            else:
                self.held.discard(name)

    def mouse_move(self, dx: float, dy: float) -> None:
        with self._lock:
            self._dx += dx
            self._dy += dy

    def mouse_button(self, button: str, down: bool, fx: float, fy: float) -> None:
        name = f"mouse_{button}"
        with self._lock:
            if down:
                self.held.add(name)
                self._pressed_this_tick.add(name)
                self._down_at[name] = (fx, fy)
            elif name in self.held:
                self.held.discard(name)
                if name in self._pressed_this_tick:
                    # pressed and released within one tick: keep it as a positioned click
                    x, y = self._down_at.get(name, (fx, fy))
                    self._clicks.append({"button": button, "x": round(x, 4), "y": round(y, 4)})

    def sample(self) -> Tick:
        """Close the current tick and return what happened during it."""
        with self._lock:
            quick = {c["button"] for c in self._clicks}
            held = sorted(
                (self.held | self._pressed_this_tick) - {f"mouse_{b}" for b in quick} - {"cmd"}
            )
            tick = Tick(held=held, dx=round(self._dx, 2), dy=round(self._dy, 2), clicks=list(self._clicks))
            self._pressed_this_tick.clear()
            self._dx = self._dy = 0.0
            self._clicks.clear()
            return tick

    def release_all(self) -> None:
        with self._lock:
            self.held.clear()


class Recorder:
    """Records frames + input ticks while ``is_focused()`` is true.

    ``grab()`` returns the game window as RGB. Input arrives via ``log`` (the app wires
    real macOS listeners to it with :func:`start_listeners`; tests feed it directly).
    """

    def __init__(
        self,
        grab: Callable[[], np.ndarray],
        is_focused: Callable[[], bool],
        log_: InputLog,
        width: int = 256,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.grab = grab
        self.is_focused = is_focused
        self.log = log_
        self.width = width
        self.sleep = sleep
        self.clock = clock
        self.frames: list[np.ndarray] = []
        self.ticks: list[Tick] = []
        self._stop = threading.Event()

    @property
    def seconds(self) -> float:
        return len(self.ticks) / FPS

    def stop(self) -> None:
        self._stop.set()

    def step(self) -> bool:
        """Record one tick. Returns False while the game isn't focused (nothing recorded)."""
        if not self.is_focused():
            self.log.release_all()
            self.log.sample()  # discard anything gathered while another app was in front
            return False
        frame = shrink(self.grab(), self.width)
        self.frames.append(frame)
        self.ticks.append(self.log.sample())
        return True

    def run(self, max_seconds: float = MAX_SECONDS) -> None:
        period = 1.0 / FPS
        while not self._stop.is_set() and self.seconds < max_seconds:
            t0 = self.clock()
            self.step()
            self.sleep(max(0.0, period - (self.clock() - t0)))

    def to_skill(self, name: str, game: str, encoder, existing: Skill | None = None) -> Skill:
        """Fingerprint the recorded frames and return (or extend) the skill."""
        # Trim idle ticks at the very start and end (time spent switching windows).
        first = next((i for i, t in enumerate(self.ticks) if not t.idle), None)
        if first is None:
            raise ValueError("nothing was recorded: no keys or mouse input while the game was in front")
        last = max(i for i, t in enumerate(self.ticks) if not t.idle)
        frames, ticks = self.frames[first : last + 1], self.ticks[first : last + 1]
        emb = encoder.encode_images(np.stack(frames)).astype(np.float32)
        if existing is not None:
            existing.add_recording(emb, ticks)
            return existing
        return Skill(name, game, emb, ticks, np.zeros(len(ticks), np.int32))


# --- macOS input listeners ------------------------------------------------------

_SPECIAL = {
    "space": "space", "enter": "return", "tab": "tab", "esc": "escape", "backspace": "delete",
    "shift": "shift", "shift_l": "shift", "shift_r": "shift", "ctrl": "ctrl", "ctrl_l": "ctrl", "ctrl_r": "ctrl",
    "alt": "alt", "alt_l": "alt", "alt_r": "alt", "cmd": "cmd", "cmd_l": "cmd", "cmd_r": "cmd",
    "up": "up", "down": "down", "left": "left", "right": "right", "caps_lock": "capslock",
}


def pynput_key_name(key) -> str | None:
    """pynput Key/KeyCode -> Cortex key name (see control/input_mac.KEYCODES)."""
    char = getattr(key, "char", None)
    if char:
        return char.lower()
    name = getattr(key, "name", None)
    if name is None:
        return None
    if name.startswith("f") and name[1:].isdigit():
        return name
    return _SPECIAL.get(name)


def start_listeners(
    log_: InputLog,
    to_fraction: Callable[[float, float], tuple[float, float]],
    stop_key: str | None = None,
    on_stop_key: Callable[[], None] | None = None,
):
    """Attach real keyboard/mouse listeners (macOS). Returns a stop() function.

    Pressing ``stop_key`` (e.g. "f12") calls ``on_stop_key``, so you can end a recording from inside the game.
    """
    from pynput import keyboard, mouse

    def on_press(key):
        name = pynput_key_name(key)
        if stop_key and name == stop_key and on_stop_key:
            on_stop_key()
            return
        log_.key(name, True)

    def on_release(key):
        log_.key(pynput_key_name(key), False)

    def on_click(x, y, button, pressed):
        fx, fy = to_fraction(x, y)
        log_.mouse_button(button.name, pressed, fx, fy)

    def intercept(event_type, event):
        # Read raw deltas, which also work when a 3D game has locked/hidden the cursor.
        try:
            import Quartz

            if event_type in (Quartz.kCGEventMouseMoved, Quartz.kCGEventLeftMouseDragged, Quartz.kCGEventRightMouseDragged):
                dx = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGMouseEventDeltaX)
                dy = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGMouseEventDeltaY)
                log_.mouse_move(dx, dy)
        except Exception:  # never break the user's mouse
            pass
        return event

    kb = keyboard.Listener(on_press=on_press, on_release=on_release)
    ms = mouse.Listener(on_click=on_click, darwin_intercept=intercept)
    kb.start()
    ms.start()

    def stop():
        kb.stop()
        ms.stop()

    return stop
