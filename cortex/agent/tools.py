"""The actions Claude can take in a game, and their local execution.

Claude calls these as tools. Several calls in one turn run back-to-back
locally, so a turn like "sprint forward 2 s, turn 30°, attack" happens
without waiting on the network in between.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from cortex.control.input_mac import KEYCODES

MOUSE_BUTTONS = {"mouse_left": "left", "mouse_right": "right"}

_KEYS_HELP = (
    "Keys are action names from the controls list (e.g. 'forward', 'jump') or raw keys "
    "('w', 'space', 'shift', '1', 'mouse_left')."
)


def _tool(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required if required is not None else list(properties),
            "additionalProperties": False,
        },
    }


_KEYS = {"type": "array", "items": {"type": "string"}, "description": _KEYS_HELP}

TOOLS: list[dict] = [
    _tool(
        "hold",
        "Hold one or more keys/mouse buttons down together for some seconds, then release them. "
        "Use it to walk, sprint, jump while moving, or keep attacking/mining. " + _KEYS_HELP,
        {"keys": _KEYS, "seconds": {"type": "number", "description": "0.05 to 5 seconds"}},
    ),
    _tool(
        "tap",
        "Press and release each key in order, `times` times (e.g. open the inventory, pick hotbar slot 3, jump once).",
        {"keys": _KEYS, "times": {"type": "integer", "description": "1 to 10"}},
    ),
    _tool(
        "look",
        "Turn the camera like moving the mouse in a 3D game. Positive right_degrees turns right, "
        "positive down_degrees looks down. A full turn around is 360.",
        {"right_degrees": {"type": "number"}, "down_degrees": {"type": "number"}},
    ),
    _tool(
        "click",
        "Move the pointer to a spot on the screenshot and click it, e.g. a menu button or an item. "
        "x and y are fractions of the screenshot width and height (0 = left/top, 1 = right/bottom).",
        {
            "x": {"type": "number"},
            "y": {"type": "number"},
            "button": {"type": "string", "enum": ["left", "right"]},
            "hold_seconds": {"type": "number", "description": "0 for a normal click"},
        },
    ),
    _tool(
        "wait",
        "Do nothing for some seconds (e.g. while something loads or grows). 0.1 to 10 seconds.",
        {"seconds": {"type": "number"}},
    ),
    _tool(
        "note",
        "Write down progress or facts worth remembering (what's done, where things are). "
        "Old screenshots are forgotten; notes are kept.",
        {"text": {"type": "string"}},
    ),
    _tool(
        "finish",
        "Stop playing: the assignment is complete, or it can't be done. Explain what happened.",
        {"success": {"type": "boolean"}, "summary": {"type": "string"}},
    ),
]


class GameInput(Protocol):
    def key(self, name: str, down: bool) -> None: ...
    def mouse_move(self, x: float, y: float) -> None: ...
    def mouse_delta(self, dx: float, dy: float) -> None: ...
    def mouse_button(self, x: float, y: float, button: str, down: bool) -> None: ...


class ToolError(Exception):
    pass


@dataclass
class Outcome:
    text: str
    is_error: bool = False
    finished: bool = False
    success: bool = False


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


@dataclass
class ActionExecutor:
    """Runs tool calls against the keyboard/mouse.

    ``to_screen(fx, fy)`` maps window fractions to global screen points.
    """

    io: GameInput
    keymap: dict[str, str]
    to_screen: Callable[[float, float], tuple[float, float]]
    look_px_per_degree: float = 6.0
    sleep: Callable[[float], None] = time.sleep
    notes: list[str] = field(default_factory=list)
    _held: set[str] = field(default_factory=set)

    def resolve(self, name: str) -> str:
        k = str(name).strip().lower()
        k = self.keymap.get(k, k)
        if k in KEYCODES or k in MOUSE_BUTTONS:
            return k
        known = ", ".join(sorted(self.keymap))
        raise ToolError(f"unknown key {name!r}. Use an action name ({known}) or a raw key.")

    def _down(self, key: str) -> None:
        if key in MOUSE_BUTTONS:
            x, y = self.to_screen(0.5, 0.5)
            self.io.mouse_button(x, y, MOUSE_BUTTONS[key], True)
        else:
            self.io.key(key, True)
        self._held.add(key)

    def _up(self, key: str) -> None:
        if key in MOUSE_BUTTONS:
            x, y = self.to_screen(0.5, 0.5)
            self.io.mouse_button(x, y, MOUSE_BUTTONS[key], False)
        else:
            self.io.key(key, False)
        self._held.discard(key)

    def release_all(self) -> None:
        for k in list(self._held):
            self._up(k)

    def execute(self, name: str, args: dict) -> Outcome:
        try:
            return getattr(self, f"_do_{name}")(**args)
        except ToolError as e:
            return Outcome(str(e), is_error=True)
        except (TypeError, AttributeError) as e:
            return Outcome(f"bad call to {name}: {e}", is_error=True)
        finally:
            self.release_all()  # never leave a key stuck down

    def _do_hold(self, keys: list[str], seconds: float) -> Outcome:
        resolved = [self.resolve(k) for k in keys]
        if not resolved:
            raise ToolError("hold needs at least one key")
        t = _clamp(seconds, 0.05, 5.0)
        for k in resolved:
            self._down(k)
        self.sleep(t)
        for k in reversed(resolved):
            self._up(k)
        return Outcome(f"held {'+'.join(resolved)} for {t:.2f}s")

    def _do_tap(self, keys: list[str], times: int) -> Outcome:
        resolved = [self.resolve(k) for k in keys]
        n = int(_clamp(times, 1, 10))
        for _ in range(n):
            for k in resolved:
                self._down(k)
                self.sleep(0.05)
                self._up(k)
                self.sleep(0.05)
        return Outcome(f"tapped {', '.join(resolved)} x{n}")

    def _do_look(self, right_degrees: float, down_degrees: float) -> Outcome:
        dx = _clamp(right_degrees, -360, 360) * self.look_px_per_degree
        dy = _clamp(down_degrees, -90, 90) * self.look_px_per_degree
        # Spread the movement over a few small steps: games smooth raw mouse input.
        steps = max(1, int(max(abs(dx), abs(dy)) // 40) + 1)
        for _ in range(steps):
            self.io.mouse_delta(dx / steps, dy / steps)
            self.sleep(0.01)
        return Outcome(f"looked right {right_degrees:.0f}°, down {down_degrees:.0f}°")

    def _do_click(self, x: float, y: float, button: str, hold_seconds: float) -> Outcome:
        if button not in ("left", "right"):
            raise ToolError("button must be 'left' or 'right'")
        sx, sy = self.to_screen(_clamp(x, 0, 1), _clamp(y, 0, 1))
        self.io.mouse_move(sx, sy)
        self.sleep(0.05)
        self.io.mouse_button(sx, sy, button, True)
        self.sleep(_clamp(hold_seconds, 0.03, 5.0))
        self.io.mouse_button(sx, sy, button, False)
        return Outcome(f"{button}-clicked at ({x:.2f}, {y:.2f})")

    def _do_wait(self, seconds: float) -> Outcome:
        t = _clamp(seconds, 0.1, 10.0)
        self.sleep(t)
        return Outcome(f"waited {t:.1f}s")

    def _do_note(self, text: str) -> Outcome:
        self.notes.append(text)
        return Outcome("noted")

    def _do_finish(self, success: bool, summary: str) -> Outcome:
        return Outcome(summary, finished=True, success=bool(success))
