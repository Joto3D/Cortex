"""Low-level keyboard and mouse injection on macOS via Quartz CGEvents.

Requires the Accessibility permission for your terminal or Python
(System Settings → Privacy & Security → Accessibility).
"""
from __future__ import annotations

# macOS virtual key codes (ANSI layout).
KEYCODES: dict[str, int] = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "=": 24, "9": 25, "7": 26,
    "-": 27, "8": 28, "0": 29, "o": 31, "u": 32, "i": 34, "p": 35, "l": 37, "j": 38,
    "k": 40, "n": 45, "m": 46,
    "return": 36, "tab": 48, "space": 49, "delete": 51, "escape": 53, "shift": 56,
    "left": 123, "right": 124, "down": 125, "up": 126,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100,
    "f9": 101, "f10": 109, "f11": 103, "f12": 111,
}


class MacInput:
    def __init__(self):
        import Quartz  # pyobjc-framework-Quartz

        self.Q = Quartz
        self._src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)

    def key(self, name: str, down: bool) -> None:
        code = KEYCODES[name.lower()]
        ev = self.Q.CGEventCreateKeyboardEvent(self._src, code, down)
        self.Q.CGEventPost(self.Q.kCGHIDEventTap, ev)

    def mouse_move(self, x: float, y: float) -> None:
        Q = self.Q
        ev = Q.CGEventCreateMouseEvent(self._src, Q.kCGEventMouseMoved, (x, y), Q.kCGMouseButtonLeft)
        Q.CGEventPost(Q.kCGHIDEventTap, ev)

    def mouse_button(self, x: float, y: float, button: str, down: bool) -> None:
        Q = self.Q
        if button == "left":
            kind = Q.kCGEventLeftMouseDown if down else Q.kCGEventLeftMouseUp
            btn = Q.kCGMouseButtonLeft
        else:
            kind = Q.kCGEventRightMouseDown if down else Q.kCGEventRightMouseUp
            btn = Q.kCGMouseButtonRight
        ev = Q.CGEventCreateMouseEvent(self._src, kind, (x, y), btn)
        Q.CGEventPost(Q.kCGHIDEventTap, ev)


def frontmost_app_name() -> str | None:
    try:
        from AppKit import NSWorkspace  # pyobjc-framework-Cocoa
    except ImportError:
        return None
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    return str(app.localizedName()) if app is not None else None
