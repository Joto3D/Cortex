"""The API the app window's JavaScript calls (``window.pywebview.api.<method>``).

Every method returns plain JSON-able data and never raises: errors come back as
``{"ok": False, "error": "..."}`` so the window can show them. No GUI code here,
so it is unit-tested directly.
"""
from __future__ import annotations

import base64
import io
import logging
import subprocess
import threading
import webbrowser
from typing import Callable

import cortex
from cortex.config import cortex_home, load_profile
from cortex.telemetry import LIVE

from . import system
from .controller import AppController

log = logging.getLogger(__name__)

WEBSITE = "https://joto3d.github.io/Cortex/"
RELEASES = "https://github.com/Joto3D/Cortex/releases/latest"
GEMINI_KEY_URL = "https://aistudio.google.com/apikey"
LINKS = {"website": WEBSITE, "updates": RELEASES, "gemini_key": GEMINI_KEY_URL,
         "claude_key": "https://console.anthropic.com/settings/keys"}


def _ok(**data) -> dict:
    return {"ok": True, **data}


def _err(e: Exception | str) -> dict:
    return {"ok": False, "error": str(e)}


def _safe(fn: Callable) -> Callable:
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as e:  # shown in the window, never crashes the app
            log.exception("bridge %s failed", fn.__name__)
            return _err(e)

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _list_windows() -> list[str]:
    from cortex.games import list_windows

    return list_windows()


def _app_icon(app_name: str) -> str | None:
    """The app's Dock icon as a PNG data URL (macOS only)."""
    try:
        import AppKit

        for app in AppKit.NSWorkspace.sharedWorkspace().runningApplications():
            if app.localizedName() == app_name and app.icon() is not None:
                img = app.icon()
                img.setSize_((64, 64))
                rep = AppKit.NSBitmapImageRep.imageRepWithData_(img.TIFFRepresentation())
                png = rep.representationUsingType_properties_(AppKit.NSBitmapImageFileTypePNG, {})
                return "data:image/png;base64," + base64.b64encode(bytes(png)).decode()
    except Exception as e:
        log.debug("no icon for %s: %s", app_name, e)
    return None


def _open(target: str) -> None:
    subprocess.run(["open", target], check=False)


class Bridge:
    def __init__(
        self,
        controller: AppController,
        list_windows: Callable[[], list[str]] = _list_windows,
        app_icon: Callable[[str], str | None] = _app_icon,
        open_url: Callable[[str], None] = webbrowser.open,
        open_path: Callable[[str], None] = _open,
        test_gemini: Callable[[str, str], str] | None = None,
    ):
        # Underscore names are not exposed to JavaScript by pywebview.
        self._ctl = controller
        self._list_windows = list_windows
        self._app_icon = app_icon
        self._open_url = open_url
        self._open_path = open_path
        self._test_gemini = test_gemini or _test_gemini
        self._icons: dict[str, str | None] = {}
        self._last_frame_id = -1
        self._on_quit: Callable[[], None] = lambda: None
        self._lock = threading.Lock()

    # -- state ------------------------------------------------------------------
    @_safe
    def state(self) -> dict:
        """Everything the window shows; polled a few times per second."""
        c = self._ctl
        s = c.settings
        brain = None
        if s.game:
            try:
                brain = c.brain_for(load_profile(s.game))
            except Exception:
                brain = None
        return _ok(
            version=cortex.__version__,
            mode=c.mode,
            status=c.status,
            last_result=c.last_result or "",
            elapsed=c.elapsed,
            game=s.game,
            game_title=c.game_title(),
            assignment=s.assignment,
            recent=s.recent,
            skills=c.skills(),
            ideas=c.example_assignments(),
            brain=brain,
            has_gemini=bool(system.get_key("gemini")),
            has_claude=bool(system.get_key("anthropic")),
            gemini_model=s.gemini_model,
            gemini_rpm=s.gemini_rpm,
            start_delay_s=s.start_delay_s,
            setup_done=s.setup_done,
            live=LIVE.snapshot(),
        )

    @_safe
    def frame(self) -> dict:
        """The latest frame Cortex saw, as a small JPEG data URL (None if unchanged or nothing yet)."""
        f = LIVE.latest_frame()
        if f is None or LIVE.frame_id == self._last_frame_id:
            return _ok(src=None)
        self._last_frame_id = LIVE.frame_id
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(f).save(buf, format="JPEG", quality=70)
        return _ok(src="data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())

    @_safe
    def windows(self) -> dict:
        """Open app windows to pick the game from (the current game first if it's open)."""
        names = [n for n in self._list_windows() if n not in ("Cortex", "Finder", "Terminal", "System Settings")]
        out = []
        for n in names:
            if n not in self._icons:
                self._icons[n] = self._app_icon(n)
            out.append({"name": n, "icon": self._icons[n]})
        return _ok(windows=out)

    @_safe
    def permissions(self) -> dict:
        return _ok(screen=system.has_screen_recording(), accessibility=system.has_accessibility())

    # -- actions ------------------------------------------------------------------
    @_safe
    def pick_window(self, app_name: str) -> dict:
        return _ok(game=self._ctl.select_window(app_name))

    @_safe
    def set_assignment(self, text: str) -> dict:
        self._ctl.set_assignment(text)
        return _ok()

    @_safe
    def start(self, assignment: str | None = None) -> dict:
        perms = self.permissions()
        if perms.get("screen") is False or perms.get("accessibility") is False:
            return _err("Cortex needs Screen Recording and Accessibility permission. Open Settings ▸ Permissions.")
        if assignment is not None:
            self._ctl.set_assignment(assignment)
        self._ctl.start()
        return _ok()

    @_safe
    def stop(self) -> dict:
        threading.Thread(target=self._ctl.stop, daemon=True).start()
        if self._ctl.recording:
            threading.Thread(target=self._ctl.stop_recording, daemon=True).start()
        return _ok()

    @_safe
    def pause(self) -> dict:
        return _ok(paused=self._ctl.toggle_pause())

    @_safe
    def record(self, name: str) -> dict:
        perms = self.permissions()
        if perms.get("screen") is False or perms.get("accessibility") is False:
            return _err("Cortex needs Screen Recording and Accessibility permission. Open Settings ▸ Permissions.")
        self._ctl.start_recording(name)
        return _ok()

    @_safe
    def stop_recording(self) -> dict:
        self._ctl.status = "Learning…"
        threading.Thread(target=self._ctl.stop_recording, daemon=True).start()
        return _ok()

    @_safe
    def delete_skill(self, name: str) -> dict:
        self._ctl.delete_skill(name)
        return _ok()

    # -- settings -----------------------------------------------------------------
    @_safe
    def save_key(self, provider: str, key: str) -> dict:
        if provider not in system.PROVIDERS:
            return _err(f"unknown provider {provider!r}")
        system.save_key(provider, key)
        return _ok()

    @_safe
    def test_key(self) -> dict:
        key = system.get_key("gemini")
        if not key:
            return _err("No Gemini key saved yet.")
        return _ok(message=self._test_gemini(key, self._ctl.settings.model_id))

    @_safe
    def set_option(self, name: str, value) -> dict:
        s = self._ctl.settings
        if name == "gemini_model" and value in ("fast", "smart"):
            s.gemini_model = value
        elif name == "gemini_rpm":
            s.gemini_rpm = max(1.0, min(60.0, float(value)))
        elif name == "start_delay_s":
            s.start_delay_s = max(0.0, min(10.0, float(value)))
        else:
            return _err(f"unknown option {name!r}")
        s.save()
        return _ok()

    @_safe
    def finish_setup(self) -> dict:
        self._ctl.mark_setup_done()
        return _ok()

    @_safe
    def open_permission(self, pane: str) -> dict:
        if pane == "screen":
            system.request_screen_recording()
        elif pane == "accessibility":
            system.has_accessibility(prompt=True)
        system.open_settings(pane)
        return _ok()

    @_safe
    def open_link(self, name: str) -> dict:
        self._open_url(LINKS[name])
        return _ok()

    @_safe
    def open_folder(self, which: str) -> dict:
        home = cortex_home()
        path = {"log": home / "cortex.log", "games": home / "games", "home": home}[which]
        if which == "log":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
            subprocess.run(["open", "-a", "Console", str(path)], check=False)
        else:
            path.mkdir(parents=True, exist_ok=True)
            self._open_path(str(path))
        return _ok()

    @_safe
    def quit(self) -> dict:
        self._ctl.stop(wait=2)
        self._on_quit()
        return _ok()


def _test_gemini(key: str, model: str) -> str:
    """One tiny request to check the key works; returns a friendly message with the round-trip time."""
    import time

    from cortex.agent.gemini import GeminiClient

    client = GeminiClient(key, model, timeout=15)
    t0 = time.monotonic()
    client.generate({"contents": [{"role": "user", "parts": [{"text": "Reply with the single word: ready"}]}],
                     "generationConfig": {"maxOutputTokens": 8}})
    return f"Gemini works ({client.model}, {1000 * (time.monotonic() - t0):.0f} ms)."
