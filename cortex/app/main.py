"""Cortex for macOS: a window to pick your game, say what to do, and watch it play, plus a menu-bar icon.

Launch with ``python -m cortex.app`` or double-click Cortex.app. The window is
HTML (``ui/index.html``) in a native WebKit view via pywebview; it talks to
Python through ``Bridge``. ``AppController`` does the actual work.
"""
from __future__ import annotations

import logging
import logging.handlers
import subprocess
from pathlib import Path

import cortex
from cortex.config import cortex_home

from .bridge import Bridge
from .controller import AppController

log = logging.getLogger(__name__)
UI = Path(__file__).parent / "ui" / "index.html"


def log_path():
    return cortex_home() / "cortex.log"


def setup_logging() -> None:
    """Log to ~/Library/Application Support/Cortex/cortex.log (the app has no console)."""
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=2_000_000, backupCount=2)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.addHandler(logging.StreamHandler())


def notify(title: str, message: str) -> None:
    """A macOS notification (works without extra entitlements)."""
    def q(s: str) -> str:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"')[:200] + '"'

    try:
        subprocess.Popen(["osascript", "-e", f"display notification {q(message)} with title {q(title)}"])
    except OSError:
        pass


def selftest() -> None:
    """Import everything the app needs at runtime (used by CI on the built Cortex.app)."""
    import importlib

    for mod in ("anthropic", "keyring", "mss", "numpy", "open_clip", "PIL", "pynput.keyboard", "pynput.mouse", "Quartz",
                "torch", "yaml", "webview", "certifi", "cortex.agent", "cortex.agent.gemini", "cortex.games",
                "cortex.loop", "cortex.perception.clip_model", "cortex.teach", "cortex.teach.recorder",
                "cortex.app.menubar", "cortex.app.bridge"):
        importlib.import_module(mod)
    assert UI.exists(), f"missing {UI}"
    from cortex.config import list_profiles

    assert "stardew" in list_profiles() and "generic_3d" in list_profiles()
    print("Cortex selftest OK")


def _alert(title: str, text: str) -> None:
    try:
        import AppKit

        a = AppKit.NSAlert.alloc().init()
        a.setMessageText_(title)
        a.setInformativeText_(text)
        a.runModal()
    except Exception:
        pass


def run() -> None:
    import webview

    ctl = AppController(notify=notify)
    bridge = Bridge(ctl)
    state = {"quitting": False}

    window = webview.create_window(
        "Cortex", url=str(UI), js_api=bridge, width=1040, height=720, min_size=(860, 600),
        background_color="#0e1014",
    )

    def show_window() -> None:
        window.show()
        try:
            import AppKit

            AppKit.NSApp.activateIgnoringOtherApps_(True)
        except Exception:
            pass

    def quit_app() -> None:
        state["quitting"] = True
        ctl.stop(wait=2)
        window.destroy()

    def on_closing():
        # Closing the window while Cortex is playing just hides it; the menu-bar icon stays.
        if not state["quitting"] and ctl.busy:
            window.hide()
            return False
        ctl.stop(wait=2)
        return True

    window.events.closing += on_closing
    bridge._on_quit = quit_app

    def on_started() -> None:
        try:
            from PyObjCTools import AppHelper

            from .menubar import MenuBar

            bar = MenuBar(ctl, show_window, quit_app)
            AppHelper.callAfter(bar.install)
        except Exception:
            log.exception("menu-bar icon unavailable")

    log.info("window ready; entering run loop")
    webview.start(on_started)
    log.info("window closed; quitting")


def main() -> None:
    import faulthandler
    import sys

    if "--selftest" in sys.argv:
        logging.basicConfig(level=logging.INFO)
        selftest()
        return
    setup_logging()
    log.info("Cortex %s starting (python %s, %s)", cortex.__version__, sys.version.split()[0], sys.executable)
    # Native crashes (segfaults in AppKit/PyObjC) bypass Python exceptions; write their stack to the log too.
    crash_file = open(log_path().with_name("crash.log"), "a")
    faulthandler.enable(crash_file)
    try:
        run()
    except Exception as e:
        import traceback

        log.critical("Cortex failed to start:\n%s", "".join(traceback.format_exception(e)))
        _alert("Cortex couldn't start", f"{type(e).__name__}: {e}\n\nDetails are in {log_path()}")
        raise


if __name__ == "__main__":
    main()
