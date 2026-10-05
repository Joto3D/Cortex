"""Cortex menu-bar app (macOS). Launch with ``python -m cortex.app`` or double-click Cortex.app.

The menu: pick a game, type or pick an assignment, Start / Pause / Stop.
"Setup…" walks through permissions and the Claude API key (stored in the
Keychain), and opens on its own the first time Cortex runs.
"""
from __future__ import annotations

import logging
import logging.handlers
import queue
import subprocess
import threading
import webbrowser

import rumps

import cortex
from cortex.config import cortex_home, load_profile, user_profile_dir

from . import system
from .controller import AppController

log = logging.getLogger(__name__)
WEBSITE = "https://joto3d.github.io/Cortex/"
RELEASES = "https://github.com/Joto3D/Cortex/releases/latest"
ADD_GAME = "Add a game…"
NEW_ASSIGNMENT = "Type a new assignment…"
ICONS = {"idle": "🌱", "starting": "⏳", "running": "▶︎", "paused": "⏸", "error": "⚠️"}


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


class CortexApp(rumps.App):
    def __init__(self):
        super().__init__("Cortex", title="🌱", quit_button=None)
        self._notes: queue.Queue[tuple[str, str]] = queue.Queue()
        self.ctl = AppController(notify=lambda t, m: self._notes.put((t, m)))
        system.load_api_key_into_env()

        self.status_item = rumps.MenuItem("Ready")
        self.detail_item = rumps.MenuItem("")
        self.game_menu = rumps.MenuItem("Game")
        self.assignment_menu = rumps.MenuItem("Assignment")
        self.start_item = rumps.MenuItem("Start", callback=self.on_start, key="s")
        self.pause_item = rumps.MenuItem("Pause", callback=self.on_pause, key="p")
        self.stop_item = rumps.MenuItem("Stop", callback=self.on_stop, key=".")
        self.menu = [
            self.status_item,
            self.detail_item,
            None,
            self.game_menu,
            self.assignment_menu,
            None,
            self.start_item,
            self.pause_item,
            self.stop_item,
            None,
            rumps.MenuItem("Setup…", callback=self.on_setup, key=","),
            rumps.MenuItem("Open games folder", callback=self.on_open_games),
            rumps.MenuItem("Show log", callback=self.on_show_log),
            None,
            rumps.MenuItem("Website", callback=lambda _: webbrowser.open(WEBSITE)),
            rumps.MenuItem("Check for updates", callback=lambda _: webbrowser.open(RELEASES)),
            rumps.MenuItem(f"About Cortex {cortex.__version__}", callback=self.on_about),
            rumps.MenuItem("Quit Cortex", callback=self.on_quit, key="q"),
        ]
        self._rebuild_games()
        self._rebuild_assignments()
        self._refresh()
        self._timer = rumps.Timer(self._tick, 0.5)
        self._timer.start()
        if not self.ctl.settings.setup_done:
            # First launch: walk through setup once the menu-bar icon is up.
            rumps.Timer(self._first_run, 1.0).start()

    # -- menu state -------------------------------------------------------------
    def _rebuild_games(self) -> None:
        self.game_menu.clear()
        for name in self.ctl.games():
            try:
                p = load_profile(name)
                label = f"{p.window_owner or name}  ·  {'fast 2D' if p.engine == 'grid' else 'thinking'}"
            except Exception:
                label = name
            item = rumps.MenuItem(label, callback=self.on_pick_game)
            item.profile_name = name
            item.state = int(name == self.ctl.settings.game)
            self.game_menu.add(item)
        self.game_menu.add(None)
        self.game_menu.add(rumps.MenuItem(ADD_GAME, callback=self.on_add_game))
        try:
            current = load_profile(self.ctl.settings.game).window_owner or self.ctl.settings.game
        except Exception:
            current = self.ctl.settings.game
        self.game_menu.title = f"Game: {current}"

    def _rebuild_assignments(self) -> None:
        m = self.assignment_menu
        m.clear()
        m.add(rumps.MenuItem(NEW_ASSIGNMENT, callback=self.on_assignment, key="n"))
        current = self.ctl.settings.assignment
        recent = self.ctl.settings.recent
        if recent:
            m.add(None)
            m.add(rumps.MenuItem("Recent"))
            for text in recent:
                item = rumps.MenuItem(_short(text, 50), callback=self.on_pick_assignment)
                item.assignment = text
                item.state = int(text == current)
                m.add(item)
        ideas = [i for i in self.ctl.example_assignments() if i not in recent]
        if ideas:
            m.add(None)
            m.add(rumps.MenuItem("Ideas"))
            for text in ideas:
                item = rumps.MenuItem(_short(text, 50), callback=self.on_pick_assignment)
                item.assignment = text
                m.add(item)
        m.title = f"Assignment: {_short(current, 32)}" if current else "Assignment: (none yet)"

    def _state(self) -> str:
        if self.ctl.running:
            if self.ctl.paused.is_set():
                return "paused"
            return "starting" if self.ctl.status.startswith("Starting") else "running"
        return "error" if self.ctl.status.startswith("Error") else "idle"

    def _refresh(self) -> None:
        state = self._state()
        running = self.ctl.running
        self.title = ICONS[state]
        elapsed = self.ctl.elapsed
        self.status_item.title = _short(self.ctl.status, 60) + (f"  ({elapsed})" if elapsed else "")
        self.detail_item.title = "F12 stops · F11 pauses" if running else _short(self.ctl.last_result or "", 60)
        self.start_item.set_callback(None if running else self.on_start)
        self.pause_item.set_callback(self.on_pause if running else None)
        self.pause_item.title = "Resume" if self.ctl.paused.is_set() else "Pause"
        self.stop_item.set_callback(self.on_stop if running else None)

    def _tick(self, _) -> None:
        self._refresh()
        while not self._notes.empty():
            title, msg = self._notes.get_nowait()
            rumps.notification(title, "", msg)

    def _first_run(self, timer) -> None:
        timer.stop()
        if rumps.alert(
            "Welcome to Cortex 🌱",
            "Cortex plays single-player games for you. You tell it what to do in plain English.\n\n"
            "First, let's give it permission to see and play your game. It takes about a minute.",
            ok="Set up", cancel="Later",
        ):
            self.on_setup(None)
        self.ctl.mark_setup_done()

    # -- callbacks --------------------------------------------------------------
    def on_pick_game(self, item) -> None:
        try:
            self.ctl.select_game(item.profile_name)
        except RuntimeError as e:
            rumps.alert("Cortex", str(e))
        self._rebuild_games()
        self._rebuild_assignments()

    def on_assignment(self, _) -> None:
        ideas = self.ctl.example_assignments()
        msg = "Tell Cortex what to do in plain English, as you'd tell a friend."
        if ideas:
            msg += "\n\nFor example:\n• " + "\n• ".join(ideas[:3])
        w = rumps.Window(msg, "New assignment", default_text=self.ctl.settings.assignment, ok="Save", cancel="Cancel", dimensions=(380, 80))
        r = w.run()
        if r.clicked:
            self.ctl.set_assignment(r.text)
            self._rebuild_assignments()
            self._refresh()

    def on_pick_assignment(self, item) -> None:
        self.ctl.set_assignment(item.assignment)
        self._rebuild_assignments()

    def on_start(self, _) -> None:
        if not self._permissions_ok():
            return
        if not self.ctl.settings.assignment:
            self.on_assignment(None)
        self.ctl.start()
        self._refresh()

    def on_pause(self, _) -> None:
        self.ctl.toggle_pause()
        self._refresh()

    def on_stop(self, _) -> None:
        threading.Thread(target=self.ctl.stop, daemon=True).start()

    def on_quit(self, _) -> None:
        self.ctl.stop(wait=2)
        rumps.quit_application()

    def on_show_log(self, _) -> None:
        path = log_path()
        path.touch(exist_ok=True)
        subprocess.run(["open", "-a", "Console", str(path)], check=False)

    def on_about(self, _) -> None:
        rumps.alert(
            f"Cortex {cortex.__version__}",
            "Plays single-player games for you.\n\nFree and open source (MIT).\n"
            "Only use it in single-player or offline games.\n\n" + WEBSITE,
        )

    def on_open_games(self, _) -> None:
        d = user_profile_dir()
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run(["open", str(d)], check=False)

    def on_add_game(self, _) -> None:
        from cortex.games import add_game, list_windows

        try:
            windows = list_windows()
        except Exception:
            windows = []
        hint = ("\n\nOpen windows: " + ", ".join(windows[:15])) if windows else ""
        r = rumps.Window(
            "Which game? Type its name exactly as it appears in the menu bar while it's open "
            "(start the game first so Cortex can take a screenshot)." + hint,
            ADD_GAME, ok="Add", cancel="Cancel", dimensions=(320, 24),
        ).run()
        name = r.text.strip()
        if not r.clicked or not name:
            return
        if not system.load_api_key_into_env():
            rumps.alert("Cortex", "Adding a game uses Claude. Add your API key in Setup… first.")
            return
        self.ctl.status = f"Asking Claude about {name}…"

        def work():
            try:
                path = add_game(name, window_owner=name if name in windows else None)
                self.ctl.settings.game = path.stem
                self.ctl.settings.save()
                self.ctl.status = f"Added {name}"
                self._notes.put(("Game added", f"{name} is ready. Pick an assignment and press Start."))
            except Exception as e:
                self.ctl.status = f"Couldn't add {name}: {e}"
                self._notes.put(("Couldn't add game", str(e)))
            rumps.Timer(lambda t: (t.stop(), self._rebuild_games(), self._rebuild_assignments()), 0.1).start()

        threading.Thread(target=work, daemon=True).start()

    # -- setup --------------------------------------------------------------------
    def _permissions_ok(self) -> bool:
        if system.has_screen_recording() is False or system.has_accessibility() is False:
            if rumps.alert("Cortex needs permissions", "Cortex needs Screen Recording and Accessibility "
                           "access to see and play the game.", ok="Open Setup", cancel="Cancel"):
                self.on_setup(None)
            return False
        return True

    def on_setup(self, _) -> None:
        steps = [
            ("Screen Recording", "Lets Cortex see the game window.", system.has_screen_recording,
             lambda: (system.request_screen_recording(), system.open_settings("screen"))),
            ("Accessibility", "Lets Cortex press keys and move the mouse.", system.has_accessibility,
             lambda: (system.has_accessibility(prompt=True), system.open_settings("accessibility"))),
            ("Input Monitoring", "Lets the F12 (stop) and F11 (pause) hotkeys work.", lambda: None,
             lambda: system.open_settings("input")),
        ]
        for name, why, check, fix in steps:
            if check() is True:
                continue
            if rumps.alert(f"Setup: {name}", f"{why}\n\nTurn on Cortex in the list that opens, then come back.",
                           ok="Open System Settings", cancel="Skip"):
                fix()
        key = system.get_api_key()
        r = rumps.Window(
            "Paste your Claude API key (from console.anthropic.com). It's stored in your Mac's Keychain.\n\n"
            "It's used to understand assignments, add games, and play 3D games.",
            "Setup: Claude API key", default_text="" if not key else "•" * 12, ok="Save", cancel="Skip",
            dimensions=(320, 24), secure=True,
        ).run()
        if r.clicked and r.text.strip() and not set(r.text.strip()) <= {"•"}:
            system.save_api_key(r.text)
        self.ctl.mark_setup_done()
        rumps.alert("You're all set", "1. Pick a game in Game ▸ (or Add a game…)\n2. Pick an assignment\n3. Press Start (⌘S) and switch to the game\n\nF12 stops Cortex at any time; F11 pauses.")


def _short(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def selftest() -> None:
    """Import everything the app needs at runtime (used by CI on the built Cortex.app)."""
    import importlib

    for mod in ("anthropic", "keyring", "mss", "numpy", "open_clip", "PIL", "pynput.keyboard", "Quartz",
                "torch", "yaml", "cortex.agent", "cortex.games", "cortex.loop", "cortex.perception.clip_model"):
        importlib.import_module(mod)
    from cortex.config import list_profiles

    assert "stardew" in list_profiles() and "generic_3d" in list_profiles()
    print("Cortex selftest OK")


def _report_startup_crash(exc: BaseException) -> None:
    """Make a startup failure visible: log it, and show a dialog if AppKit is usable."""
    import traceback

    text = "".join(traceback.format_exception(exc))
    log.critical("Cortex failed to start:\n%s", text)
    try:
        rumps.alert("Cortex couldn't start", f"{type(exc).__name__}: {exc}\n\nDetails are in {log_path()}")
    except Exception:
        pass


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
        app = CortexApp()
        log.info("menu ready; entering run loop")
        app.run()
    except Exception as e:
        _report_startup_crash(e)
        raise


if __name__ == "__main__":
    main()
