"""Cortex menu-bar app (macOS). Launch with ``python -m cortex.app`` or double-click Cortex.app.

The menu: pick a game, type an assignment, Start / Pause / Stop. "Setup…"
walks through permissions and the Claude API key (stored in the Keychain).
"""
from __future__ import annotations

import logging
import queue
import subprocess
import threading
import webbrowser

import rumps

from cortex.config import user_profile_dir

from . import system
from .controller import AppController

log = logging.getLogger(__name__)
WEBSITE = "https://joto3d.github.io/Cortex/"
ADD_GAME = "Add a game…"


class CortexApp(rumps.App):
    def __init__(self):
        super().__init__("Cortex", title="🌱", quit_button=None)
        self._notes: queue.Queue[tuple[str, str]] = queue.Queue()
        self.ctl = AppController(notify=lambda t, m: self._notes.put((t, m)))
        system.load_api_key_into_env()

        self.status_item = rumps.MenuItem("Ready")
        self.game_menu = rumps.MenuItem("Game")
        self.assignment_item = rumps.MenuItem("Assignment…", callback=self.on_assignment)
        self.start_item = rumps.MenuItem("Start", callback=self.on_start)
        self.pause_item = rumps.MenuItem("Pause", callback=self.on_pause)
        self.stop_item = rumps.MenuItem("Stop", callback=self.on_stop)
        self.menu = [
            self.status_item,
            None,
            self.game_menu,
            self.assignment_item,
            None,
            self.start_item,
            self.pause_item,
            self.stop_item,
            None,
            rumps.MenuItem("Setup…", callback=self.on_setup),
            rumps.MenuItem("Open games folder", callback=self.on_open_games),
            rumps.MenuItem("Website", callback=lambda _: webbrowser.open(WEBSITE)),
            rumps.MenuItem("Quit Cortex", callback=self.on_quit),
        ]
        self._rebuild_games()
        self._refresh()
        self._timer = rumps.Timer(self._tick, 0.5)
        self._timer.start()

    # -- menu state -------------------------------------------------------------
    def _rebuild_games(self) -> None:
        self.game_menu.clear()
        for name in self.ctl.games():
            item = rumps.MenuItem(name, callback=self.on_pick_game)
            item.state = int(name == self.ctl.settings.game)
            self.game_menu.add(item)
        self.game_menu.add(None)
        self.game_menu.add(rumps.MenuItem(ADD_GAME, callback=self.on_add_game))
        self.game_menu.title = f"Game: {self.ctl.settings.game}"

    def _refresh(self) -> None:
        running = self.ctl.running
        self.status_item.title = self.ctl.status
        a = self.ctl.settings.assignment
        self.assignment_item.title = f"Assignment: {a[:40] + ('…' if len(a) > 40 else '')}" if a else "Assignment…"
        self.start_item.set_callback(None if running else self.on_start)
        self.pause_item.set_callback(self.on_pause if running else None)
        self.pause_item.title = "Resume" if self.ctl.paused.is_set() else "Pause"
        self.stop_item.set_callback(self.on_stop if running else None)
        self.title = ("⏸" if self.ctl.paused.is_set() else "▶︎") if running else "🌱"

    def _tick(self, _) -> None:
        self._refresh()
        while not self._notes.empty():
            title, msg = self._notes.get_nowait()
            rumps.notification(title, "", msg)

    # -- callbacks --------------------------------------------------------------
    def on_pick_game(self, item) -> None:
        try:
            self.ctl.select_game(item.title)
        except RuntimeError as e:
            rumps.alert("Cortex", str(e))
        self._rebuild_games()

    def on_assignment(self, _) -> None:
        ideas = self.ctl.example_assignments()
        msg = "Tell Cortex what to do in plain English."
        if ideas:
            msg += "\n\nIdeas:\n• " + "\n• ".join(ideas)
        w = rumps.Window(msg, "Assignment", default_text=self.ctl.settings.assignment, ok="Save", cancel="Cancel", dimensions=(360, 80))
        r = w.run()
        if r.clicked:
            self.ctl.set_assignment(r.text)
            self._refresh()

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
                self._notes.put(("Game added", f"{name} is ready. Set an assignment and press Start."))
            except Exception as e:
                self.ctl.status = f"Couldn't add {name}: {e}"
                self._notes.put(("Couldn't add game", str(e)))
            rumps.Timer(lambda t: (t.stop(), self._rebuild_games()), 0.1).start()

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
        rumps.alert("Setup done", "Pick a game, set an assignment, then press Start.\nF12 stops Cortex at any time; F11 pauses.")


def selftest() -> None:
    """Import everything the app needs at runtime (used by CI on the built Cortex.app)."""
    import importlib

    for mod in ("anthropic", "keyring", "mss", "numpy", "open_clip", "PIL", "pynput.keyboard", "Quartz",
                "torch", "yaml", "cortex.agent", "cortex.games", "cortex.loop", "cortex.perception.clip_model"):
        importlib.import_module(mod)
    from cortex.config import list_profiles

    assert "stardew" in list_profiles() and "generic_3d" in list_profiles()
    print("Cortex selftest OK")


def main() -> None:
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if "--selftest" in sys.argv:
        selftest()
        return
    CortexApp().run()


if __name__ == "__main__":
    main()
