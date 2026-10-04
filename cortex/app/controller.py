"""Everything the menu-bar app does, without any UI code (so it can be unit-tested).

The bot runs on a background thread. The UI polls ``status`` and calls
start / toggle_pause / stop.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Callable

from cortex.config import cortex_home, list_profiles, load_profile

log = logging.getLogger(__name__)


@dataclass
class Settings:
    game: str = "stardew"
    assignment: str = ""

    @classmethod
    def load(cls) -> "Settings":
        try:
            data = json.loads((cortex_home() / "settings.json").read_text())
            return cls(game=str(data.get("game", cls.game)), assignment=str(data.get("assignment", "")))
        except (OSError, ValueError):
            return cls()

    def save(self) -> None:
        path = cortex_home() / "settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"game": self.game, "assignment": self.assignment}, indent=2))


def _default_run_grid(profile, assignment, stop, paused, on_status) -> str:
    from cortex.assignment import interpret
    from cortex.loop import run

    mission = interpret(assignment, profile)
    on_status(mission.summary or "playing")
    return run(profile, mission, stop=stop, paused=paused)


def _default_run_agent(profile, assignment, stop, paused, on_status) -> str:
    from cortex.loop import run_agent

    result = run_agent(profile, assignment, stop=stop, paused=paused, on_status=on_status)
    return f"{result.summary} ({result.steps} turns, ~${result.cost_usd:.2f})"


class AppController:
    def __init__(
        self,
        settings: Settings | None = None,
        notify: Callable[[str, str], None] = lambda title, msg: None,
        run_grid=_default_run_grid,
        run_agent=_default_run_agent,
        start_delay_s: float = 3.0,
    ):
        self.settings = settings or Settings.load()
        if self.settings.game not in self.games():
            self.settings.game = "stardew"
        self.notify = notify
        self._run_grid = run_grid
        self._run_agent = run_agent
        self.start_delay_s = start_delay_s
        self.stop_event = threading.Event()
        self.paused = threading.Event()
        self._thread: threading.Thread | None = None
        self.status = "Ready"
        self.last_result: str | None = None

    # -- choices --------------------------------------------------------------
    def games(self) -> list[str]:
        return list_profiles()

    def select_game(self, name: str) -> None:
        if self.running:
            raise RuntimeError("stop the bot before switching games")
        load_profile(name)  # validate
        self.settings.game = name
        self.settings.save()

    def set_assignment(self, text: str) -> None:
        self.settings.assignment = text.strip()
        self.settings.save()

    def example_assignments(self) -> list[str]:
        p = load_profile(self.settings.game)
        if p.engine == "agent":
            return list(p.agent.get("example_assignments") or [])
        return ["harvest everything, then water the crops", "clear the weeds and break 5 rocks", "do all the chores"]

    # -- running ----------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        profile = load_profile(self.settings.game)
        self.stop_event.clear()
        self.paused.clear()
        self.last_result = None
        self._thread = threading.Thread(target=self._work, args=(profile,), daemon=True, name="cortex-bot")
        self._thread.start()

    def _work(self, profile) -> None:
        game = profile.window_owner or profile.name
        try:
            if self.start_delay_s:
                self.status = f"Starting in {self.start_delay_s:.0f}s: switch to {game}"
                if self.stop_event.wait(self.start_delay_s):
                    self.status = "Stopped"
                    return
            self.status = f"Playing {game}"
            runner = self._run_agent if profile.engine == "agent" else self._run_grid
            result = runner(profile, self.settings.assignment, self.stop_event, self.paused, self._on_status)
            self.last_result = str(result)
            self.status = f"Finished: {result}"
            self.notify("Cortex finished", str(result))
        except Exception as e:  # show any failure in the menu instead of dying silently
            log.exception("bot crashed")
            self.last_result = f"Error: {e}"
            self.status = f"Error: {e}"
            self.notify("Cortex stopped with an error", str(e))

    def _on_status(self, text: str) -> None:
        self.status = text if len(text) <= 80 else text[:77] + "..."

    def toggle_pause(self) -> bool:
        if self.paused.is_set():
            self.paused.clear()
        else:
            self.paused.set()
        return self.paused.is_set()

    def stop(self, wait: float = 5.0) -> None:
        self.stop_event.set()
        self.paused.clear()
        if self._thread is not None:
            self._thread.join(timeout=wait)
