"""Everything the menu-bar app does, without any UI code (so it can be unit-tested).

The bot runs on a background thread. The UI polls ``status`` and calls
start / toggle_pause / stop.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from cortex.config import cortex_home, list_profiles, load_profile

log = logging.getLogger(__name__)


MAX_RECENT = 6


@dataclass
class Settings:
    game: str = "stardew"
    assignment: str = ""
    recent: list[str] = field(default_factory=list)
    setup_done: bool = False

    @classmethod
    def load(cls) -> "Settings":
        try:
            data = json.loads((cortex_home() / "settings.json").read_text())
            return cls(
                game=str(data.get("game", cls.game)),
                assignment=str(data.get("assignment", "")),
                recent=[str(r) for r in data.get("recent", [])][:MAX_RECENT],
                setup_done=bool(data.get("setup_done", False)),
            )
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self) -> None:
        path = cortex_home() / "settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"game": self.game, "assignment": self.assignment, "recent": self.recent, "setup_done": self.setup_done}
        path.write_text(json.dumps(data, indent=2))


def _default_run_grid(profile, assignment, stop, paused, on_status) -> str:
    from cortex.assignment import interpret
    from cortex.loop import run

    mission = interpret(assignment, profile)
    on_status(mission.summary or "playing")
    return run(profile, mission, stop=stop, paused=paused)


def _default_has_api_key() -> bool:
    from . import system

    return system.load_api_key_into_env()


def _default_run_skill(profile, assignment, stop, paused, on_status) -> str:
    from cortex.loop import run_skill

    return run_skill(profile, assignment, stop=stop, paused=paused, on_status=on_status).summary


def _default_make_recorder(profile):
    """Real recorder for the game window plus macOS input listeners. Returns (recorder, cleanup)."""
    from cortex.capture.screen import WindowCapture
    from cortex.control.input_mac import frontmost_app_name
    from cortex.teach.recorder import InputLog, Recorder, start_listeners

    cap = WindowCapture(profile.window_owner)

    def to_fraction(x: float, y: float) -> tuple[float, float]:
        w = cap.window
        return (x - w.x) / max(w.width, 1), (y - w.y) / max(w.height, 1)

    log_ = InputLog(ignore={profile.controls.kill_switch, profile.controls.pause})
    recorder = Recorder(cap.grab, lambda: frontmost_app_name() == profile.window_owner, log_)
    # The kill-switch key (F12) also ends a recording, so you never have to leave the game.
    stop_listeners = start_listeners(log_, to_fraction, profile.controls.kill_switch, recorder.stop)
    return recorder, stop_listeners


def _default_learn(recorder, name: str, profile) -> "object":
    from cortex.loop import build_encoder
    from cortex.teach import list_skills
    from cortex.teach.skill import slug

    existing = next((s for s in list_skills(profile.name) if slug(s.name) == slug(name)), None)
    skill = recorder.to_skill(name, profile.name, build_encoder(profile), existing)
    skill.save()
    return skill


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
        run_skill=_default_run_skill,
        has_api_key: Callable[[], bool] = _default_has_api_key,
        make_recorder=_default_make_recorder,
        learn=_default_learn,
        start_delay_s: float = 3.0,
    ):
        self.settings = settings or Settings.load()
        if self.settings.game not in self.games():
            self.settings.game = "stardew"
        self.notify = notify
        self._run_grid = run_grid
        self._run_agent = run_agent
        self._run_skill = run_skill
        self._has_api_key = has_api_key
        self._make_recorder = make_recorder
        self._learn = learn
        self._recorder = None
        self._rec_thread: threading.Thread | None = None
        self._rec_stop = threading.Event()
        self.recording_name: str | None = None
        self.start_delay_s = start_delay_s
        self.stop_event = threading.Event()
        self.paused = threading.Event()
        self._thread: threading.Thread | None = None
        self.status = "Ready"
        self.last_result: str | None = None
        self.started_at: float | None = None

    # -- choices --------------------------------------------------------------
    def games(self) -> list[str]:
        return list_profiles()

    def select_game(self, name: str) -> None:
        if self.busy:
            raise RuntimeError("stop the bot before switching games")
        load_profile(name)  # validate
        self.settings.game = name
        self.settings.save()

    def set_assignment(self, text: str) -> None:
        text = text.strip()
        self.settings.assignment = text
        if text:
            self.settings.recent = [text] + [r for r in self.settings.recent if r != text][: MAX_RECENT - 1]
        self.settings.save()

    def mark_setup_done(self) -> None:
        self.settings.setup_done = True
        self.settings.save()

    @property
    def elapsed(self) -> str:
        """Running time as m:ss, or "" when idle."""
        if not self.busy or self.started_at is None:
            return ""
        s = int(time.monotonic() - self.started_at)
        return f"{s // 60}:{s % 60:02d}"

    def skills(self) -> list[str]:
        from cortex.teach import list_skills

        return [s.name for s in list_skills(load_profile(self.settings.game).name)]

    def example_assignments(self) -> list[str]:
        p = load_profile(self.settings.game)
        if p.engine == "skill":
            return []
        if p.engine == "agent":
            return list(p.agent.get("example_assignments") or [])
        return ["harvest everything, then water the crops", "clear the weeds and break 5 rocks", "do all the chores"]

    # -- running ----------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def recording(self) -> bool:
        return self._rec_thread is not None and self._rec_thread.is_alive()

    @property
    def busy(self) -> bool:
        return self.running or self.recording

    def runner_for(self, profile):
        """Which way to play: Claude only when a key is set; otherwise learned skills."""
        if profile.engine == "skill":
            return self._run_skill
        if profile.engine == "agent":
            if self._has_api_key():
                return self._run_agent
            if self.skills():
                return self._run_skill
            raise RuntimeError(
                "This game needs a Claude API key for thinking mode. No key? Teach it instead: "
                "Teach ▸ Record new skill…, then play for a few minutes."
            )
        return self._run_grid

    def start(self) -> None:
        if self.busy:
            return
        profile = load_profile(self.settings.game)
        self.stop_event.clear()
        self.paused.clear()
        self.last_result = None
        self.started_at = time.monotonic()
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
            runner = self.runner_for(profile)
            self.status = f"Playing {game}"
            result = runner(profile, self.settings.assignment, self.stop_event, self.paused, self._on_status)
            self.last_result = str(result)
            self.status = f"Finished: {result}"
            self.notify("Cortex finished", str(result))
        except Exception as e:  # show any failure in the menu instead of dying silently
            log.exception("bot crashed")
            self.last_result = f"Error: {e}"
            self.status = f"Error: {e}"
            self.notify("Cortex stopped with an error", str(e))

    # -- teaching -----------------------------------------------------------------
    def start_recording(self, name: str) -> None:
        name = name.strip()
        if not name:
            raise ValueError("give the skill a name, like “chop trees”")
        if self.busy:
            raise RuntimeError("stop the bot before recording")
        profile = load_profile(self.settings.game)
        self.recording_name = name
        self.last_result = None
        self.started_at = time.monotonic()
        self._rec_stop = threading.Event()
        self._rec_thread = threading.Thread(target=self._record_work, args=(profile, name), daemon=True, name="cortex-record")
        self._rec_thread.start()

    def stop_recording(self, wait: float = 60.0) -> None:
        self._rec_stop.set()
        if self._recorder is not None:
            self._recorder.stop()
        if self._rec_thread is not None:
            self._rec_thread.join(timeout=wait)

    def _record_work(self, profile, name: str) -> None:
        game = profile.window_owner or profile.name
        cleanup = lambda: None  # noqa: E731
        try:
            if self.start_delay_s:
                self.status = f"Recording starts in {self.start_delay_s:.0f}s: switch to {game}"
                if self._rec_stop.wait(self.start_delay_s):
                    self.status = "Recording cancelled"
                    return
            self._recorder, cleanup = self._make_recorder(profile)
            self.status = f"⏺ Recording “{name}”: play, then press F12"
            self._recorder.run()
            cleanup()
            cleanup = lambda: None  # noqa: E731
            self.status = f"Learning “{name}”…"
            skill = self._learn(self._recorder, name, profile)
            self.set_assignment(name)
            msg = f"Learned “{name}” ({skill.seconds:.0f}s of examples). Press Start to watch it play."
            self.last_result = msg
            self.status = f"Learned “{name}”"
            self.notify("Cortex learned a skill", msg)
        except Exception as e:
            log.exception("recording failed")
            self.last_result = f"Error: {e}"
            self.status = f"Error: {e}"
            self.notify("Recording failed", str(e))
        finally:
            cleanup()
            self._recorder = None
            self.recording_name = None

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
