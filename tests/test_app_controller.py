import threading
import time

import pytest

from cortex.app.controller import AppController, Settings


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_HOME", str(tmp_path))
    return tmp_path


def wait_until(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def make(**runners):
    notes = []
    runners.setdefault("has_api_key", lambda: True)
    ctl = AppController(Settings(), notify=lambda t, m: notes.append((t, m)), start_delay_s=0, **runners)
    return ctl, notes


def test_settings_persist(home):
    ctl, _ = make()
    ctl.select_game("generic_3d")
    ctl.set_assignment("  build a hut  ")
    s = Settings.load()
    assert (s.game, s.assignment) == ("generic_3d", "build a hut")


def test_dispatches_by_engine_and_reports_result():
    seen = []

    def grid(profile, assignment, stop, paused, on_status):
        seen.append(("grid", profile.name, assignment))
        return "assignment complete"

    def agent(profile, assignment, stop, paused, on_status):
        seen.append(("agent", profile.name, assignment))
        on_status("looking around")
        return "built a hut"

    ctl, notes = make(run_grid=grid, run_agent=agent)
    ctl.set_assignment("water the crops")
    ctl.start()
    assert wait_until(lambda: not ctl.running)
    ctl.select_game("generic_3d")
    ctl.set_assignment("build a hut")
    ctl.start()
    assert wait_until(lambda: not ctl.running)
    assert seen == [("grid", "stardew", "water the crops"), ("agent", "generic_3d", "build a hut")]
    assert ctl.status == "Finished: built a hut"
    assert notes[-1] == ("Cortex finished", "built a hut")


def test_pause_stop_and_no_game_switch_while_running():
    started = threading.Event()

    def agent(profile, assignment, stop, paused, on_status):
        started.set()
        while not stop.is_set():
            time.sleep(0.01)
        return "stopped"

    ctl, _ = make(run_agent=agent)
    ctl.select_game("generic_3d")
    ctl.start()
    assert started.wait(2)
    assert ctl.running
    assert ctl.toggle_pause() is True and ctl.paused.is_set()
    with pytest.raises(RuntimeError):
        ctl.select_game("stardew")
    ctl.stop()
    assert not ctl.running and not ctl.paused.is_set()


def test_crash_is_reported_not_raised():
    def boom(*a):
        raise ValueError("no game window")

    ctl, notes = make(run_grid=boom)
    ctl.start()
    assert wait_until(lambda: not ctl.running)
    assert ctl.status == "Error: no game window"
    assert notes[-1][0] == "Cortex stopped with an error"


def test_start_delay_can_be_cancelled():
    ctl = AppController(Settings(), start_delay_s=5, run_grid=lambda *a: pytest.fail("should not run"))
    ctl.start()
    assert wait_until(lambda: ctl.status.startswith("Starting in"))
    ctl.stop()
    assert ctl.status == "Stopped"


def test_unknown_saved_game_falls_back_to_stardew():
    ctl = AppController(Settings(game="deleted_game"))
    assert ctl.settings.game == "stardew"


def test_recent_assignments_dedupe_and_cap():
    ctl, _ = make()
    for a in ["a", "b", "c", "a", "d", "e", "f", "g"]:
        ctl.set_assignment(a)
    assert ctl.settings.recent == ["g", "f", "e", "d", "a", "c"]
    ctl.set_assignment("")
    assert ctl.settings.recent[0] == "g"
    assert Settings.load().recent == ctl.settings.recent


def test_setup_done_persists_and_elapsed():
    ctl, _ = make()
    assert not Settings.load().setup_done
    ctl.mark_setup_done()
    assert Settings.load().setup_done
    assert ctl.elapsed == ""


# --- teaching & engine choice --------------------------------------------------------

import numpy as np  # noqa: E402

from cortex.teach import InputLog, Recorder, Skill, Tick  # noqa: E402


class FakeRecorder:
    def __init__(self):
        self._stop = threading.Event()
        self.ran = False

    def run(self):
        self.ran = True
        self._stop.wait(5)

    def stop(self):
        self._stop.set()


def test_recording_flow_learns_and_selects_skill():
    rec = FakeRecorder()
    cleaned = []
    learned = []

    def learn(recorder, name, profile):
        learned.append((recorder, name, profile.name))
        return Skill(name, profile.name, np.zeros((20, 3), np.float32), [Tick()] * 20, np.zeros(20, np.int32))

    ctl, notes = make(make_recorder=lambda p: (rec, lambda: cleaned.append(1)), learn=learn)
    ctl.start_recording("chop trees")
    assert wait_until(lambda: rec.ran)
    assert ctl.recording and ctl.busy and ctl.status.startswith("⏺ Recording")
    with pytest.raises(RuntimeError):
        ctl.start_recording("again")
    ctl.stop_recording()
    assert not ctl.recording
    assert learned == [(rec, "chop trees", "stardew")]
    assert cleaned == [1]
    assert ctl.settings.assignment == "chop trees"
    assert notes[-1][0] == "Cortex learned a skill" and "2s" in notes[-1][1]


def test_recording_needs_a_name_and_reports_errors():
    ctl, notes = make(make_recorder=lambda p: (_ for _ in ()).throw(RuntimeError("no game window")))
    with pytest.raises(ValueError):
        ctl.start_recording("  ")
    ctl.start_recording("fish")
    assert wait_until(lambda: not ctl.recording)
    assert ctl.status == "Error: no game window" and notes[-1][0] == "Recording failed"


def test_engine_choice_without_api_key(home):
    from cortex.games import add_game_offline

    played = []
    runner = lambda name: (lambda p, a, s, pa, st: played.append((name, p.name, a)) or "ok")  # noqa: E731
    ctl, _ = make(run_grid=runner("grid"), run_agent=runner("agent"), run_skill=runner("skill"), has_api_key=lambda: False)

    # agent game, no key, no skills: a helpful error, not a crash
    ctl.select_game("generic_3d")
    with pytest.raises(RuntimeError, match="Teach"):
        ctl.runner_for(load("generic_3d"))

    # once it has a skill, it plays the skill instead
    Skill("build hut", "generic_3d", np.zeros((10, 3), np.float32), [Tick()] * 10, np.zeros(10, np.int32)).save()
    assert ctl.skills() == ["build hut"]
    assert ctl.runner_for(load("generic_3d")) is ctl._run_skill

    # with a key, agent games use Claude
    ctl._has_api_key = lambda: True
    assert ctl.runner_for(load("generic_3d")) is ctl._run_agent

    # games added offline always use skills; Stardew stays on its fast grid mode
    add_game_offline("Hollow Knight")
    assert ctl.runner_for(load("hollow_knight")) is ctl._run_skill
    assert ctl.runner_for(load("stardew")) is ctl._run_grid


def load(name):
    from cortex.config import load_profile

    return load_profile(name)
