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
