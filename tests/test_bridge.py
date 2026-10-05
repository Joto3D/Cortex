import pytest

from cortex.app.bridge import Bridge
from cortex.app.controller import AppController, Settings
from cortex.telemetry import LIVE


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_HOME", str(tmp_path))
    for env in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    import cortex.app.system as system

    store = {}
    monkeypatch.setattr(system, "get_key", lambda p: store.get(p))
    monkeypatch.setattr(system, "save_key", lambda p, k: store.__setitem__(p, k.strip()) if k.strip() else store.pop(p, None))
    monkeypatch.setattr(system, "has_screen_recording", lambda: True)
    monkeypatch.setattr(system, "has_accessibility", lambda prompt=False: True)
    return tmp_path


def make(**kw):
    opened = []
    ctl = AppController(Settings(), start_delay_s=0, has_key=lambda p: False, run_skill=lambda *a: "ok")
    b = Bridge(ctl, list_windows=lambda: ["Finder", "Celeste", "Cortex"], app_icon=lambda n: None,
               open_url=opened.append, open_path=opened.append, **kw)
    return b, ctl, opened


def test_state_before_a_game_is_picked():
    b, _, _ = make()
    s = b.state()
    assert s["ok"] and s["game"] == "" and s["brain"] is None and s["mode"] == "idle"
    assert s["skills"] == [] and not s["has_gemini"] and not s["setup_done"]
    assert b.windows()["windows"] == [{"name": "Celeste", "icon": None}]
    assert b.start("go")["ok"] is False  # no game: an error message, not an exception


def test_pick_window_save_key_and_options():
    b, ctl, opened = make(test_gemini=lambda key, model: f"ok {key} {model}")
    assert b.pick_window("Celeste")["game"] == "celeste"
    assert b.state()["game_title"] == "Celeste"
    assert b.test_key()["ok"] is False
    assert b.save_key("gemini", " AIza123 ")["ok"]
    assert b.state()["has_gemini"]
    assert b.test_key()["message"] == "ok AIza123 gemini-flash-lite-latest"
    assert b.set_option("gemini_model", "smart")["ok"] and ctl.settings.model_id == "gemini-flash-latest"
    assert b.set_option("gemini_rpm", 500)["ok"] and ctl.settings.gemini_rpm == 60
    assert b.set_option("nope", 1)["ok"] is False
    assert b.save_key("evil", "x")["ok"] is False
    b.open_link("gemini_key")
    assert opened == ["https://aistudio.google.com/apikey"]
    b.finish_setup()
    assert Settings.load().setup_done


def test_frame_is_sent_once_per_new_frame():
    import numpy as np

    b, _, _ = make()
    LIVE.reset()
    assert b.frame()["src"] is None
    LIVE.frame(np.zeros((100, 200, 3), np.uint8), every_s=0)
    src = b.frame()["src"]
    assert src.startswith("data:image/jpeg;base64,")
    assert b.frame()["src"] is None


def test_errors_come_back_as_messages():
    b, ctl, _ = make()
    b.pick_window("Celeste")
    r = b.start("do it")
    assert r["ok"]  # starts; fails in the background because there's no brain yet
    import time

    for _ in range(100):
        if not ctl.running:
            break
        time.sleep(0.01)
    assert b.state()["mode"] == "error" and "Gemini key" in b.state()["status"]
    assert b.delete_skill("nope")["ok"]
