"""Build and drive the real menu-bar app code against a rumps stand-in (see tests/fakes/rumps)."""
import importlib
import sys
from pathlib import Path

import pytest

FAKES = str(Path(__file__).parent / "fakes")


@pytest.fixture
def app_module(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.syspath_prepend(FAKES)
    for name in ("rumps", "cortex.app.main"):
        sys.modules.pop(name, None)
    mod = importlib.import_module("cortex.app.main")
    yield mod
    for name in ("rumps", "cortex.app.main"):
        sys.modules.pop(name, None)


def titles(item):
    return [None if i is None else i.title for i in item.items]


def test_fake_rumps_matches_real_clear_crash(app_module):
    import rumps

    with pytest.raises(AttributeError):
        rumps.MenuItem("never had children").clear()


def test_app_starts_with_fresh_settings(app_module):
    app = app_module.CortexApp()  # crashed in v0.1.0/v0.1.1: clear() on an empty submenu
    assert app.title == "🌱"
    assert app.status_item.title == "Ready"
    assert "Add a game…" in titles(app.game_menu)
    assert app.game_menu.title.startswith("Game: ")
    assert titles(app.assignment_menu)[0] == "Type a new assignment…"


def test_menus_rebuild_after_changes(app_module):
    app = app_module.CortexApp()
    app.ctl.set_assignment("water the crops")
    app._rebuild_assignments()
    app._rebuild_assignments()  # rebuilding a populated submenu must work too
    assert "water the crops" in titles(app.assignment_menu)
    assert app.assignment_menu.title == "Assignment: water the crops"

    picked = next(i for i in app.game_menu.items if i is not None and getattr(i, "profile_name", "") == "generic_3d")
    app.on_pick_game(picked)
    assert app.ctl.settings.game == "generic_3d"
    assert app.game_menu.title == "Game: Minecraft"


def test_main_logs_startup(app_module, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["Cortex"])
    app_module.main()
    log = (tmp_path / "cortex.log").read_text()
    assert "entering run loop" in log


def test_teach_menu_and_taught_skills(app_module):
    import numpy as np

    from cortex.teach import Skill, Tick

    app = app_module.CortexApp()
    assert titles(app.teach_menu)[0] == "Record new skill…"
    assert "No skills yet for this game" in titles(app.teach_menu)

    Skill("water the crops", "stardew", np.zeros((10, 3), np.float32), [Tick()] * 10, np.zeros(10, np.int32)).save()
    app._rebuild_teach()
    app._rebuild_assignments()
    assert "✓ water the crops" in titles(app.teach_menu)
    assert app.teach_menu.title == "Teach (1 skills)"
    assert "Skills you taught" in titles(app.assignment_menu)


def test_add_game_without_api_key_adds_it_for_teaching(app_module, monkeypatch):
    import rumps

    from cortex.config import load_profile

    monkeypatch.setattr(app_module.system, "load_api_key_into_env", lambda: False)
    app = app_module.CortexApp()
    rumps.window_responses.append((1, "Hollow Knight"))
    rumps.alert_answers.append(0)  # "Record a skill now?" -> Later
    app.on_add_game(None)

    p = load_profile("hollow_knight")
    assert p.engine == "skill" and p.window_owner == "Hollow Knight"
    assert app.ctl.settings.game == "hollow_knight"
    assert "Hollow Knight  ·  taught" in titles(app.game_menu)
