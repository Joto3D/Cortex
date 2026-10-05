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
