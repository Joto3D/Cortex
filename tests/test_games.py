import json
from types import SimpleNamespace

import pytest

from cortex import config
from cortex.config import list_profiles, load_profile
from cortex.games import _SCHEMA, add_game, build_profile, slugify

DRAFT = {
    "description": "A first-person survival game.",
    "keys": [
        {"action": "Move Forward", "key": "W"},
        {"action": "jump", "key": "space"},
        {"action": "attack", "key": "mouse_left"},
        {"action": "teleport", "key": "hyperdrive"},  # invalid key -> dropped
    ],
    "tips": ["Eat when hungry."],
    "reflexes": [
        {"name": "Low Health", "roi": [0.1, 0.9, 0.3, 0.95], "prompt": "empty health bar", "otherwise": "full health bar", "keys": ["s"]},
        {"name": "bad", "roi": [0.5, 0.5, 0.2, 0.2], "prompt": "x", "otherwise": "y", "keys": ["s"]},  # inverted roi
    ],
    "look_px_per_degree": 8,
    "example_assignments": ["gather wood", "build a shelter"],
}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_HOME", str(tmp_path))
    return tmp_path


def test_slugify():
    assert slugify("Minecraft: Java Edition") == "minecraft_java_edition"
    assert slugify("!!!") == "game"


def test_build_profile_keeps_valid_parts():
    raw = build_profile("Survive!", "SurviveApp", DRAFT)
    assert raw["engine"] == "agent"
    assert raw["agent"]["keys"] == {"move_forward": "w", "jump": "space", "attack": "mouse_left"}
    assert [r["name"] for r in raw["reflexes"]] == ["low_health"]
    assert raw["agent"]["look_px_per_degree"] == 8
    assert raw["game"]["window_owner"] == "SurviveApp"


def test_user_profiles_are_listed_and_loaded(home):
    class FakeClient:
        def __init__(self):
            self.kwargs = None
            self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

        def create(self, **kw):
            self.kwargs = kw
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=json.dumps(DRAFT))])

    client = FakeClient()
    path = add_game("Survive Island", "Survive", screenshot=False, client=client)
    assert path == home / "games" / "survive_island.yaml"
    assert client.kwargs["output_config"]["format"]["schema"] == _SCHEMA
    assert "Survive Island" in client.kwargs["messages"][0]["content"][0]["text"]

    assert list_profiles()[0] == "survive_island"
    assert {"stardew", "generic_3d"} <= set(list_profiles())
    p = load_profile("survive_island")
    assert p.engine == "agent" and p.window_owner == "Survive"
    assert p.reflexes[0].keys == ("s",)


def test_grid_profile_requires_labels():
    with pytest.raises(ValueError, match="labels"):
        config.profile_from_dict({"engine": "grid", "game": {}})
    with pytest.raises(ValueError, match="engine"):
        config.profile_from_dict({"engine": "warp"})


def test_unknown_profile_name_lists_known(home):
    with pytest.raises(FileNotFoundError, match="stardew"):
        load_profile("nope")
