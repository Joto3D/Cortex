from types import SimpleNamespace

import numpy as np
import pytest

from cortex.agent import TOOLS, ActionExecutor, GameAgent
from cortex.agent.reflexes import Reflexes
from cortex.config import ReflexSpec, load_profile

from .conftest import COLORS, FakeEncoder


class RecordingIO:
    def __init__(self):
        self.events = []

    def key(self, name, down):
        self.events.append(("key", name, down))

    def mouse_move(self, x, y):
        self.events.append(("move", x, y))

    def mouse_delta(self, dx, dy):
        self.events.append(("delta", dx, dy))

    def mouse_button(self, x, y, button, down):
        self.events.append(("button", button, down, x, y))


def make_executor(io=None, keymap=None):
    io = io or RecordingIO()
    ex = ActionExecutor(
        io,
        keymap if keymap is not None else {"forward": "w", "jump": "space", "attack": "mouse_left"},
        to_screen=lambda fx, fy: (100 + fx * 800, 50 + fy * 600),
        look_px_per_degree=5,
        sleep=lambda s: None,
    )
    return ex, io


# --- tools -------------------------------------------------------------------

def test_tool_schemas_are_strict():
    names = {t["name"] for t in TOOLS}
    assert names == {"hold", "tap", "look", "click", "wait", "note", "finish"}
    for t in TOOLS:
        s = t["input_schema"]
        assert t["strict"] is True and s["additionalProperties"] is False
        assert set(s["required"]) == set(s["properties"])


def test_hold_resolves_action_names_and_releases():
    ex, io = make_executor()
    out = ex.execute("hold", {"keys": ["forward", "jump"], "seconds": 1.5})
    assert not out.is_error
    assert io.events == [("key", "w", True), ("key", "space", True), ("key", "space", False), ("key", "w", False)]


def test_hold_mouse_button_uses_window_centre():
    ex, io = make_executor()
    ex.execute("hold", {"keys": ["attack"], "seconds": 1})
    assert io.events == [("button", "left", True, 500.0, 350.0), ("button", "left", False, 500.0, 350.0)]


def test_unknown_key_is_an_error_and_nothing_stays_held():
    ex, io = make_executor()
    out = ex.execute("hold", {"keys": ["forward", "teleport"], "seconds": 1})
    assert out.is_error and "teleport" in out.text
    assert not ex._held


def test_look_sends_relative_mouse_in_steps():
    ex, io = make_executor()
    ex.execute("look", {"right_degrees": 30, "down_degrees": -10})
    deltas = [e for e in io.events if e[0] == "delta"]
    assert len(deltas) > 1
    assert sum(d[1] for d in deltas) == pytest.approx(150)
    assert sum(d[2] for d in deltas) == pytest.approx(-50)


def test_click_maps_fractions_to_screen_and_clamps():
    ex, io = make_executor()
    ex.execute("click", {"x": 0.25, "y": 1.7, "button": "right", "hold_seconds": 0})
    assert io.events[0] == ("move", 300.0, 650.0)
    assert io.events[1][:3] == ("button", "right", True)


def test_note_and_finish():
    ex, _ = make_executor()
    assert ex.execute("note", {"text": "iron at spawn"}).text == "noted"
    assert ex.notes == ["iron at spawn"]
    out = ex.execute("finish", {"success": True, "summary": "built a hut"})
    assert out.finished and out.success and out.text == "built a hut"


# --- agent loop ---------------------------------------------------------------

def tool_use(id_, name, **inp):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


def response(*blocks, stop="tool_use", tokens=(1000, 100)):
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop,
        usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1], cache_creation_input_tokens=0, cache_read_input_tokens=0),
    )


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        # Snapshot message structure (the list keeps growing after the call).
        self.calls.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
        return self.responses.pop(0)


@pytest.fixture
def agent_profile():
    return load_profile("generic_3d")


def make_agent(profile, responses, **kw):
    client = ScriptedClient(responses)
    ex, io = make_executor(keymap={k: v for k, v in profile.agent["keys"].items()})
    grabs = []

    def grab():
        grabs.append(1)
        return np.full((60, 80, 3), 120, np.uint8)

    agent = GameAgent(profile, "chop a tree", grab, ex, client=client, sleep=lambda s: None, **kw)
    return agent, client, io, grabs


def test_agent_runs_tools_in_order_and_finishes(agent_profile):
    agent, client, io, grabs = make_agent(agent_profile, [
        response(tool_use("t1", "look", right_degrees=10, down_degrees=0), tool_use("t2", "hold", keys=["forward"], seconds=1)),
        response(tool_use("t3", "finish", success=True, summary="tree chopped")),
    ])
    result = agent.run()
    assert result.success and result.reason == "finished" and result.summary == "tree chopped"
    assert result.steps == 2
    kinds = [e[0] for e in io.events]
    assert kinds.index("delta") < kinds.index("key")  # look ran before walking

    first = client.calls[0]
    assert first["model"] == "claude-opus-5-5"
    assert first["tools"] == TOOLS
    assert "chop a tree" in first["messages"][0]["content"][0]["text"]
    assert first["messages"][0]["content"][1]["type"] == "image"
    assert "forward: w" in first["system"]
    edit = first["context_management"]["edits"][0]
    assert edit["type"] == "clear_tool_uses_20250919" and edit["keep"]["type"] == "tool_uses"
    assert "context-management-2025-06-27" in first["betas"]

    # Second request: tool results for both calls; the screenshot rides on the last one.
    results = client.calls[1]["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert results[-1]["content"][-1]["type"] == "image"
    assert all(c["type"] == "text" for c in results[0]["content"])
    assert len(grabs) == 2


def test_agent_stops_at_budget(agent_profile):
    expensive = [response(tool_use(f"t{i}", "wait", seconds=1), tokens=(200_000, 1000)) for i in range(10)]
    agent, client, _, _ = make_agent(agent_profile, expensive)
    agent.max_cost = 1.0
    result = agent.run()
    assert result.reason == "budget" and result.cost_usd >= 1.0
    assert len(client.calls) == 2  # $0.82 per turn


def test_agent_stops_when_user_stops(agent_profile):
    stop = {"now": False}

    def respond():
        stop["now"] = True
        return response(tool_use("t1", "hold", keys=["forward"], seconds=1))

    agent, client, io, _ = make_agent(agent_profile, [], should_stop=lambda: stop["now"])
    client.responses = [None]
    client.beta.messages.create = lambda **kw: respond()
    result = agent.run()
    assert result.reason == "stopped"
    assert not any(e[0] == "key" for e in io.events)  # the queued hold never ran


def test_agent_refusal_and_max_steps(agent_profile):
    agent, _, _, _ = make_agent(agent_profile, [response(stop="refusal")])
    assert agent.run().reason == "refusal"

    agent, _, _, _ = make_agent(agent_profile, [response(tool_use(f"t{i}", "wait", seconds=1)) for i in range(3)])
    agent.max_steps = 3
    assert agent.run().reason == "max_steps"


def test_agent_nudges_once_when_no_tool_called(agent_profile):
    text = SimpleNamespace(type="text", text="I think I'm done?")
    agent, client, _, _ = make_agent(agent_profile, [
        response(text, stop="end_turn"),
        response(tool_use("t1", "finish", success=False, summary="no trees here")),
    ])
    result = agent.run()
    assert result.reason == "finished" and not result.success
    assert client.calls[1]["messages"][-1]["content"].startswith("Keep playing")


# --- reflexes -------------------------------------------------------------------

def test_reflex_fires_with_cooldown():
    spec = ReflexSpec(name="low_hp", prompt="crop_ripe", otherwise="grass", keys=("s",), roi=(0, 0, 0.5, 1), threshold=0.6, cooldown_s=2)
    pressed = []
    r = Reflexes((spec,), FakeEncoder(), press=pressed.append)
    frame = np.zeros((10, 20, 3), np.uint8)
    frame[:, :10] = COLORS["crop_ripe"]
    frame[:, 10:] = COLORS["grass"]
    assert r.check(frame, now=0.0) == ["low_hp"]
    assert r.check(frame, now=1.0) == []           # cooling down
    assert r.check(frame, now=2.5) == ["low_hp"]
    calm = np.full((10, 20, 3), COLORS["grass"], np.uint8)
    assert r.check(calm, now=10.0) == []
    assert pressed == [("s",), ("s",)]
