import json
from types import SimpleNamespace

import pytest

from cortex import assignment as asg
from cortex.assignment import (
    AssignmentError,
    ClaudeInterpreter,
    KeywordInterpreter,
    Mission,
    Step,
    interpret,
    mission_from_json,
)
from cortex.config import load_profile


@pytest.fixture(scope="module")
def stardew():
    return load_profile("stardew")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("water the crops", [Step("water")]),
        ("harvest everything, then water the crops and break 5 rocks",
         [Step("harvest"), Step("water"), Step("clear_stone", 5)]),
        ("Please water my plants first, then pick the ripe ones", [Step("water"), Step("harvest")]),
        ("chop twelve sticks then mow the weeds", [Step("clear_twigs", 12), Step("clear_weeds")]),
        ("Watering, weeding.", [Step("water"), Step("clear_weeds")]),
    ],
)
def test_keyword_interpreter(stardew, text, expected):
    m = KeywordInterpreter(stardew).interpret(text)
    assert list(m.steps) == expected
    assert m.source == "keywords"


def test_keyword_interpreter_everything_and_unsupported(stardew):
    m = KeywordInterpreter(stardew).interpret("do all the chores and then feed the chickens")
    assert [s.task for s in m.steps] == [t.name for t in stardew.tasks]
    assert m.unsupported == ("feed the chickens",)


def test_pickaxe_is_not_harvesting(stardew):
    m = KeywordInterpreter(stardew).interpret("use the pickaxe")
    assert [s.task for s in m.steps] == ["clear_stone"]


def test_empty_assignment_means_everything(stardew):
    m = interpret("   ", stardew)
    assert m.source == "default" and len(m.steps) == len(stardew.tasks)


def test_mission_from_json_validates(stardew):
    m = mission_from_json("x", {"steps": [{"task": "water", "limit": 0}], "summary": "s"}, stardew, "claude")
    assert m.steps == (Step("water", None),)
    with pytest.raises(AssignmentError):
        mission_from_json("x", {"steps": [{"task": "fly", "limit": None}]}, stardew, "claude")


class FakeMessages:
    def __init__(self, payload, stop_reason="end_turn"):
        self.payload = payload
        self.stop_reason = stop_reason
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        content = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=json.dumps(self.payload))]
        return SimpleNamespace(stop_reason=self.stop_reason, content=content)


def fake_client(payload, stop_reason="end_turn"):
    msgs = FakeMessages(payload, stop_reason)
    return SimpleNamespace(beta=SimpleNamespace(messages=msgs)), msgs


def test_claude_interpreter_request_and_parse(stardew):
    payload = {
        "steps": [{"task": "clear_stone", "limit": 3}, {"task": "water", "limit": None}],
        "unsupported": ["pet the dog"],
        "summary": "Break 3 rocks, then water.",
    }
    client, msgs = fake_client(payload)
    m = ClaudeInterpreter(stardew, client=client).interpret("smash three rocks, water, pet the dog")
    assert m.steps == (Step("clear_stone", 3), Step("water"))
    assert m.unsupported == ("pet the dog",) and m.source == "claude"

    kw = msgs.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["messages"] == [{"role": "user", "content": "smash three rocks, water, pet the dog"}]
    fmt = kw["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    task_enum = fmt["schema"]["properties"]["steps"]["items"]["properties"]["task"]["enum"]
    assert task_enum == [t.name for t in stardew.tasks]
    assert "Water planted crops" in kw["system"]
    assert kw["fallbacks"] == "default"


def test_claude_refusal_raises(stardew):
    client, _ = fake_client({}, stop_reason="refusal")
    with pytest.raises(AssignmentError):
        ClaudeInterpreter(stardew, client=client).interpret("water")


def test_interpret_falls_back_to_keywords_without_credentials(stardew, monkeypatch):
    def no_creds(*a, **k):
        raise asg._ClaudeUnavailable("no Claude credentials")

    monkeypatch.setattr(asg, "_ask_claude", no_creds)
    m = interpret("water then harvest", stardew)
    assert m.source == "keywords" and [s.task for s in m.steps] == ["water", "harvest"]
    with pytest.raises(AssignmentError):
        interpret("water", stardew, mode="claude")


def test_mission_describe(stardew):
    text = Mission("a", (Step("water"), Step("harvest", 2)), "keywords", "", ("x",)).describe()
    assert "1. water (until done)" in text and "2. harvest x2" in text and "Can't do: x" in text
