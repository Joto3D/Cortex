import json
import threading
import time

import numpy as np
import pytest

from cortex.agent import ActionExecutor
from cortex.agent.gemini import (
    GeminiClient, GeminiError, GeminiPilot, RateLimited, function_declarations, parse_calls,
)
from cortex.config import load_profile


class IO:
    def __init__(self):
        self.events = []

    def key(self, name, down):
        self.events.append(("key", name, down, time.monotonic()))

    def mouse_move(self, x, y):
        pass

    def mouse_delta(self, dx, dy):
        self.events.append(("delta", dx, dy, time.monotonic()))

    def mouse_button(self, x, y, button, down):
        self.events.append(("button", button, down, time.monotonic()))


def reply(*calls):
    parts = [{"functionCall": {"name": n, "args": a}} for n, a in calls]
    return 200, json.dumps({"candidates": [{"content": {"role": "model", "parts": parts}}]}).encode()


class Transport:
    """Fake HTTP: returns queued replies, records requests."""

    def __init__(self, replies, delay=0.0):
        self.replies = list(replies)
        self.requests = []
        self.delay = delay

    def __call__(self, url, headers, body, timeout):
        self.requests.append((url, headers, json.loads(body), time.monotonic()))
        time.sleep(self.delay)
        if not self.replies:
            return reply(("finish", {"success": True, "summary": "all done"}))
        r = self.replies.pop(0)
        return r() if callable(r) else r


def frame():
    return np.zeros((90, 160, 3), np.uint8)


def pilot(transport, rpm=600, **kw):
    profile = load_profile("generic_3d")
    ex = ActionExecutor(kw.pop("io", IO()), {"forward": "w", "jump": "space"}, lambda x, y: (x, y))
    client = GeminiClient("AIza-test", "gemini-2.5-flash-lite", transport=transport)
    return GeminiPilot(profile, "collect wood", frame, ex, client, rpm=rpm, **kw), ex


def test_function_declarations_are_gemini_schema():
    decls = {d["name"]: d for d in function_declarations(with_skills=True)}
    assert {"hold", "tap", "look", "click", "wait", "note", "finish", "use_skill"} <= set(decls)
    p = decls["hold"]["parameters"]
    assert p["type"] == "OBJECT" and p["properties"]["keys"]["type"] == "ARRAY"
    assert p["properties"]["keys"]["items"]["type"] == "STRING"
    assert "additionalProperties" not in json.dumps(decls)
    assert "use_skill" not in {d["name"] for d in function_declarations(with_skills=False)}


def test_request_shape_and_plan_execution():
    t = Transport([reply(("note", {"text": "trees to the north"}), ("hold", {"keys": ["forward", "jump"], "seconds": 0.2}),
                         ("look", {"right_degrees": 10, "down_degrees": 0}))])
    io = IO()
    p, ex = pilot(t, io=io)
    res = p.run()
    assert res.success and res.summary == "all done" and res.reason == "finished"

    url, headers, body, _ = t.requests[0]
    assert url.endswith("/models/gemini-2.5-flash-lite:generateContent")
    assert headers["x-goog-api-key"] == "AIza-test"
    assert body["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
    parts = body["contents"][0]["parts"]
    assert "collect wood" in parts[0]["text"] and parts[1]["inlineData"]["mimeType"] == "image/jpeg"
    assert "Your 3D Game" in body["systemInstruction"]["parts"][0]["text"]
    # the second request carries the notes and what was done
    second = t.requests[1][2]["contents"][0]["parts"][0]["text"]
    assert "trees to the north" in second
    assert ex.notes == ["trees to the north"]
    keys = [(e[1], e[2]) for e in io.events if e[0] == "key"]
    assert ("w", True) in keys and ("space", True) in keys and keys[-1][1] is False


def test_next_plan_is_requested_while_the_current_one_runs():
    # Gemini takes 0.3 s; the first plan holds W for 1.2 s. The second request must start
    # before the hold ends, so the keys never wait for the network.
    t = Transport([reply(("hold", {"keys": ["w"], "seconds": 1.2})), reply(("hold", {"keys": ["d"], "seconds": 0.2}))], delay=0.3)
    io = IO()
    p, _ = pilot(t, io=io)
    p.latency = 0.3
    p.run()
    w_down = next(e[3] for e in io.events if e[:3] == ("key", "w", True))
    w_up = next(e[3] for e in io.events if e[:3] == ("key", "w", False))
    second_request = t.requests[1][3]
    assert w_down < second_request < w_up


def test_rate_limit_backs_off_and_keeps_playing():
    status = []
    t = Transport([
        reply(("hold", {"keys": ["w"], "seconds": 0.1})),
        (429, json.dumps({"error": {"message": "quota", "details": [{"retryDelay": "0.2s"}]}}).encode()),
    ])
    p, _ = pilot(t, on_status=status.append)
    assert p.run().reason == "finished"
    assert any("Free-tier limit" in s for s in status)
    assert len(t.requests) == 3


def test_requests_respect_rpm():
    t = Transport([reply(("tap", {"keys": ["e"], "times": 1}))] * 2)
    p, _ = pilot(t, rpm=120)  # one request per 0.5 s
    p.run()
    gaps = np.diff([r[3] for r in t.requests])
    assert (gaps >= 0.45).all()


def test_bad_key_stops_with_a_clear_message():
    t = Transport([(400, json.dumps({"error": {"message": "API key not valid. Please pass a valid API key."}}).encode())])
    p, _ = pilot(t)
    res = p.run()
    assert res.reason == "error" and "Settings" in res.summary


def test_unknown_model_falls_back_and_thinking_is_dropped_if_unsupported():
    t = Transport([
        (404, b'{"error": {"message": "models/gemini-x is not found"}}'),
        (400, b'{"error": {"message": "Thinking level is not supported for this model."}}'),
        reply(("wait", {"seconds": 0.1})),
    ])
    c = GeminiClient("k", "gemini-x", transport=t)
    c.generate({"contents": []})
    urls = [r[0] for r in t.requests]
    assert "gemini-x" in urls[0] and "gemini-flash-lite-latest" in urls[1]
    assert "thinkingConfig" in t.requests[1][2]["generationConfig"]
    assert "thinkingConfig" not in t.requests[2][2]["generationConfig"]


def test_rate_limited_error_and_parse():
    t = Transport([(429, b'{"error": {"details": [{"retryDelay": "7s"}]}}')])
    with pytest.raises(RateLimited) as e:
        GeminiClient("k", transport=t).generate({})
    assert e.value.retry_after == 7
    with pytest.raises(GeminiError):
        parse_calls({"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(GeminiError, match="No Gemini API key"):
        GeminiClient("")


def test_use_skill_runs_locally_and_stop_works():
    played = []
    stop = threading.Event()

    def play(name, seconds, should_stop):
        played.append((name, seconds))
        stop.set()
        return "played"

    t = Transport([reply(("use_skill", {"name": "collect wood", "seconds": 4}))] * 5)
    p, _ = pilot(t, skills=["collect wood"], play_skill=play, should_stop=stop.is_set)
    assert "use_skill" in json.dumps(p.build_request(frame())["tools"])
    assert "“collect wood”" in p.system_prompt()
    res = p.run()
    assert res.reason == "stopped" and played == [("collect wood", 4.0)]
