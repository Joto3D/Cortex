"""Play any game with Google Gemini (free API key from aistudio.google.com).

Two speeds, so the game never waits on the network:

* The **action lane** (a local thread) executes the current plan back-to-back:
  key holds, mouse looks, clicks, or a taught skill at 10 Hz.
* The **planning lane** sends a fresh screenshot to Gemini *while the current
  plan is still running*, and swaps in the new plan as soon as it arrives.

Each request is stateless (assignment + notes + recent actions + one screenshot),
which keeps requests small and fast and fits the free tier's request limits.
Requests are spaced to stay under ``rpm`` requests per minute; on a rate-limit
error Cortex backs off and the action lane keeps playing.
"""
from __future__ import annotations

import json
import logging
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from cortex.config import Profile
from cortex.telemetry import LIVE

from .agent import AgentResult, encode_screenshot
from .tools import TOOLS, ActionExecutor

log = logging.getLogger(__name__)

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
FAST_MODEL = "gemini-flash-lite-latest"
SMART_MODEL = "gemini-flash-latest"
# Tried in order if a model id is unknown to the API (ids change over time).
FALLBACK_MODELS = (FAST_MODEL, "gemini-2.5-flash-lite", SMART_MODEL, "gemini-2.5-flash")
KEY_URL = "https://aistudio.google.com/apikey"

SYSTEM = """You play a video game on the user's Mac for them by calling tools that press keys and move the mouse.

Game: {game}
{description}
Controls (action name -> key):
{keys}
{tips}
How this works:
- You get a new screenshot about every {interval:.0f} seconds. Each time, plan the next ~{horizon:.0f} seconds of play \
as a sequence of tool calls. They run in order, immediately, while you look at the next screenshot.
- Your new plan replaces whatever from the old plan hasn't started yet, so always plan from what you see now.
- Keep each hold short (0.2-3 s) so you can correct course. Combine keys in one hold to move and act together.
- If no controls are listed, use the usual ones for this kind of game: WASD to move, mouse to look or aim, \
space to jump, left click to attack/use/mine, right click to place/use, E or Tab for inventory, Esc for menus.
- Use `note` for progress and locations worth remembering; notes are shown to you every time.
- Call `finish` when the assignment is done, or if it is impossible.
{skills}
Rules: only play this single-player game. Never type into chat, never buy anything, never change account or \
system settings, never quit the game or delete saves."""

USE_SKILL = {
    "name": "use_skill",
    "description": "Play one of the skills the user taught by showing, for some seconds. It runs locally and reacts "
                   "instantly, so prefer it whenever a taught skill fits what needs doing.",
    "input_schema": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "seconds": {"type": "number", "description": "2 to 30"}},
        "required": ["name", "seconds"],
    },
}


class GeminiError(RuntimeError):
    pass


class RateLimited(GeminiError):
    def __init__(self, retry_after: float, message: str = ""):
        super().__init__(message or f"rate limited; retry in {retry_after:.0f}s")
        self.retry_after = retry_after


def _gemini_schema(schema: dict) -> dict:
    """Claude-style JSON schema -> the OpenAPI subset Gemini function declarations accept."""
    out: dict = {}
    for k, v in schema.items():
        if k in ("additionalProperties", "strict"):
            continue
        if k == "type":
            out["type"] = str(v).upper()
        elif k == "properties":
            out["properties"] = {name: _gemini_schema(p) for name, p in v.items()}
        elif k == "items":
            out["items"] = _gemini_schema(v)
        else:
            out[k] = v
    return out


def function_declarations(with_skills: bool) -> list[dict]:
    tools = TOOLS + ([USE_SKILL] if with_skills else [])
    return [{"name": t["name"], "description": t["description"], "parameters": _gemini_schema(t["input_schema"])} for t in tools]


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # bundled with the app; the system Python on macOS may have no CA certificates

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _urllib_transport(url: str, headers: dict, body: bytes, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _retry_after(payload: dict) -> float:
    for d in (payload.get("error") or {}).get("details") or []:
        m = re.match(r"([\d.]+)s", str(d.get("retryDelay", "")))
        if m:
            return float(m.group(1))
    return 10.0


class GeminiClient:
    """Minimal Gemini REST client (no SDK needed). ``transport`` is injectable for tests."""

    def __init__(self, api_key: str, model: str = FAST_MODEL, transport=_urllib_transport, timeout: float = 30.0):
        if not api_key:
            raise GeminiError("No Gemini API key. Get a free one at " + KEY_URL)
        self.api_key = api_key.strip()
        self.model = model
        self.transport = transport
        self.timeout = timeout
        self._thinking_ok = True

    def _thinking(self) -> dict | None:
        if not self._thinking_ok:
            return None
        # Fastest answers: no (or minimal) thinking. Gemini 2.x takes a budget, 3.x takes a level.
        return {"thinkingBudget": 0} if "2." in self.model else {"thinkingLevel": "minimal"}

    def generate(self, body: dict) -> dict:
        tried: list[str] = []
        while True:
            req = dict(body)
            gen = dict(req.get("generationConfig") or {})
            thinking = self._thinking()
            if thinking:
                gen["thinkingConfig"] = thinking
            req["generationConfig"] = gen
            status, raw = self.transport(
                API_URL.format(model=self.model),
                {"Content-Type": "application/json", "x-goog-api-key": self.api_key},
                json.dumps(req).encode(),
                self.timeout,
            )
            try:
                payload = json.loads(raw or b"{}")
            except ValueError:
                payload = {}
            if status == 200:
                return payload
            message = str((payload.get("error") or {}).get("message", raw[:200]))
            if status == 429:
                raise RateLimited(_retry_after(payload), "Gemini's free tier limit was reached; slowing down.")
            if status == 400 and thinking and "think" in message.lower():
                log.info("model %s doesn't take thinkingConfig %s; retrying without", self.model, thinking)
                self._thinking_ok = False
                continue
            if status == 404 and self.model not in tried:
                tried.append(self.model)
                nxt = next((m for m in FALLBACK_MODELS if m not in tried), None)
                if nxt:
                    log.warning("Gemini model %s not available; trying %s", self.model, nxt)
                    self.model = nxt
                    self._thinking_ok = True
                    continue
            if status in (400, 401, 403) and ("api key" in message.lower() or "permission" in message.lower()):
                raise GeminiError(f"Gemini rejected the API key: {message}. Check it in Settings (free key: {KEY_URL}).")
            raise GeminiError(f"Gemini error {status}: {message}")


def parse_calls(payload: dict) -> list[tuple[str, dict]]:
    """Function calls from a generateContent response, in order."""
    if (payload.get("promptFeedback") or {}).get("blockReason"):
        raise GeminiError(f"Gemini blocked the request ({payload['promptFeedback']['blockReason']})")
    calls = []
    for cand in payload.get("candidates") or []:
        for part in (cand.get("content") or {}).get("parts") or []:
            fc = part.get("functionCall")
            if fc and fc.get("name"):
                calls.append((str(fc["name"]), dict(fc.get("args") or {})))
        break  # only the first candidate
    return calls


def estimate_seconds(name: str, args: dict) -> float:
    """Roughly how long an action takes to run, for pipelining."""
    try:
        if name in ("hold", "wait", "use_skill"):
            return float(args.get("seconds", 1))
        if name == "tap":
            return 0.1 * max(1, int(args.get("times", 1))) * max(1, len(args.get("keys") or [1]))
        if name == "click":
            return 0.1 + float(args.get("hold_seconds", 0))
        if name == "look":
            return 0.05
    except (TypeError, ValueError):
        pass
    return 0.0


def describe(name: str, args: dict) -> str:
    if name == "hold":
        return f"hold {'+'.join(map(str, args.get('keys') or []))} {float(args.get('seconds', 0)):.1f}s"
    if name == "tap":
        return f"tap {' '.join(map(str, args.get('keys') or []))}" + (f" ×{args.get('times')}" if args.get("times", 1) != 1 else "")
    if name == "look":
        return f"look {float(args.get('right_degrees', 0)):+.0f}° {float(args.get('down_degrees', 0)):+.0f}°"
    if name == "click":
        return f"{args.get('button', 'left')} click ({float(args.get('x', 0)):.2f}, {float(args.get('y', 0)):.2f})"
    if name == "use_skill":
        return f"skill “{args.get('name')}” {float(args.get('seconds', 0)):.0f}s"
    if name == "wait":
        return f"wait {float(args.get('seconds', 0)):.1f}s"
    return name


@dataclass
class _Queued:
    name: str
    args: dict
    seconds: float


class GeminiPilot:
    def __init__(
        self,
        profile: Profile,
        goal: str,
        grab: Callable[[], np.ndarray],
        executor: ActionExecutor,
        client: GeminiClient,
        rpm: float = 12.0,
        skills: list[str] | None = None,
        play_skill: Callable[[str, float, Callable[[], bool]], str] | None = None,
        should_stop: Callable[[], bool] = lambda: False,
        is_paused: Callable[[], bool] = lambda: False,
        on_status: Callable[[str], None] = lambda s: None,
        max_minutes: float = 60.0,
        screenshot_width: int = 768,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.profile = profile
        self.goal = goal.strip() or "Play the game sensibly and make progress."
        self.grab = grab
        self.executor = executor
        self.client = client
        self.interval = 60.0 / max(rpm, 1.0)
        self.skills = list(skills or []) if play_skill else []
        self.play_skill = play_skill
        self.should_stop = should_stop
        self.is_paused = is_paused
        self.on_status = on_status
        self.max_minutes = max_minutes
        self.width = screenshot_width
        self.sleep = sleep
        self.clock = clock

        self._lock = threading.Lock()
        self._queue: deque[_Queued] = deque()
        self._current: tuple[_Queued, float] | None = None  # (action, started at)
        self._done = threading.Event()
        self._finished: tuple[bool, str] | None = None
        self.recent: deque[str] = deque(maxlen=12)
        self.requests = 0
        self.errors = 0
        self.latency = 1.0  # running estimate of Gemini's response time, seconds

    # -- prompt --------------------------------------------------------------
    @property
    def horizon(self) -> float:
        return min(8.0, max(2.5, self.interval + 1.0))

    def system_prompt(self) -> str:
        keys = self.profile.agent.get("keys") or {}
        tips = self.profile.agent.get("tips") or []
        skills = ""
        if self.skills:
            skills = "Taught skills (call use_skill with one of these names): " + ", ".join(f"“{s}”" for s in self.skills) + "\n"
        return SYSTEM.format(
            game=self.profile.window_owner or self.profile.name,
            description=self.profile.description,
            keys="\n".join(f"- {a}: {k}" for a, k in keys.items()) or "- (none listed)",
            tips=("Tips:\n" + "\n".join(f"- {t}" for t in tips) + "\n") if tips else "",
            interval=self.interval,
            horizon=self.horizon,
            skills=skills,
        )

    def build_request(self, frame: np.ndarray) -> dict:
        notes = self.executor.notes[-10:]
        with self._lock:
            queued = [describe(q.name, q.args) for q in self._queue]
        text = [f"Assignment: {self.goal}"]
        if notes:
            text.append("Your notes:\n" + "\n".join(f"- {n}" for n in notes))
        if self.recent:
            text.append("What you did recently (oldest first):\n" + "\n".join(f"- {r}" for r in self.recent))
        if queued:
            text.append("Not started yet (will be replaced by your new plan): " + "; ".join(queued))
        text.append(f"The game right now. Plan the next ~{self.horizon:.0f} seconds:")
        return {
            "systemInstruction": {"parts": [{"text": self.system_prompt()}]},
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": "\n\n".join(text)},
                    {"inlineData": {"mimeType": "image/jpeg", "data": encode_screenshot(frame, self.width, quality=60)}},
                ],
            }],
            "tools": [{"functionDeclarations": function_declarations(bool(self.skills))}],
            "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1024},
        }

    # -- action lane ---------------------------------------------------------
    def remaining_seconds(self) -> float:
        with self._lock:
            left = sum(q.seconds for q in self._queue)
            if self._current is not None:
                q, started = self._current
                left += max(0.0, q.seconds - (self.clock() - started))
        return left

    def _set_plan(self, calls: list[tuple[str, dict]]) -> None:
        with self._lock:
            self._queue.clear()
            for name, args in calls:
                self._queue.append(_Queued(name, args, estimate_seconds(name, args)))

    def _stopping(self) -> bool:
        return self._done.is_set() or self.should_stop()

    def _act(self) -> None:
        busy = idle = 0.0
        while not self._stopping():
            if self.is_paused():
                with self._lock:
                    self._queue.clear()
                self.executor.release_all()
                self.sleep(0.1)
                continue
            with self._lock:
                q = self._queue.popleft() if self._queue else None
                if q is not None:
                    self._current = (q, self.clock())
            if q is None:
                self.sleep(0.02)
                idle += 0.02
                continue
            t0 = self.clock()
            try:
                if q.name == "use_skill" and self.play_skill:
                    seconds = max(1.0, min(30.0, float(q.args.get("seconds", 5))))
                    text = self.play_skill(str(q.args.get("name", "")), seconds, self._stopping)
                else:
                    out = self.executor.execute(q.name, dict(q.args))
                    text = out.text
                    if out.is_error:
                        text = "error: " + text
            except Exception as e:  # an action must never kill the lane
                log.warning("action %s failed: %s", q.name, e)
                text = f"error: {e}"
            busy += self.clock() - t0
            with self._lock:
                self._current = None
            self.recent.append(f"{describe(q.name, q.args)} → {text}")
            LIVE.action(describe(q.name, q.args))
            if busy + idle > 0:
                LIVE.busy_pct = 100.0 * busy / (busy + idle)

    # -- planning lane -------------------------------------------------------
    def run(self) -> AgentResult:
        actor = threading.Thread(target=self._act, daemon=True, name="cortex-actions")
        actor.start()
        start = self.clock()
        next_request = start
        nudges = 0
        try:
            while not self.should_stop():
                now = self.clock()
                if now - start > self.max_minutes * 60:
                    return self._result(False, f"Played for {self.max_minutes:.0f} minutes.", "max_steps")
                if self.is_paused():
                    self.on_status("Paused")
                    self.sleep(0.2)
                    continue
                # Ask for the next plan so it arrives just as the current one runs out (never faster than the limit).
                if now < next_request or self.remaining_seconds() > self.latency + 0.3:
                    self.sleep(0.05)
                    continue
                frame = self.grab()
                LIVE.frame(frame, every_s=0)
                self.on_status("Gemini is looking…")
                t0 = self.clock()
                next_request = t0 + self.interval
                try:
                    payload = self.client.generate(self.build_request(frame))
                    self.requests += 1
                    self.errors = 0
                except RateLimited as e:
                    next_request = self.clock() + max(e.retry_after, self.interval)
                    self.on_status(f"Free-tier limit: waiting {e.retry_after:.0f}s (still playing)")
                    log.info("rate limited for %.0fs", e.retry_after)
                    continue
                except GeminiError as e:
                    if "API key" in str(e):
                        return self._result(False, str(e), "error")
                    self.errors += 1
                    log.warning("Gemini request failed (%d in a row): %s", self.errors, e)
                    if self.errors >= 5:
                        return self._result(False, f"Gemini kept failing: {e}", "error")
                    next_request = self.clock() + min(30.0, 2.0 ** self.errors)
                    continue
                except (OSError, ValueError) as e:  # network down, timeout
                    self.errors += 1
                    log.warning("network error talking to Gemini (%d in a row): %s", self.errors, e)
                    if self.errors >= 5:
                        return self._result(False, f"Couldn't reach Gemini: {e}", "error")
                    next_request = self.clock() + min(30.0, 2.0 ** self.errors)
                    continue
                took = self.clock() - t0
                self.latency = 0.7 * self.latency + 0.3 * took
                LIVE.think_ms = 1000 * took

                calls = parse_calls(payload)
                for name, args in calls:
                    if name == "finish":
                        return self._result(bool(args.get("success", True)), str(args.get("summary", "Done.")), "finished")
                notes = [(n, a) for n, a in calls if n == "note"]
                for _, args in notes:
                    self.executor.execute("note", {"text": str(args.get("text", ""))})
                actions = [(n, a) for n, a in calls if n != "note"]
                if not actions:
                    nudges += 1
                    if nudges >= 4:
                        return self._result(False, "Gemini stopped giving actions.", "error")
                    continue
                nudges = 0
                self._set_plan(actions)
                plan = " → ".join(describe(n, a) for n, a in actions[:5])
                LIVE.thought = (notes[-1][1].get("text", "") + "  ·  " if notes else "") + plan
                self.on_status(plan)
            return self._result(False, "Stopped by you.", "stopped")
        finally:
            self._done.set()
            actor.join(timeout=10)
            self.executor.release_all()

    def _result(self, success: bool, summary: str, reason: str) -> AgentResult:
        self._done.set()
        log.info("gemini pilot %s after %d requests: %s", reason, self.requests, summary)
        return AgentResult(success, summary, self.requests, 0.0, reason)
