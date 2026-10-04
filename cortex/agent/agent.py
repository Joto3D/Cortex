"""A Claude vision agent that plays any game, including 3D ones.

Each turn: send a screenshot -> Claude calls one or more game tools (hold keys,
look, click, ...) -> run them locally -> send the next screenshot. A turn takes
a few seconds, so this plays like a careful human. Instant reactions are left to
local CLIP reflexes (see ``reflexes.py``).

Old screenshots are pruned server-side with context editing, so long sessions
stay cheap. The conversation history is append-only. Claude keeps what matters
in ``note`` calls, and those tool inputs are not cleared.
"""
from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from cortex.config import Profile

from .tools import TOOLS, ActionExecutor

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
# $ per million tokens: (input, output). Used only to enforce the spending limit.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5-1": (10.0, 50.0),
}

SYSTEM = """You are playing a video game on the user's Mac for them, using the tools to press keys and move the mouse.

Game: {game}
{description}

Controls (action name -> key):
{keys}
{tips}
How to play well:
- Look at the screenshot carefully, then act. Each tool call happens immediately in the real game.
- Call several tools in one turn when you are confident (e.g. look, then hold forward 1.5 s, then attack), \
so the game keeps moving. Keep each hold short (0.2-3 s) so you can correct course.
- After your tools run you get a new screenshot. Check that your actions worked before repeating them.
- Use `note` to remember progress and locations. Old screenshots are dropped, but notes are kept.
- Call `finish` when the assignment is done, or if it is impossible or you are stuck after several tries.

Rules:
- Only play this single-player game. Never type into chat, never contact other players, and never buy anything \
or change account, payment, or system settings.
- Don't quit or uninstall the game, and don't delete saves or worlds.
- If a menu, dialog, or loading screen blocks play, deal with it sensibly (close it, or wait)."""


@dataclass
class AgentResult:
    success: bool
    summary: str
    steps: int
    cost_usd: float
    reason: str  # "finished" | "stopped" | "max_steps" | "budget" | "refusal" | "error"


def encode_screenshot(frame: np.ndarray, width: int = 1280, quality: int = 70) -> str:
    """RGB frame -> base64 JPEG, downscaled to ``width`` px wide."""
    from PIL import Image

    img = Image.fromarray(np.ascontiguousarray(frame[..., :3]))
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _image_block(b64: str) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}


class GameAgent:
    def __init__(
        self,
        profile: Profile,
        goal: str,
        grab: Callable[[], np.ndarray],
        executor: ActionExecutor,
        client=None,
        should_stop: Callable[[], bool] = lambda: False,
        is_paused: Callable[[], bool] = lambda: False,
        on_status: Callable[[str], None] = lambda s: None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        cfg = profile.agent
        self.profile = profile
        self.goal = goal.strip() or "Play the game sensibly and make progress."
        self.grab = grab
        self.executor = executor
        self.model = cfg.get("model", DEFAULT_MODEL)
        self.effort = cfg.get("effort", "low")
        self.max_steps = int(cfg.get("max_steps", 150))
        self.max_cost = float(cfg.get("max_cost_usd", 3.0))
        self.width = int(cfg.get("screenshot_width", 1280))
        self.keep_tool_uses = int(cfg.get("keep_tool_uses", 8))
        self.should_stop = should_stop
        self.is_paused = is_paused
        self.on_status = on_status
        self.sleep = sleep
        self.cost = 0.0
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client

    # -- prompt -------------------------------------------------------------
    def system_prompt(self) -> str:
        keys = self.profile.agent.get("keys") or {}
        tips = self.profile.agent.get("tips") or []
        return SYSTEM.format(
            game=self.profile.window_owner or self.profile.name,
            description=self.profile.description,
            keys="\n".join(f"- {a}: {k}" for a, k in keys.items()) or "- (none listed; use raw keys)",
            tips=("Tips:\n" + "\n".join(f"- {t}" for t in tips) + "\n") if tips else "",
        )

    def _request(self, messages: list) -> object:
        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=self.system_prompt(),
            tools=TOOLS,
            messages=messages,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
            context_management={
                "edits": [
                    {
                        "type": "clear_tool_uses_20250919",
                        "trigger": {"type": "input_tokens", "value": 30000},
                        "keep": {"type": "tool_uses", "value": self.keep_tool_uses},
                    }
                ]
            },
            betas=["context-management-2025-06-27", "server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

    def _add_cost(self, usage) -> None:
        pin, pout = PRICES.get(self.model, PRICES[DEFAULT_MODEL])
        tokens_in = (
            (getattr(usage, "input_tokens", 0) or 0)
            + 1.25 * (getattr(usage, "cache_creation_input_tokens", 0) or 0)
            + 0.1 * (getattr(usage, "cache_read_input_tokens", 0) or 0)
        )
        self.cost += (tokens_in * pin + (getattr(usage, "output_tokens", 0) or 0) * pout) / 1e6

    # -- main loop ----------------------------------------------------------
    def run(self) -> AgentResult:
        first = [
            {"type": "text", "text": f"Your assignment: {self.goal}\n\nHere is the game right now."},
            _image_block(encode_screenshot(self.grab(), self.width)),
        ]
        messages: list = [{"role": "user", "content": first}]
        nudged = False

        for step in range(1, self.max_steps + 1):
            while self.is_paused() and not self.should_stop():
                self.executor.release_all()
                self.sleep(0.2)
            if self.should_stop():
                return self._result(False, "Stopped by you.", step - 1, "stopped")
            if self.cost >= self.max_cost:
                return self._result(False, f"Spending limit (${self.max_cost:.2f}) reached.", step - 1, "budget")

            self.on_status(f"thinking (turn {step})")
            response = self._request(messages)
            self._add_cost(response.usage)

            if response.stop_reason == "refusal":
                return self._result(False, "Claude declined to continue this assignment.", step, "refusal")
            messages.append({"role": "assistant", "content": response.content})
            calls = [b for b in response.content if b.type == "tool_use"]

            if not calls:
                if nudged:
                    text = next((b.text for b in response.content if b.type == "text"), "")
                    return self._result(False, text or "Claude stopped without finishing.", step, "error")
                nudged = True
                messages.append({"role": "user", "content": "Keep playing: call a game tool, or call finish if you're done."})
                continue
            nudged = False

            results, finished = [], None
            for call in calls:
                if self.should_stop():
                    out_text, is_err = "not run: the user stopped the bot", True
                else:
                    self.on_status(f"{call.name} {call.input}")
                    out = self.executor.execute(call.name, dict(call.input))
                    out_text, is_err = out.text, out.is_error
                    if out.finished:
                        finished = out
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": [{"type": "text", "text": out_text}], "is_error": is_err})

            if finished is not None:
                return self._result(finished.success, finished.text, step, "finished")

            # Attach the new screenshot to the last tool result, so context editing can clear it later.
            results[-1]["content"].append(_image_block(encode_screenshot(self.grab(), self.width)))
            messages.append({"role": "user", "content": results})

        return self._result(False, f"Stopped after {self.max_steps} turns.", self.max_steps, "max_steps")

    def _result(self, success: bool, summary: str, steps: int, reason: str) -> AgentResult:
        self.executor.release_all()
        log.info("agent %s after %d turns, ~$%.2f: %s", reason, steps, self.cost, summary)
        return AgentResult(success, summary, steps, round(self.cost, 4), reason)
