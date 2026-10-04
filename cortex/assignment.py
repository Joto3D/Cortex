"""Turn a plain-English assignment into a Mission the bot can execute.

    "harvest everything, then water the crops and clear 5 rocks"
      -> [harvest (until done), water (until done), clear_stone (x5)]

Two interpreters:
  * ClaudeInterpreter makes one Claude API call with a JSON-schema-constrained output.
    It understands free-form phrasing, and is only called once, *before* the game
    loop starts, so it never slows down the bot.
  * KeywordInterpreter is an offline fallback that matches each task's ``keywords``
    from the profile. It needs no API key and no network.

``interpret()`` tries Claude first and falls back to keywords automatically.
Preview a plan without playing:  python -m cortex.assignment "water then harvest"
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field

from cortex.config import Profile, load_profile

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"


@dataclass(frozen=True)
class Step:
    task: str                 # a task name from the profile
    limit: int | None = None  # how many actions; None = until no targets are visible

    def describe(self) -> str:
        return f"{self.task} x{self.limit}" if self.limit else f"{self.task} (until done)"


@dataclass(frozen=True)
class Mission:
    assignment: str
    steps: tuple[Step, ...]
    source: str = "default"                  # "claude" | "keywords" | "default"
    summary: str = ""
    unsupported: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def everything(cls, profile: Profile) -> "Mission":
        return cls("do all the farm chores", tuple(Step(t.name) for t in profile.tasks), "default", "Do every chore.")

    def describe(self) -> str:
        lines = [f"Assignment: {self.assignment!r}  (understood by: {self.source})"]
        if self.summary:
            lines.append(f"Plan: {self.summary}")
        lines += [f"  {i}. {s.describe()}" for i, s in enumerate(self.steps, 1)]
        if self.unsupported:
            lines.append("Can't do: " + "; ".join(self.unsupported))
        return "\n".join(lines)


class AssignmentError(Exception):
    pass


# --- offline keyword interpreter -------------------------------------------

_SPLIT = re.compile(r"\s*(?:[,;.!]|\bthen\b|\bafter that\b|\bafterwards\b|\band\b|\bnext\b|\bfinally\b)\s*")
_EVERYTHING = re.compile(r"\b(everything|all (?:the |my )?(?:chores|work|tasks|jobs)|whole farm|do it all|all of it)\b")
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "a dozen": 12,
}
_NUMBER = re.compile(r"\b(\d+|" + "|".join(_NUMBER_WORDS) + r")\b")
_FILLER = {
    "first", "please", "can", "you", "could", "go", "the", "a", "an", "my", "all", "of", "to", "then",
    "it", "i", "want", "would", "like", "do", "some", "start", "by", "with", "also", "now", "and",
}


class KeywordInterpreter:
    def __init__(self, profile: Profile):
        self.profile = profile
        self._patterns = [
            (t.name, re.compile(r"\b(?:" + "|".join(map(re.escape, t.keywords)) + r")(?:s|es|ing|ed)?\b"))
            for t in profile.tasks
            if t.keywords
        ]

    def interpret(self, assignment: str) -> Mission:
        steps: list[Step] = []
        unsupported: list[str] = []
        for clause in _SPLIT.split(assignment.lower()):
            clause = clause.strip()
            if not clause:
                continue
            hits = sorted(
                (m.start(), name) for name, pat in self._patterns for m in [pat.search(clause)] if m
            )
            if hits:
                limit = _parse_number(clause)
                steps += [Step(name, limit) for _, name in hits]
            elif _EVERYTHING.search(clause):
                steps += [Step(t.name) for t in self.profile.tasks]
            elif set(re.findall(r"[a-z']+", clause)) - _FILLER:
                unsupported.append(clause)
        summary = ", then ".join(s.describe() for s in steps)
        return Mission(assignment, tuple(steps), "keywords", summary, tuple(unsupported))


def _parse_number(clause: str) -> int | None:
    m = _NUMBER.search(clause)
    if not m:
        return None
    word = m.group(1)
    n = int(word) if word.isdigit() else _NUMBER_WORDS[word]
    return n if n > 0 else None


# --- Claude interpreter -----------------------------------------------------

_SYSTEM = """You turn a player's instructions into a plan for a bot that plays a farming game.

The bot can only perform these tasks (name: what it does):
{tasks}

Rules:
- Return the steps in the order the player wants them done. Use task names exactly as listed.
- Set "limit" to a count when the player gives one (e.g. "break 5 rocks" -> 5). Otherwise use null, \
which means "keep going until none are left on screen".
- If the player asks for everything / all the chores, include every task in the listed order.
- Put any part of the request the bot cannot do into "unsupported" as a short phrase. Never map it to an \
unrelated task.
- "summary" is one short, friendly sentence telling the player what the bot will do."""


class ClaudeInterpreter:
    def __init__(self, profile: Profile, model: str = DEFAULT_MODEL, client=None):
        import anthropic

        self.profile = profile
        self.model = model
        self.client = client or anthropic.Anthropic()

    def _schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string", "enum": [t.name for t in self.profile.tasks]},
                            "limit": {"type": ["integer", "null"]},
                        },
                        "required": ["task", "limit"],
                        "additionalProperties": False,
                    },
                },
                "unsupported": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string"},
            },
            "required": ["steps", "unsupported", "summary"],
            "additionalProperties": False,
        }

    def interpret(self, assignment: str) -> Mission:
        tasks = "\n".join(f"- {t.name}: {t.description or ', '.join(t.targets)}" for t in self.profile.tasks)
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=_SYSTEM.format(tasks=tasks),
            messages=[{"role": "user", "content": assignment}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": self._schema()}},
            # If a safety classifier declines, the API retries on a suitable fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise AssignmentError("Claude declined to interpret this assignment")
        if response.stop_reason == "max_tokens":
            raise AssignmentError("Claude's answer was cut off")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise AssignmentError("Claude returned no plan")
        return mission_from_json(assignment, json.loads(text), self.profile, source="claude")


def mission_from_json(assignment: str, data: dict, profile: Profile, source: str) -> Mission:
    """Validate a plan dict (from Claude or a file) against the profile."""
    known = {t.name for t in profile.tasks}
    steps = []
    for s in data.get("steps", []):
        if s.get("task") not in known:
            raise AssignmentError(f"unknown task {s.get('task')!r}")
        limit = s.get("limit")
        if limit is not None and (not isinstance(limit, int) or limit <= 0):
            limit = None
        steps.append(Step(s["task"], limit))
    return Mission(
        assignment,
        tuple(steps),
        source,
        str(data.get("summary", "")),
        tuple(str(u) for u in data.get("unsupported", [])),
    )


class _ClaudeUnavailable(Exception):
    pass


def _ask_claude(assignment: str, profile: Profile, model: str) -> Mission:
    try:
        import anthropic
    except ImportError as e:
        raise _ClaudeUnavailable("the anthropic package isn't installed (pip install -e '.[ai]')") from e
    try:
        return ClaudeInterpreter(profile, model).interpret(assignment)
    except TypeError as e:  # the SDK raises this when no API key or `ant auth login` profile is found
        raise _ClaudeUnavailable(f"no Claude credentials: {e}") from e
    except anthropic.APIConnectionError as e:
        raise _ClaudeUnavailable(f"can't reach the Claude API: {e}") from e
    except anthropic.APIStatusError as e:  # auth, rate limit, server errors (after SDK retries)
        raise _ClaudeUnavailable(f"Claude API error {e.status_code}: {e.message}") from e
    except (AssignmentError, json.JSONDecodeError) as e:
        raise _ClaudeUnavailable(str(e)) from e


def interpret(assignment: str, profile: Profile, mode: str = "auto", model: str = DEFAULT_MODEL) -> Mission:
    """Understand an assignment. ``mode`` is "auto" (Claude, falling back to keywords), "claude" or "offline"."""
    if not assignment.strip():
        return Mission.everything(profile)
    if mode != "offline":
        try:
            return _ask_claude(assignment, profile, model)
        except _ClaudeUnavailable as e:
            if mode == "claude":
                raise AssignmentError(str(e)) from e
            log.warning("Claude unavailable, so using the offline keyword parser: %s", e)
    return KeywordInterpreter(profile).interpret(assignment)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Preview how Cortex understands an assignment (no game needed).")
    ap.add_argument("assignment", nargs="+")
    ap.add_argument("--profile", default="stardew")
    ap.add_argument("--offline", action="store_true", help="use the keyword parser only")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    mission = interpret(" ".join(a.assignment), load_profile(a.profile), "offline" if a.offline else "auto", a.model)
    print(mission.describe())


if __name__ == "__main__":
    main()
