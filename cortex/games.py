"""Add any game to Cortex: Claude drafts a game profile from the game's name and a screenshot.

    python -m cortex.games list
    python -m cortex.games add "Minecraft"            # picks the window whose app is named Minecraft
    python -m cortex.games add "Valheim" --window "valheim"

New games use the ``agent`` engine, which works for 2D and 3D games alike. The
profile is saved to ~/Library/Application Support/Cortex/games/<name>.yaml.
Open it to fine-tune key bindings, tips, or reflexes.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import yaml

from cortex.config import list_profiles, load_profile, profile_from_dict, user_profile_dir
from cortex.control.input_mac import KEYCODES

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
_VALID_KEYS = set(KEYCODES) | {"mouse_left", "mouse_right"}

_PROMPT = """I want an AI agent to play the video game "{game}" on a Mac for me. It plays by pressing keys \
and moving the mouse while looking at screenshots. Fill in a profile for this game.

- description: 2-3 sentences: what kind of game it is, the view (first person / third person / top-down / side), \
and how you move and interact.
- keys: the game's default controls as action -> key. Keys must be lowercase key names such as w, a, s, d, space, \
shift, ctrl, alt, tab, escape, e, q, 1-9, f1-f12, up/down/left/right, or mouse_left / mouse_right.
- tips: up to 6 short practical tips for an AI playing it from screenshots.
- reflexes: 0-3 dangers worth reacting to instantly, judged by a small image model on part of the screen \
(e.g. a nearly empty health bar). roi is [x0, y0, x1, y1] as fractions of the window where that HUD element \
is. prompt describes the danger state and otherwise describes the safe state, both as short visual \
descriptions. keys is what to press. Leave the list empty if unsure.
- look_px_per_degree: mouse pixels per degree of camera turn at default sensitivity (use 6 if unsure; \
0 if the game has no mouse-look).
- example_assignments: 3 short things a player might ask the agent to do.
{screenshot_note}"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "keys": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"action": {"type": "string"}, "key": {"type": "string"}},
                "required": ["action", "key"],
                "additionalProperties": False,
            },
        },
        "tips": {"type": "array", "items": {"type": "string"}},
        "reflexes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "roi": {"type": "array", "items": {"type": "number"}},
                    "prompt": {"type": "string"},
                    "otherwise": {"type": "string"},
                    "keys": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "roi", "prompt", "otherwise", "keys"],
                "additionalProperties": False,
            },
        },
        "look_px_per_degree": {"type": "number"},
        "example_assignments": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["description", "keys", "tips", "reflexes", "look_px_per_degree", "example_assignments"],
    "additionalProperties": False,
}


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "game"


def list_windows() -> list[str]:
    """App names that currently have a visible window (macOS)."""
    import Quartz

    opts = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    names = {
        str(w.get("kCGWindowOwnerName"))
        for w in Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID) or []
        if w.get("kCGWindowLayer", 0) == 0 and w.get("kCGWindowOwnerName")
    }
    return sorted(names - {"Window Server", "Dock", "Cortex"})


def ask_claude(game: str, screenshot_b64: str | None, client=None, model: str = DEFAULT_MODEL) -> dict:
    """One structured-output call that returns the profile fields as a dict."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    note = "A screenshot of the game as it looks right now is attached; use it to place HUD regions." if screenshot_b64 else ""
    content: list = [{"type": "text", "text": _PROMPT.format(game=game, screenshot_note=note)}]
    if screenshot_b64:
        content.insert(0, {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": screenshot_b64}})
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        messages=[{"role": "user", "content": content}],
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": _SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to describe this game")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise RuntimeError("Claude returned no profile")
    return json.loads(text)


def build_profile(game: str, window_owner: str, draft: dict, model: str = DEFAULT_MODEL) -> dict:
    """Turn Claude's draft into a valid agent profile dict, dropping anything malformed."""
    keys = {}
    for item in draft.get("keys", []):
        action = slugify(str(item.get("action", "")))
        key = str(item.get("key", "")).strip().lower()
        if action and key in _VALID_KEYS:
            keys[action] = key
        else:
            log.warning("skipping control %r -> %r (unknown key)", item.get("action"), item.get("key"))

    reflexes = []
    for r in draft.get("reflexes", []):
        roi = [min(1.0, max(0.0, float(v))) for v in r.get("roi", [])]
        rkeys = [str(k).lower() for k in r.get("keys", []) if str(k).lower() in _VALID_KEYS or str(k).lower() in keys]
        if len(roi) != 4 or roi[0] >= roi[2] or roi[1] >= roi[3] or not rkeys:
            log.warning("skipping reflex %r (bad region or keys)", r.get("name"))
            continue
        reflexes.append({
            "name": slugify(str(r["name"])), "roi": roi, "prompt": r["prompt"], "otherwise": r["otherwise"],
            "keys": rkeys, "threshold": 0.75, "cooldown_s": 3,
        })

    look = float(draft.get("look_px_per_degree") or 0) or 6.0
    raw = {
        "engine": "agent",
        "game": {"window_owner": window_owner, "description": str(draft.get("description", "")).strip()},
        "agent": {
            "model": model,
            "effort": "low",
            "max_steps": 150,
            "max_cost_usd": 3.0,
            "screenshot_width": 1280,
            "look_px_per_degree": round(max(0.5, min(look, 50.0)), 2),
            "keys": keys,
            "tips": [str(t) for t in draft.get("tips", [])][:6],
            "example_assignments": [str(t) for t in draft.get("example_assignments", [])][:5],
        },
        "reflexes": reflexes,
        "controls": {"kill_switch": "f12", "pause": "f11"},
        "loop": {"pause_when_unfocused": True},
    }
    profile_from_dict(raw, name=slugify(game))  # raises if invalid
    return raw


def save_profile(game: str, raw: dict, directory: Path | None = None) -> Path:
    d = directory or user_profile_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{slugify(game)}.yaml"
    header = f"# Cortex profile for {game}, drafted by Claude. Edit freely: keys, tips and reflexes.\n"
    path.write_text(header + yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, width=100))
    return path


def add_game(game: str, window_owner: str | None = None, screenshot: bool = True, client=None, model: str = DEFAULT_MODEL) -> Path:
    """Draft and save a profile for ``game``. Returns the saved file path."""
    owner = window_owner or game
    shot = None
    if screenshot:
        try:
            from cortex.agent.agent import encode_screenshot
            from cortex.capture.screen import WindowCapture

            shot = encode_screenshot(WindowCapture(owner).grab(), 1280)
        except Exception as e:  # game not running, not macOS, no permission...
            log.warning("no screenshot of %r (%s); drafting from the name only", owner, e)
    draft = ask_claude(game, shot, client=client, model=model)
    return save_profile(game, build_profile(game, owner, draft, model))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Manage the games Cortex can play.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list known games")
    sub.add_parser("windows", help="list open app windows (to find a game's window name)")
    add = sub.add_parser("add", help="add a game (Claude drafts its profile)")
    add.add_argument("game", help='the game name, e.g. "Minecraft"')
    add.add_argument("--window", help="app name of the game window, if different from the game name")
    add.add_argument("--no-screenshot", action="store_true")
    add.add_argument("--model", default=DEFAULT_MODEL)
    show = sub.add_parser("show", help="print a game's profile")
    show.add_argument("game")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if a.cmd == "list":
        for name in list_profiles():
            p = load_profile(name)
            print(f"{name:20s} {p.engine:6s} {p.window_owner}")
    elif a.cmd == "windows":
        print("\n".join(list_windows()))
    elif a.cmd == "show":
        from cortex.config import find_profile

        print(find_profile(a.game).read_text())
    elif a.cmd == "add":
        path = add_game(a.game, a.window, not a.no_screenshot, model=a.model)
        print(f"Saved {path}\nTry it:  python -m cortex.loop --game {path.stem}")
        examples = load_profile(path).agent.get("example_assignments") or []
        if examples:
            print("Ideas:  " + " | ".join(examples))
    else:  # pragma: no cover
        ap.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
