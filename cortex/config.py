"""Game profile loading.

A profile is a YAML file describing one game. ``engine`` picks how it's played:

* ``grid`` (default): fast tile-grid CLIP perception + rule-based planner, for
  top-down 2D games (see ``cortex/profiles/stardew.yaml``).
* ``agent``: a Claude vision agent that can play any game, including 3D ones
  (see ``cortex/profiles/generic_3d.yaml``).
* ``skill``: plays skills you taught it by recording yourself (``cortex/teach``).
  Works for any game, fully offline, no API key. Any profile can also play skills.

Bundled profiles live in ``cortex/profiles``. Games you add yourself are saved
to ``~/Library/Application Support/Cortex/games`` (override with CORTEX_HOME)
and take precedence over bundled ones with the same name.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import os

import yaml

PROFILE_DIR = Path(__file__).parent / "profiles"
ENGINES = ("grid", "agent", "skill")


def cortex_home() -> Path:
    """Where user data (added games, settings) lives."""
    env = os.environ.get("CORTEX_HOME")
    if env:
        return Path(env)
    return Path.home() / "Library" / "Application Support" / "Cortex"


def user_profile_dir() -> Path:
    return cortex_home() / "games"


@dataclass(frozen=True)
class ReflexSpec:
    """An instant local reaction: when a screen region matches ``prompt`` (vs ``otherwise``), press ``keys``."""

    name: str
    prompt: str
    otherwise: str
    keys: tuple[str, ...]
    roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    threshold: float = 0.7
    cooldown_s: float = 2.0


@dataclass(frozen=True)
class LabelSpec:
    name: str
    prompts: tuple[str, ...]
    walkable: bool


@dataclass(frozen=True)
class TaskSpec:
    name: str
    targets: tuple[str, ...]
    action: str  # "tool" | "interact"
    tool: str | None = None
    description: str = ""
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class Controls:
    up: str = "w"
    down: str = "s"
    left: str = "a"
    right: str = "d"
    kill_switch: str = "f12"
    pause: str = "f11"
    tool_cooldown_s: float = 0.45
    interact_cooldown_s: float = 0.30
    min_hold_s: float = 0.08
    jitter_ms: tuple[int, int] = (15, 40)
    max_attempts_per_tile: int = 3


@dataclass(frozen=True)
class Profile:
    name: str
    window_owner: str
    tile_px: int
    grid_phase: tuple[int, int] | None
    perception: dict[str, Any]
    labels: tuple[LabelSpec, ...]
    scenes: dict[str, tuple[str, ...]]
    tools: dict[str, str]
    tasks: tuple[TaskSpec, ...]
    controls: Controls
    energy_roi: tuple[float, float, float, float] | None
    min_energy: float
    target_hz: float
    pause_when_unfocused: bool
    engine: str = "grid"
    description: str = ""
    agent: dict[str, Any] = field(default_factory=dict)
    reflexes: tuple[ReflexSpec, ...] = ()
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def label_names(self) -> list[str]:
        return [l.name for l in self.labels]

    def task(self, name: str) -> TaskSpec:
        for t in self.tasks:
            if t.name == name:
                return t
        raise KeyError(name)

    @property
    def walkable_labels(self) -> frozenset[str]:
        return frozenset(l.name for l in self.labels if l.walkable)


def find_profile(name_or_path: str | Path) -> Path:
    path = Path(name_or_path)
    if path.suffix:
        return path
    for d in (user_profile_dir(), PROFILE_DIR):
        candidate = d / f"{name_or_path}.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no game profile named {str(name_or_path)!r}; known: {', '.join(list_profiles())}")


def list_profiles() -> list[str]:
    """Names of every available game profile (your added games first, then bundled ones)."""
    names: list[str] = []
    for d in (user_profile_dir(), PROFILE_DIR):
        if d.is_dir():
            names += [p.stem for p in sorted(d.glob("*.yaml")) if p.stem not in names]
    return names


def load_profile(name_or_path: str | Path) -> Profile:
    """Load a profile by name (``"stardew"``) or by file path."""
    path = find_profile(name_or_path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return profile_from_dict(raw, name=path.stem)


def profile_from_dict(raw: dict[str, Any], name: str = "custom") -> Profile:
    game = raw.get("game", {})
    engine = raw.get("engine", "grid")
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, not {engine!r}")
    labels = tuple(
        LabelSpec(name=k, prompts=tuple(v["prompts"]), walkable=bool(v.get("walkable", True)))
        for k, v in (raw.get("labels") or {}).items()
    )
    if engine == "grid" and not labels:
        raise ValueError("a grid-engine profile needs 'labels'")
    label_names = {l.name for l in labels}

    tasks = []
    for t in raw.get("tasks", []):
        spec = TaskSpec(
            name=t["name"],
            targets=tuple(t["targets"]),
            action=t["action"],
            tool=t.get("tool"),
            description=t.get("description", ""),
            keywords=tuple(k.lower() for k in t.get("keywords", ())),
        )
        unknown = set(spec.targets) - label_names
        if unknown:
            raise ValueError(f"task {spec.name!r} targets unknown labels: {sorted(unknown)}")
        if spec.action not in ("tool", "interact"):
            raise ValueError(f"task {spec.name!r}: action must be 'tool' or 'interact'")
        if spec.action == "tool" and spec.tool not in raw.get("tools", {}):
            raise ValueError(f"task {spec.name!r}: tool {spec.tool!r} missing from 'tools'")
        tasks.append(spec)

    c = dict(raw.get("controls", {}))
    if "jitter_ms" in c:
        c["jitter_ms"] = tuple(c["jitter_ms"])
    hud = raw.get("hud", {})
    loop = raw.get("loop", {})
    reflexes = tuple(
        ReflexSpec(
            name=r["name"],
            prompt=r["prompt"],
            otherwise=r.get("otherwise", "a normal video game screen"),
            keys=tuple(str(k) for k in r["keys"]),
            roi=tuple(r.get("roi", (0.0, 0.0, 1.0, 1.0))),
            threshold=float(r.get("threshold", 0.7)),
            cooldown_s=float(r.get("cooldown_s", 2.0)),
        )
        for r in raw.get("reflexes") or ()
    )

    return Profile(
        name=name,
        window_owner=game.get("window_owner", ""),
        tile_px=int(game.get("tile_px", 64)),
        grid_phase=tuple(game["grid_phase"]) if game.get("grid_phase") else None,
        perception=dict(raw.get("perception", {})),
        labels=labels,
        scenes={k: tuple(v) for k, v in raw.get("scenes", {}).items()},
        tools={k: str(v) for k, v in raw.get("tools", {}).items()},
        tasks=tuple(tasks),
        controls=Controls(**c),
        energy_roi=tuple(hud["energy_roi"]) if "energy_roi" in hud else None,
        min_energy=float(hud.get("min_energy", 0.0)),
        target_hz=float(loop.get("target_hz", 30)),
        pause_when_unfocused=bool(loop.get("pause_when_unfocused", True)),
        engine=engine,
        description=str(game.get("description", "")),
        agent=dict(raw.get("agent") or {}),
        reflexes=reflexes,
        raw=raw,
    )
