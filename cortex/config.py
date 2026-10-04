"""Game profile loading.

A profile is a YAML file (see ``cortex/profiles/stardew.yaml``) describing the
tile size, CLIP labels/prompts, tool hotbar, task priorities and controls for
one game.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROFILE_DIR = Path(__file__).parent / "profiles"


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
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def label_names(self) -> list[str]:
        return [l.name for l in self.labels]

    @property
    def walkable_labels(self) -> frozenset[str]:
        return frozenset(l.name for l in self.labels if l.walkable)


def load_profile(name_or_path: str | Path) -> Profile:
    """Load a profile by bundled name (``"stardew"``) or by file path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = PROFILE_DIR / f"{name_or_path}.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)
    return profile_from_dict(raw, name=path.stem)


def profile_from_dict(raw: dict[str, Any], name: str = "custom") -> Profile:
    game = raw.get("game", {})
    labels = tuple(
        LabelSpec(name=k, prompts=tuple(v["prompts"]), walkable=bool(v.get("walkable", True)))
        for k, v in raw["labels"].items()
    )
    label_names = {l.name for l in labels}

    tasks = []
    for t in raw.get("tasks", []):
        spec = TaskSpec(
            name=t["name"],
            targets=tuple(t["targets"]),
            action=t["action"],
            tool=t.get("tool"),
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

    return Profile(
        name=name,
        window_owner=game.get("window_owner", ""),
        tile_px=int(game.get("tile_px", 64)),
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
        raw=raw,
    )
