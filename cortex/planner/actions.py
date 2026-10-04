"""Actions the planner emits. They describe intent, and the controller turns them into input events."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Move:
    direction: str  # "up" | "down" | "left" | "right"


@dataclass(frozen=True)
class Stop:
    pass


@dataclass(frozen=True)
class UseTool:
    tool: str                      # key into profile.tools
    tile: tuple[int, int]
    screen_px: tuple[float, float]  # target tile centre in frame pixels


@dataclass(frozen=True)
class Interact:
    tile: tuple[int, int]
    screen_px: tuple[float, float]


@dataclass(frozen=True)
class Wait:
    reason: str


@dataclass(frozen=True)
class Done:
    reason: str


Action = Move | Stop | UseTool | Interact | Wait | Done
