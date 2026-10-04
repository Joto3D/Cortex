"""Rule-based farming planner.

It runs every frame in microseconds, so there is no LLM in the hot loop. Each
tick it:
  * holds back while a menu or dialog is open, a tool animation is playing, or energy is low,
  * takes the highest-priority task that has a target on screen,
  * walks (BFS) to the nearest tile next to a target, stops, then clicks it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cortex.config import Profile, TaskSpec
from cortex.world.state import WorldState

from .actions import Action, Done, Interact, Move, Stop, UseTool, Wait
from .pathing import adjacent_target, bfs_to_any, dilate8, direction


@dataclass
class Plan:
    task: TaskSpec
    path: list[tuple[int, int]]
    target: tuple[int, int]


@dataclass
class _State:
    moving: str | None = None
    move_started: float = 0.0
    cooldown_until: float = 0.0
    attempts: dict[tuple[int, int], int] = field(default_factory=dict)


class FarmPlanner:
    def __init__(self, profile: Profile):
        self.profile = profile
        self.controls = profile.controls
        self.s = _State()
        self.last_plan: Plan | None = None

    # -- planning -----------------------------------------------------------
    def find_plan(self, world: WorldState) -> Plan | None:
        smap = world.map
        ignored = np.zeros(smap.shape, bool)
        for (r, c), n in self.s.attempts.items():
            if n >= self.controls.max_attempts_per_tile and smap.in_bounds(r, c):
                ignored[r, c] = True

        for task in self.profile.tasks:
            targets = smap.mask(*task.targets) & ~ignored
            if not targets.any():
                continue
            # Stand on any walkable tile touching a target other than itself
            # (crops are walkable, so standing in a crop row is fine).
            stand_ok = world.walkable & dilate8(targets, include_self=False)
            path = bfs_to_any(world.walkable, world.player, stand_ok)
            if path is None:
                continue
            target = adjacent_target(path[-1], _without(targets, path[-1]))
            if target is None:
                continue
            return Plan(task=task, path=path, target=target)
        return None

    # -- per-frame decision -----------------------------------------------------
    def step(self, world: WorldState, now: float) -> Action:
        if world.scene != "world":
            return self._halt(Wait(f"scene is {world.scene}"))
        if now < self.s.cooldown_until:
            return Wait("animation")
        if world.energy < self.profile.min_energy:
            return self._halt(Done(f"energy low ({world.energy:.0%})"))

        plan = self.find_plan(world)
        self.last_plan = plan
        if plan is None:
            return self._halt(Done("no reachable work on screen"))

        if len(plan.path) > 1:
            want = direction(plan.path[0], plan.path[1])
            s = self.s
            if s.moving and want != s.moving and now - s.move_started < self.controls.min_hold_s:
                want = s.moving  # hysteresis: avoid jittering between directions at tile edges
            if want != s.moving:
                s.moving, s.move_started = want, now
            s.attempts.clear()  # screen tiles shift as the camera scrolls
            return Move(want)

        # At the stand tile. Stop first, so the next frame sees a settled camera and grid.
        if self.s.moving:
            self.s.moving = None
            return Stop()

        t = plan.target
        self.s.attempts[t] = self.s.attempts.get(t, 0) + 1
        px = world.map.tile_center_px(*t)
        if plan.task.action == "interact":
            self.s.cooldown_until = now + self.controls.interact_cooldown_s
            return Interact(tile=t, screen_px=px)
        self.s.cooldown_until = now + self.controls.tool_cooldown_s
        return UseTool(tool=plan.task.tool or "", tile=t, screen_px=px)

    def _halt(self, action: Action) -> Action:
        self.s.moving = None
        return action


def _without(mask: np.ndarray, rc: tuple[int, int]) -> np.ndarray:
    m = mask.copy()
    m[rc] = False
    return m
