"""Rule-based farming planner.

It runs every frame in microseconds, so there is no LLM in the hot loop. Each
tick it:
  * holds back while a menu or dialog is open, a tool animation is playing, or energy is low,
  * takes the current step of the Mission (from the player's assignment). With no
    mission, every task counts and the highest-priority one with a target on screen wins,
  * walks (BFS) to the nearest tile next to a target, stops, then clicks it.

A step ends when its action limit is reached, or when its targets have been
missing for ``empty_frames_to_finish`` frames in a row. Waiting a few frames
means one flickery frame of perception doesn't skip a step.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cortex.assignment import Mission
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


@dataclass(frozen=True)
class _Step:
    tasks: tuple[TaskSpec, ...]
    limit: int | None
    label: str


class FarmPlanner:
    def __init__(self, profile: Profile, mission: Mission | None = None, empty_frames_to_finish: int = 5):
        self.profile = profile
        self.controls = profile.controls
        self.mission = mission
        self.s = _State()
        self.last_plan: Plan | None = None
        self.empty_frames_to_finish = empty_frames_to_finish
        if mission is None:
            self._steps = [_Step(profile.tasks, None, "all chores")]
        else:
            self._steps = [_Step((profile.task(st.task),), st.limit, st.describe()) for st in mission.steps]
        self.step_index = 0
        self._actions_in_step = 0
        self._empty_frames = 0

    @property
    def current_step(self) -> str:
        if self.step_index >= len(self._steps):
            return "finished"
        st = self._steps[self.step_index]
        done = f" {self._actions_in_step}/{st.limit}" if st.limit else ""
        return f"step {self.step_index + 1}/{len(self._steps)}: {st.label}{done}"

    def _advance(self) -> None:
        self.step_index += 1
        self._actions_in_step = 0
        self._empty_frames = 0
        self.s.attempts.clear()

    # -- planning -----------------------------------------------------------
    def find_plan(self, world: WorldState, tasks: tuple[TaskSpec, ...] | None = None) -> Plan | None:
        smap = world.map
        ignored = np.zeros(smap.shape, bool)
        for (r, c), n in self.s.attempts.items():
            if n >= self.controls.max_attempts_per_tile and smap.in_bounds(r, c):
                ignored[r, c] = True

        for task in tasks if tasks is not None else self.profile.tasks:
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

        while True:
            if self.step_index >= len(self._steps):
                self.last_plan = None
                done = "assignment complete" if self.mission else "no reachable work on screen"
                return self._halt(Done(done))
            step = self._steps[self.step_index]
            if step.limit is not None and self._actions_in_step >= step.limit:
                self._advance()
                continue
            plan = self.find_plan(world, step.tasks)
            self.last_plan = plan
            if plan is not None:
                self._empty_frames = 0
                break
            self._empty_frames += 1
            if self._empty_frames < self.empty_frames_to_finish:
                return self._halt(Wait(f"looking for work ({step.label})"))
            self._advance()

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
        self._actions_in_step += 1
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
