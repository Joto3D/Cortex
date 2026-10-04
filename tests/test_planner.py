import numpy as np

from cortex.assignment import Mission, Step
from cortex.perception.tilegrid import SemanticMap
from cortex.planner.actions import Done, Interact, Move, Stop, UseTool, Wait
from cortex.planner.farm_planner import FarmPlanner
from cortex.planner.pathing import bfs_to_any, dilate8
from cortex.world.state import WorldState

from .conftest import COLORS, TILE, WALKABLE

NAMES = list(COLORS)


def world_from(rows: list[str], player=None, **kw) -> WorldState:
    """Build a world from ASCII art: . grass  d crop_dry  w crop_wet  r crop_ripe  # stone  x weed  @ player."""
    key = {".": "grass", "d": "crop_dry", "w": "crop_wet", "r": "crop_ripe", "#": "stone", "x": "weed", "@": "player"}
    labels = np.array([[NAMES.index(key[ch]) for ch in row] for row in rows], np.int32)
    if player is None:
        rr, cc = np.nonzero(labels == NAMES.index("player"))
        player = (int(rr[0]), int(cc[0]))
    smap = SemanticMap(labels, np.ones(labels.shape, np.float32), (0, 0), TILE, NAMES)
    walkable = np.isin(labels, [NAMES.index(n) for n in WALKABLE])
    return WorldState(map=smap, player=player, walkable=walkable, **kw)


def test_dilate8():
    m = np.zeros((5, 5), bool)
    m[2, 2] = True
    assert dilate8(m).sum() == 9
    d = dilate8(m, include_self=False)
    assert d.sum() == 8 and not d[2, 2]
    corner = np.zeros((3, 3), bool)
    corner[0, 0] = True
    assert dilate8(corner).sum() == 4


def test_bfs_routes_around_obstacles():
    walk = np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 1, 1],
        ],
        bool,
    )
    goal = np.zeros_like(walk)
    goal[2, 2] = True
    path = bfs_to_any(walk, (2, 0), goal)
    assert path[0] == (2, 0) and path[-1] == (2, 2)
    assert len(path) == 9  # up, across the top, down the right side
    assert bfs_to_any(walk, (2, 0), np.zeros_like(walk)) is None


def test_walks_to_crop_stops_then_waters(profile):
    planner = FarmPlanner(profile)
    w = world_from(["@...d"])
    a = planner.step(w, now=0.0)
    assert a == Move("right")
    assert planner.last_plan.target == (0, 4)

    # Arrived next to the crop: stop first, then use the watering can on it.
    w = world_from(["...@d"])
    assert planner.step(w, now=0.1) == Stop()
    a = planner.step(w, now=0.2)
    assert isinstance(a, UseTool) and a.tool == "watering_can" and a.tile == (0, 4)
    assert a.screen_px == (4.5 * TILE, 0.5 * TILE)
    # Tool animation: no new decisions until the cooldown ends.
    assert planner.step(w, now=0.3) == Wait("animation")


def test_harvest_beats_watering_and_uses_interact(profile):
    planner = FarmPlanner(profile)
    w = world_from(["d@r"])
    a = planner.step(w, now=0.0)
    assert isinstance(a, Interact) and a.tile == (0, 2)


def test_can_work_diagonally_and_inside_crop_rows(profile):
    planner = FarmPlanner(profile)
    w = world_from(
        [
            "...",
            ".@.",
            "..d",
        ]
    )
    a = planner.step(w, now=0.0)
    assert isinstance(a, UseTool) and a.tile == (2, 2)


def test_routes_around_stone(profile):
    planner = FarmPlanner(profile)
    w = world_from(
        [
            "@#.",
            ".#.",
            "...",
            "...",
            "..d",
        ]
    )
    assert planner.step(w, now=0.0) == Move("down")
    path = planner.last_plan.path
    assert all(w.walkable[p] for p in path[1:])


def test_gives_up_on_a_tile_after_max_attempts(profile):
    planner = FarmPlanner(profile, empty_frames_to_finish=1)
    w = world_from(["@d"])
    t = 0.0
    for _ in range(profile.controls.max_attempts_per_tile):
        assert isinstance(planner.step(w, now=t), UseTool)
        t += 10
    assert isinstance(planner.step(w, now=t), Done)


def test_halts_on_menus_low_energy_and_no_work(profile):
    planner = FarmPlanner(profile, empty_frames_to_finish=1)
    assert isinstance(planner.step(world_from(["@d"], scene="dialog"), now=0), Wait)
    assert isinstance(planner.step(world_from(["@d"], energy=0.05), now=0), Done)
    assert planner.step(world_from(["@ww"]), now=0) == Done("no reachable work on screen")


def test_unreachable_target_is_skipped(profile):
    planner = FarmPlanner(profile, empty_frames_to_finish=1)
    w = world_from(
        [
            "@.#..",
            "..#.d",
            "..#..",
        ]
    )
    assert isinstance(planner.step(w, now=0), Done)


def test_waits_a_few_frames_before_deciding_there_is_no_work(profile):
    planner = FarmPlanner(profile, empty_frames_to_finish=3)
    w = world_from(["@ww"])
    assert isinstance(planner.step(w, now=0), Wait)
    assert isinstance(planner.step(w, now=0), Wait)
    assert isinstance(planner.step(w, now=0), Done)


def test_mission_runs_steps_in_order_ignoring_other_tasks(profile):
    # Watering is asked for first, so the bot waters even though harvesting has higher default priority.
    mission = Mission("water then harvest", (Step("water"), Step("harvest")))
    planner = FarmPlanner(profile, mission, empty_frames_to_finish=1)
    w = world_from(["d@r"])
    a = planner.step(w, now=0)
    assert isinstance(a, UseTool) and a.tile == (0, 0)
    assert planner.current_step.startswith("step 1/2")

    # The crop is watered, so step 1 runs out of targets and step 2 starts on the same frame.
    w = world_from(["w@r"])
    a = planner.step(w, now=10)
    assert isinstance(a, Interact) and a.tile == (0, 2)
    assert planner.current_step.startswith("step 2/2")

    assert planner.step(world_from(["w@w"]), now=20) == Done("assignment complete")


def test_mission_step_limit(profile):
    mission = Mission("clear 2 weeds", (Step("weeds", limit=2),))
    planner = FarmPlanner(profile, mission, empty_frames_to_finish=1)
    w = world_from(["x@x", "xxx"])
    assert isinstance(planner.step(w, now=0), UseTool)
    assert planner.current_step.endswith("1/2")
    assert isinstance(planner.step(w, now=10), UseTool)
    assert planner.step(w, now=20) == Done("assignment complete")
