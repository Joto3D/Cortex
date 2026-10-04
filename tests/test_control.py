import numpy as np

from cortex.control.controller import Controller
from cortex.overlay.debug import render
from cortex.perception.perceiver import Perceiver
from cortex.planner.actions import Interact, Move, Stop, UseTool
from cortex.planner.farm_planner import FarmPlanner

from .conftest import render_grid


class RecordingInput:
    def __init__(self):
        self.events = []

    def key(self, name, down):
        self.events.append(("key", name, down))

    def mouse_move(self, x, y):
        self.events.append(("move", x, y))

    def mouse_button(self, x, y, button, down):
        self.events.append(("click", button, down, x, y))


def make_controller(profile):
    io = RecordingInput()
    ctl = Controller(profile, io, to_screen=lambda x, y: (100 + x / 2, 50 + y / 2), sleep=lambda s: None)
    return ctl, io


def test_movement_keys_are_held_and_switched(profile):
    ctl, io = make_controller(profile)
    ctl.apply(Move("right"))
    ctl.apply(Move("right"))
    assert io.events == [("key", "d", True)]
    ctl.apply(Move("up"))
    assert io.events[1:] == [("key", "d", False), ("key", "w", True)]
    ctl.apply(Stop())
    assert io.events[-1] == ("key", "w", False)
    assert ctl.held is None


def test_tool_use_selects_slot_once_and_clicks_in_screen_space(profile):
    ctl, io = make_controller(profile)
    ctl.apply(UseTool("watering_can", (1, 1), (40.0, 60.0)))
    assert io.events[:2] == [("key", "2", True), ("key", "2", False)]
    assert io.events[2] == ("move", 120.0, 80.0)
    assert io.events[3:] == [("click", "left", True, 120.0, 80.0), ("click", "left", False, 120.0, 80.0)]
    io.events.clear()
    ctl.apply(UseTool("watering_can", (1, 2), (56.0, 60.0)))
    assert not any(e[0] == "key" for e in io.events)


def test_interact_is_right_click(profile):
    ctl, io = make_controller(profile)
    ctl.apply(Interact((0, 0), (8.0, 8.0)))
    assert ("click", "right", True, 104.0, 54.0) in io.events


def test_end_to_end_perceive_plan_act(profile, encoder):
    """Full pipeline on a synthetic frame: player walks toward the dry crop, and the overlay renders."""
    frame = render_grid(
        [
            ["grass", "grass", "grass", "grass", "grass"],
            ["grass", "player", "grass", "grass", "crop_dry"],
            ["grass", "grass", "grass", "grass", "grass"],
        ],
        phase=(3, 5),
    )
    world = Perceiver(profile, encoder).perceive(frame)
    planner = FarmPlanner(profile)
    action = planner.step(world, now=0.0)
    assert action == Move("right")

    ctl, io = make_controller(profile)
    ctl.apply(action)
    assert io.events == [("key", "d", True)]

    img = render(frame, world, planner.last_plan)
    assert img.shape == frame.shape and img.dtype == np.uint8
    assert not np.array_equal(img, frame)
