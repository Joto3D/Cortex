"""Main loop: capture → perceive → plan → act, at a fixed rate.

    python -m cortex.loop                                  # asks what you want done
    python -m cortex.loop -a "harvest, then water the crops" --debug
    python -m cortex.loop --dry-run                        # perceive and plan, but send no input

F12 stops the bot and F11 toggles pause (both configurable in the profile).
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

from cortex.assignment import DEFAULT_MODEL, Mission, interpret
from cortex.config import Profile, load_profile
from cortex.planner.actions import Done, Wait
from cortex.planner.farm_planner import FarmPlanner

log = logging.getLogger("cortex")


class Hotkeys:
    """Global kill-switch and pause hotkeys (pynput, needs Input Monitoring permission).

    Pass existing events to share them with another controller (e.g. the menu-bar app).
    """

    def __init__(
        self,
        kill: str,
        pause: str,
        stop: threading.Event | None = None,
        paused: threading.Event | None = None,
    ):
        self.stop = stop or threading.Event()
        self.paused = paused or threading.Event()
        try:
            from pynput import keyboard
        except ImportError:
            log.warning("pynput not installed: no global kill switch, so use Ctrl+C")
            return
        kill_key = getattr(keyboard.Key, kill, None)
        pause_key = getattr(keyboard.Key, pause, None)

        def on_press(key):
            if key == kill_key:
                self.stop.set()
                return False
            if key == pause_key:
                if self.paused.is_set():
                    self.paused.clear()
                else:
                    self.paused.set()
                log.info("paused" if self.paused.is_set() else "resumed")

        self._listener = keyboard.Listener(on_press=on_press, daemon=True)
        self._listener.start()


def build_encoder(profile: Profile):
    from cortex.perception.clip_model import ClipEncoder

    p = profile.perception
    fb = (p.get("fallback_model", "ViT-B-32"), p.get("fallback_pretrained", "laion2b_s34b_b79k"))
    return ClipEncoder(p.get("model", "MobileCLIP-S1"), p.get("pretrained", "datacompdr"), fallback=fb)


def run(
    profile: Profile,
    mission: Mission | None = None,
    debug: bool = False,
    dry_run: bool = False,
    record: Path | None = None,
    stop: threading.Event | None = None,
    paused: threading.Event | None = None,
) -> str:
    """Play a grid-engine game until done or stopped. Returns why it stopped."""
    from cortex.capture.screen import WindowCapture
    from cortex.control.controller import Controller
    from cortex.control.input_mac import MacInput, frontmost_app_name
    from cortex.perception.perceiver import Perceiver

    cap = WindowCapture(profile.window_owner)
    perceiver = Perceiver(profile, build_encoder(profile))
    planner = FarmPlanner(profile, mission)
    backend = _NullInput() if dry_run else MacInput()
    ctl = Controller(profile, backend, to_screen=cap.to_screen)
    keys = Hotkeys(profile.controls.kill_switch, profile.controls.pause, stop, paused)
    if record:
        record.mkdir(parents=True, exist_ok=True)

    period = 1.0 / profile.target_hz
    frame_no = 0
    reason = "stopped by user"
    log.info("running; %s to stop, %s to pause", profile.controls.kill_switch, profile.controls.pause)
    try:
        while not keys.stop.is_set():
            t0 = time.perf_counter()
            if keys.paused.is_set() or (
                profile.pause_when_unfocused and frontmost_app_name() not in (None, profile.window_owner)
            ):
                ctl.release_all()
                time.sleep(0.1)
                continue

            frame = cap.grab()
            t1 = time.perf_counter()
            world = perceiver.perceive(frame)
            t2 = time.perf_counter()
            action = planner.step(world, now=t2)
            if not isinstance(action, Wait) or frame_no % 20 == 0:
                log.debug("%s", action)
            ctl.apply(action)
            t3 = time.perf_counter()

            if record and frame_no % 10 == 0:
                from PIL import Image

                Image.fromarray(frame.copy()).save(record / f"frame_{frame_no:06d}.png")
            if debug:
                status = (
                    f"cap {1e3*(t1-t0):.0f}ms  perc {1e3*(t2-t1):.0f}ms  act {1e3*(t3-t2):.0f}ms  "
                    f"energy {world.energy:.0%}  {world.scene}  {planner.current_step}  {type(action).__name__}"
                )
                from cortex.overlay.debug import show

                if not show(frame, world, planner.last_plan, status):
                    break
            if isinstance(action, Done):
                reason = action.reason
                break
            if frame_no % 40 == 0:
                log.info("%s", planner.current_step)
            frame_no += 1
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        ctl.release_all()
    log.info("stopped: %s", reason)
    return reason


def run_agent(
    profile: Profile,
    goal: str,
    dry_run: bool = False,
    stop: threading.Event | None = None,
    paused: threading.Event | None = None,
    on_status=lambda s: None,
):
    """Play an agent-engine game (any game, including 3D) with Claude. Returns an AgentResult."""
    from cortex.agent import ActionExecutor, GameAgent
    from cortex.capture.screen import WindowCapture
    from cortex.control.input_mac import MacInput, frontmost_app_name

    cap = WindowCapture(profile.window_owner)
    backend = _NullInput() if dry_run else MacInput()
    keys = Hotkeys(profile.controls.kill_switch, profile.controls.pause, stop, paused)
    cfg = profile.agent

    def to_screen(fx: float, fy: float) -> tuple[float, float]:
        w = cap.window
        return w.x + fx * w.width, w.y + fy * w.height

    def make_executor() -> ActionExecutor:
        return ActionExecutor(
            backend, {k.lower(): str(v).lower() for k, v in (cfg.get("keys") or {}).items()},
            to_screen, float(cfg.get("look_px_per_degree", 6.0)),
        )

    def unfocused() -> bool:
        return profile.pause_when_unfocused and frontmost_app_name() not in (None, profile.window_owner)

    reflex_thread = None
    if profile.reflexes:
        from cortex.agent.reflexes import Reflexes, ReflexThread

        reflex_exec = make_executor()
        reflexes = Reflexes(
            profile.reflexes, build_encoder(profile),
            press=lambda ks: reflex_exec.execute("hold", {"keys": list(ks), "seconds": 0.3}),
        )
        reflex_thread = ReflexThread(reflexes, cap.grab).start()

    executor = make_executor()
    agent = GameAgent(
        profile, goal, cap.grab, executor,
        should_stop=keys.stop.is_set,
        is_paused=lambda: keys.paused.is_set() or unfocused(),
        on_status=on_status,
    )
    log.info("playing %s with Claude; %s to stop, %s to pause", profile.name, profile.controls.kill_switch, profile.controls.pause)
    try:
        return agent.run()
    finally:
        executor.release_all()
        if reflex_thread:
            reflex_thread.stop()


def run_skill(
    profile: Profile,
    assignment: str,
    dry_run: bool = False,
    stop: threading.Event | None = None,
    paused: threading.Event | None = None,
    on_status=lambda s: None,
):
    """Play a skill you taught by showing (any game, offline). Returns a PlayResult."""
    from cortex.capture.screen import WindowCapture
    from cortex.control.input_mac import MacInput, frontmost_app_name
    from cortex.teach import choose_skill, list_skills, play_skill

    skills = list_skills(profile.name)
    if not skills:
        raise RuntimeError(
            f"Cortex hasn't learned anything for {profile.window_owner or profile.name} yet. "
            "Use Teach ▸ Record new skill… and play for a few minutes."
        )
    encoder = build_encoder(profile)
    skill = choose_skill(assignment, skills, encoder)
    cap = WindowCapture(profile.window_owner)
    backend = _NullInput() if dry_run else MacInput()
    keys = Hotkeys(profile.controls.kill_switch, profile.controls.pause, stop, paused)

    def to_screen(fx: float, fy: float) -> tuple[float, float]:
        w = cap.window
        return w.x + fx * w.width, w.y + fy * w.height

    def is_paused() -> bool:
        return keys.paused.is_set() or (
            profile.pause_when_unfocused and frontmost_app_name() not in (None, profile.window_owner)
        )

    log.info("playing skill %r (%.0fs recorded) in %s", skill.name, skill.seconds, profile.name)
    return play_skill(skill, cap.grab, encoder, backend, to_screen,
                      should_stop=keys.stop.is_set, is_paused=is_paused, on_status=on_status)


class _NullInput:
    def key(self, name, down):
        log.debug("key %s %s", name, "down" if down else "up")

    def mouse_move(self, x, y):
        pass

    def mouse_delta(self, dx, dy):
        log.debug("mouse delta %.0f,%.0f", dx, dy)

    def mouse_button(self, x, y, button, down):
        log.debug("mouse %s %s at %.0f,%.0f", button, "down" if down else "up", x, y)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Cortex: plays games for you")
    ap.add_argument("-a", "--assignment", help='what to do, in plain English, e.g. "harvest, then water the crops"')
    ap.add_argument("--offline", action="store_true", help="understand the assignment with keywords only (no Claude)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Claude model used to understand the assignment")
    ap.add_argument("-y", "--yes", action="store_true", help="start without confirming the plan")
    ap.add_argument("--profile", "--game", default="stardew", help="game profile name (see `python -m cortex.games list`) or a YAML path")
    ap.add_argument("--debug", action="store_true", help="show the perception overlay window")
    ap.add_argument("--dry-run", action="store_true", help="don't send any input")
    ap.add_argument("--record", type=Path, help="save every 10th frame here (for prompt tuning and tests)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    profile = load_profile(a.profile)

    if profile.engine == "skill":
        result = run_skill(profile, a.assignment or "", dry_run=a.dry_run, on_status=lambda s: log.info("%s", s))
        print(result.summary)
        return

    if profile.engine == "agent":
        goal = a.assignment
        if goal is None and sys.stdin.isatty():
            goal = input(f"What should I do in {profile.window_owner or profile.name}? > ")
        if not a.yes and sys.stdin.isatty():
            input("Press Enter, then switch to the game window within 3 seconds...")
            time.sleep(3)
        result = run_agent(profile, goal or "", dry_run=a.dry_run, on_status=lambda s: log.info("%s", s))
        print(f"{'Done' if result.success else 'Stopped'}: {result.summary}  ({result.steps} turns, ~${result.cost_usd:.2f})")
        return

    text = a.assignment
    if text is None and sys.stdin.isatty():
        text = input("What should I do? (press Enter for all chores) > ")
    mission = interpret(text or "", profile, "offline" if a.offline else "auto", a.model)
    print(mission.describe())
    if not mission.steps:
        print("I couldn't find anything I know how to do in that assignment.")
        sys.exit(1)
    if not a.yes and sys.stdin.isatty():
        if input("Start? Switch to the game window after pressing Enter. [Y/n] ").strip().lower() in ("n", "no"):
            return
        time.sleep(3)  # time to focus the game window

    run(profile, mission, debug=a.debug, dry_run=a.dry_run, record=a.record)


if __name__ == "__main__":
    main()
