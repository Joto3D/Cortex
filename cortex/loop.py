"""Main loop: capture → perceive → plan → act, at a fixed rate.

    python -m cortex.loop --profile stardew --debug
    python -m cortex.loop --dry-run      # perceive and plan, but send no input

F12 stops the bot and F11 toggles pause (both configurable in the profile).
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path

from cortex.config import Profile, load_profile
from cortex.planner.actions import Done, Wait
from cortex.planner.farm_planner import FarmPlanner

log = logging.getLogger("cortex")


class Hotkeys:
    """Global kill-switch and pause hotkeys (pynput, needs Input Monitoring permission)."""

    def __init__(self, kill: str, pause: str):
        self.stop = threading.Event()
        self.paused = threading.Event()
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
    fb = (p["fallback_model"], p["fallback_pretrained"]) if "fallback_model" in p else None
    return ClipEncoder(p.get("model", "MobileCLIP-S1"), p.get("pretrained", "datacompdr"), fallback=fb)


def run(profile: Profile, debug: bool = False, dry_run: bool = False, record: Path | None = None) -> str:
    from cortex.capture.screen import WindowCapture
    from cortex.control.controller import Controller
    from cortex.control.input_mac import MacInput, frontmost_app_name
    from cortex.perception.perceiver import Perceiver

    cap = WindowCapture(profile.window_owner)
    perceiver = Perceiver(profile, build_encoder(profile))
    planner = FarmPlanner(profile)
    backend = _NullInput() if dry_run else MacInput()
    ctl = Controller(profile, backend, to_screen=cap.to_screen)
    keys = Hotkeys(profile.controls.kill_switch, profile.controls.pause)
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
                    f"energy {world.energy:.0%}  {world.scene}  {type(action).__name__}"
                )
                from cortex.overlay.debug import show

                if not show(frame, world, planner.last_plan, status):
                    break
            if isinstance(action, Done):
                reason = action.reason
                break
            frame_no += 1
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        ctl.release_all()
    log.info("stopped: %s", reason)
    return reason


class _NullInput:
    def key(self, name, down):
        log.debug("key %s %s", name, "down" if down else "up")

    def mouse_move(self, x, y):
        pass

    def mouse_button(self, x, y, button, down):
        log.debug("mouse %s %s at %.0f,%.0f", button, "down" if down else "up", x, y)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Cortex farming bot")
    ap.add_argument("--profile", default="stardew", help="bundled profile name or path to a YAML file")
    ap.add_argument("--debug", action="store_true", help="show the perception overlay window")
    ap.add_argument("--dry-run", action="store_true", help="don't send any input")
    ap.add_argument("--record", type=Path, help="save every 10th frame here (for prompt tuning and tests)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(load_profile(a.profile), debug=a.debug, dry_run=a.dry_run, record=a.record)


if __name__ == "__main__":
    main()
