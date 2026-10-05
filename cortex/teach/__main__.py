"""Teach Cortex from the terminal.

    python -m cortex.teach record --game my_game "collect wood"   # play; Ctrl+C (or --seconds) to finish
    python -m cortex.teach list --game my_game
    python -m cortex.teach play --game my_game "get some wood"
    python -m cortex.teach delete --game my_game "collect wood"
"""
from __future__ import annotations

import argparse
import logging
import threading
import time

from cortex.config import load_profile

from .skill import list_skills, slug


def record(game: str, name: str, seconds: float | None) -> None:
    from cortex.capture.screen import WindowCapture
    from cortex.control.input_mac import frontmost_app_name
    from cortex.loop import build_encoder

    from .recorder import InputLog, Recorder, start_listeners

    profile = load_profile(game)
    cap = WindowCapture(profile.window_owner)

    def to_fraction(x: float, y: float) -> tuple[float, float]:
        w = cap.window
        return (x - w.x) / max(w.width, 1), (y - w.y) / max(w.height, 1)

    log_ = InputLog(ignore={profile.controls.kill_switch, profile.controls.pause})
    rec = Recorder(cap.grab, lambda: frontmost_app_name() == profile.window_owner, log_)
    stop_listeners = start_listeners(log_, to_fraction, profile.controls.kill_switch, rec.stop)
    print(f"Recording “{name}” in 3 s. Switch to {profile.window_owner} and play. "
          f"Press {profile.controls.kill_switch.upper()} (or Ctrl+C here) to finish.")
    time.sleep(3)
    if seconds:
        threading.Timer(seconds, rec.stop).start()
    try:
        rec.run()
    except KeyboardInterrupt:
        pass
    finally:
        stop_listeners()
    print(f"Recorded {rec.seconds:.0f} s. Learning…")
    existing = next((s for s in list_skills(profile.name) if slug(s.name) == slug(name)), None)
    skill = rec.to_skill(name, profile.name, build_encoder(profile), existing)
    path = skill.save()
    print(f"Learned “{name}”: {skill.seconds:.0f} s of examples saved to {path}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Teach Cortex a game by showing it.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("record", "play", "delete"):
        p = sub.add_parser(cmd)
        p.add_argument("--game", required=True)
        p.add_argument("skill", nargs="?" if cmd == "play" else None, default="")
        if cmd == "record":
            p.add_argument("--seconds", type=float, help="stop automatically after this long")
        if cmd == "play":
            p.add_argument("--dry-run", action="store_true")
    ls = sub.add_parser("list")
    ls.add_argument("--game", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    game = load_profile(a.game).name

    if a.cmd == "record":
        record(a.game, a.skill, a.seconds)
    elif a.cmd == "list":
        skills = list_skills(game)
        if not skills:
            print("No skills yet. Record one with: python -m cortex.teach record --game", game, '"what you do"')
        for s in skills:
            print(f"{s.name:30s} {s.seconds:6.0f} s  ({int(s.segments.max()) + 1} recordings)")
    elif a.cmd == "play":
        from cortex.loop import run_skill

        res = run_skill(load_profile(a.game), a.skill, dry_run=a.dry_run, on_status=print)
        print(res.summary)
    elif a.cmd == "delete":
        match = [s for s in list_skills(game) if slug(s.name) == slug(a.skill)]
        for s in match:
            s.delete()
        print(f"Deleted {len(match)} skill(s).")


if __name__ == "__main__":
    main()
