"""Latency benchmark for perception (and capture, on macOS).

    python -m cortex.bench --image recordings/frame_000000.png
    python -m cortex.bench --capture            # live window capture timing
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from cortex.config import load_profile


def _stats(xs: list[float]) -> str:
    a = np.array(xs) * 1e3
    return f"median {np.median(a):6.1f} ms   p95 {np.percentile(a, 95):6.1f} ms"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="stardew")
    ap.add_argument("--image", action="append", default=[], help="screenshot(s) to perceive")
    ap.add_argument("--capture", action="store_true", help="benchmark live window capture")
    ap.add_argument("-n", type=int, default=50)
    a = ap.parse_args(argv)
    profile = load_profile(a.profile)

    if a.capture:
        from cortex.capture.screen import WindowCapture

        cap = WindowCapture(profile.window_owner)
        ts = []
        for _ in range(a.n):
            t = time.perf_counter()
            cap.grab()
            ts.append(time.perf_counter() - t)
        print(f"capture       {_stats(ts)}")

    if a.image:
        from PIL import Image

        from cortex.loop import build_encoder
        from cortex.perception.perceiver import Perceiver

        enc = build_encoder(profile)
        frames = [np.asarray(Image.open(p).convert("RGB")) for p in a.image]
        perc = Perceiver(profile, enc)

        t = time.perf_counter()
        perc.perceive(frames[0])
        print(f"cold frame    {1e3 * (time.perf_counter() - t):6.1f} ms  (empty cache, every tile encoded)")

        ts = []
        for i in range(a.n):
            t = time.perf_counter()
            perc.perceive(frames[i % len(frames)])
            ts.append(time.perf_counter() - t)
        print(f"warm frame    {_stats(ts)}")
        st = perc.tiles.stats
        print(f"cache         {st['encoded']} tiles encoded of {st['tiles']} seen")

        perc.tiles.clear_cache()
        smap = perc.tiles.perceive(frames[0])
        names, counts = np.unique([smap.name_at(r, c) for r in range(smap.shape[0]) for c in range(smap.shape[1])], return_counts=True)
        print("labels        " + ", ".join(f"{n}={c}" for n, c in sorted(zip(names, counts), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
