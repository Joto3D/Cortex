"""Instant local reactions while Claude is thinking.

Each reflex compares one screen region against two descriptions with CLIP,
e.g. "a health bar with one heart left" vs "a health bar with many hearts".
When the first wins with probability >= threshold, it taps the reflex's keys.
There's no network round-trip, so it reacts in tens of milliseconds.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np

from cortex.config import ReflexSpec
from cortex.perception.clip_model import Encoder
from cortex.perception.hud import crop_frac
from cortex.perception.zeroshot import ZeroShotClassifier

log = logging.getLogger(__name__)


class Reflexes:
    def __init__(
        self,
        specs: tuple[ReflexSpec, ...],
        encoder: Encoder,
        press: Callable[[tuple[str, ...]], None],
    ):
        self.specs = specs
        self.press = press
        self._clf = [ZeroShotClassifier(encoder, {"yes": [s.prompt], "no": [s.otherwise]}) for s in specs]
        self._last_fired: dict[str, float] = {}

    def check(self, frame: np.ndarray, now: float) -> list[str]:
        """Classify each reflex's region; fire (and return) the ones that trigger."""
        fired = []
        for spec, clf in zip(self.specs, self._clf):
            if now - self._last_fired.get(spec.name, -1e9) < spec.cooldown_s:
                continue
            crop = crop_frac(frame, spec.roi)
            if crop.size == 0:
                continue
            emb = clf.encoder.encode_images(np.ascontiguousarray(crop)[None])
            z = clf.logits(emb)[0]
            p = np.exp(z - z.max())
            p /= p.sum()
            if p[0] >= spec.threshold:
                self._last_fired[spec.name] = now
                fired.append(spec.name)
                log.info("reflex %s (p=%.2f): pressing %s", spec.name, p[0], "+".join(spec.keys))
                self.press(spec.keys)
        return fired


class ReflexThread:
    """Runs ``Reflexes.check`` on fresh frames in the background until stopped."""

    def __init__(self, reflexes: Reflexes, grab: Callable[[], np.ndarray], hz: float = 10.0):
        self.reflexes = reflexes
        self.grab = grab
        self.period = 1.0 / hz
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="cortex-reflexes")

    def start(self) -> "ReflexThread":
        if self.reflexes.specs:
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self.reflexes.check(self.grab(), t0)
            except Exception as e:  # a reflex must never crash the bot
                log.warning("reflex check failed: %s", e)
            self._stop.wait(max(0.0, self.period - (time.monotonic() - t0)))
