"""Turns a frame into a WorldState. This is all of the bot's perception."""
from __future__ import annotations

import time

import numpy as np

from cortex.config import Profile
from cortex.world.state import WorldState, locate_player

from .clip_model import Encoder
from .hud import bar_fill
from .tilegrid import TileGridPerceiver
from .zeroshot import ZeroShotClassifier


class Perceiver:
    def __init__(self, profile: Profile, encoder: Encoder):
        p = profile.perception
        template = p.get("prompt_template", "{}")
        self.profile = profile
        tile_clf = ZeroShotClassifier(encoder, {l.name: l.prompts for l in profile.labels}, template)
        self.tiles = TileGridPerceiver(
            tile_clf,
            tile_px=profile.tile_px,
            cache_size=int(p.get("cache_size", 50_000)),
            min_confidence=float(p.get("min_confidence", 0.0)),
        )
        self.scene_clf = ZeroShotClassifier(encoder, profile.scenes) if profile.scenes else None
        self.scene_every = max(1, int(p.get("scene_every_n_frames", 10)))
        self._frame_no = 0
        self._scene = "world"
        walkable = profile.walkable_labels
        self._walkable_idx = np.array([n in walkable for n in tile_clf.labels], bool)
        self.timings: dict[str, float] = {}

    def perceive(self, frame: np.ndarray) -> WorldState:
        t0 = time.perf_counter()
        if self.scene_clf is not None and self._frame_no % self.scene_every == 0:
            H, W = frame.shape[:2]
            step = max(1, min(H, W) // 256)
            idx, _ = self.scene_clf.classify(np.ascontiguousarray(frame[::step, ::step, :3])[None])
            self._scene = self.scene_clf.labels[int(idx[0])]
        self._frame_no += 1
        t1 = time.perf_counter()

        smap = self.tiles.perceive(frame)
        t2 = time.perf_counter()

        known = smap.labels >= 0
        walkable = np.zeros(smap.shape, bool)
        walkable[known] = self._walkable_idx[smap.labels[known]]
        H, W = frame.shape[:2]
        player = locate_player(smap, (W, H))
        energy = bar_fill(frame, self.profile.energy_roi) if self.profile.energy_roi else 1.0
        t3 = time.perf_counter()

        self.timings = {"scene_ms": (t1 - t0) * 1e3, "tiles_ms": (t2 - t1) * 1e3, "state_ms": (t3 - t2) * 1e3}
        return WorldState(map=smap, player=player, walkable=walkable, energy=energy, scene=self._scene, timestamp=t3)
