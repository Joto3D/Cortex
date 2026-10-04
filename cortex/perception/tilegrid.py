"""Turn a screen frame into a semantic tile map.

Farming sims are drawn on a fixed tile grid, so we:
  1. estimate where the grid lines fall on screen (the camera scrolls by pixels,
     so the grid phase changes while the player walks),
  2. slice the frame into whole tiles (a zero-copy reshape),
  3. look each tile up in a content-hash cache and send only *unseen* tiles to
     CLIP in one batch.

Pixel-art tiles repeat a lot (grass, soil, crops), so after the first second
of play nearly every tile is a cache hit and most frames encode few or no tiles.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .zeroshot import ZeroShotClassifier

UNKNOWN = -1


@dataclass
class SemanticMap:
    labels: np.ndarray          # (R, C) int, index into label_names or UNKNOWN
    probs: np.ndarray           # (R, C) float
    origin: tuple[int, int]     # pixel (x, y) of tile (0, 0)'s top-left corner
    tile_px: int
    label_names: list[str]

    @property
    def shape(self) -> tuple[int, int]:
        return self.labels.shape  # type: ignore[return-value]

    def index(self, name: str) -> int:
        return self.label_names.index(name)

    def mask(self, *names: str) -> np.ndarray:
        idx = [self.label_names.index(n) for n in names if n in self.label_names]
        return np.isin(self.labels, idx)

    def name_at(self, r: int, c: int) -> str:
        i = int(self.labels[r, c])
        return "unknown" if i == UNKNOWN else self.label_names[i]

    def tile_center_px(self, r: int, c: int) -> tuple[float, float]:
        x0, y0 = self.origin
        return x0 + (c + 0.5) * self.tile_px, y0 + (r + 0.5) * self.tile_px

    def tile_at_px(self, x: float, y: float) -> tuple[int, int]:
        x0, y0 = self.origin
        return int((y - y0) // self.tile_px), int((x - x0) // self.tile_px)

    def in_bounds(self, r: int, c: int) -> bool:
        R, C = self.shape
        return 0 <= r < R and 0 <= c < C


def estimate_grid_phase(frame: np.ndarray, tile_px: int, stride: int = 16) -> tuple[int, int]:
    """Estimate the (x, y) pixel offset of the tile grid.

    Neighbouring tiles usually differ, so strong colour edges pile up on tile
    boundaries. We sum absolute gradients along each axis, fold them modulo
    ``tile_px`` and take the strongest bin. Only every ``stride``-th row/column
    of the green channel is sampled, which keeps this to well under 1 ms at 1440p.
    """
    rows = frame[::stride, :, 1].astype(np.int16)
    gx = np.abs(np.diff(rows, axis=1)).sum(axis=0)       # edge strength between x and x+1
    cols = frame[:, ::stride, 1].astype(np.int16)
    gy = np.abs(np.diff(cols, axis=0)).sum(axis=1)
    return _fold_argmax(gx, tile_px), _fold_argmax(gy, tile_px)


def _fold_argmax(profile: np.ndarray, period: int) -> int:
    bins = np.bincount(np.arange(len(profile)) % period, weights=profile, minlength=period)
    # An edge between x and x+1 means a tile starts at x+1.
    return int((bins.argmax() + 1) % period)


def slice_tiles(frame: np.ndarray, tile_px: int, phase: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    """Return a (R, C, t, t, 3) view of all whole tiles plus the grid origin."""
    px, py = phase
    H, W = frame.shape[:2]
    R = (H - py) // tile_px
    C = (W - px) // tile_px
    region = frame[py : py + R * tile_px, px : px + C * tile_px, :3]
    tiles = region.reshape(R, tile_px, C, tile_px, 3).swapaxes(1, 2)
    return tiles, (px, py)


class TileGridPerceiver:
    """Classify every tile on screen using a CLIP zero-shot classifier and a hash cache."""

    def __init__(
        self,
        classifier: ZeroShotClassifier,
        tile_px: int,
        cache_size: int = 50_000,
        min_confidence: float = 0.0,
        hash_samples: int = 16,
    ):
        self.classifier = classifier
        self.tile_px = tile_px
        self.cache_size = cache_size
        self.min_confidence = min_confidence
        # Sample every `stride` pixels for the cache key. For pixel art at integer
        # zoom this lands on (nearly) one sample per game pixel.
        self.stride = max(1, tile_px // hash_samples)
        self._cache: OrderedDict[int, tuple[int, float]] = OrderedDict()
        self.stats = {"tiles": 0, "encoded": 0}

    def perceive(self, frame: np.ndarray, phase: tuple[int, int] | None = None) -> SemanticMap:
        if phase is None:
            phase = estimate_grid_phase(frame, self.tile_px)
        tiles, origin = slice_tiles(frame, self.tile_px, phase)
        R, C = tiles.shape[:2]
        s = self.stride
        # Only the small key samples are copied; tile pixels are copied for cache misses only.
        keys = np.ascontiguousarray(tiles[:, :, ::s, ::s]).reshape(R * C, -1)

        labels = np.empty(R * C, np.int32)
        probs = np.empty(R * C, np.float32)
        misses: dict[int, list[int]] = {}
        for i in range(R * C):
            k = hash(keys[i].tobytes())
            hit = self._cache.get(k)
            if hit is None:
                misses.setdefault(k, []).append(i)
            else:
                self._cache.move_to_end(k)
                labels[i], probs[i] = hit

        if misses:
            first = np.array([idx[0] for idx in misses.values()])
            batch = tiles[first // C, first % C]
            li, lp = self.classifier.classify(batch)
            for (k, idxs), l, p in zip(misses.items(), li, lp):
                val = (int(l) if p >= self.min_confidence else UNKNOWN, float(p))
                self._cache[k] = val
                for i in idxs:
                    labels[i], probs[i] = val
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

        self.stats["tiles"] += R * C
        self.stats["encoded"] += len(misses)
        return SemanticMap(
            labels=labels.reshape(R, C),
            probs=probs.reshape(R, C),
            origin=origin,
            tile_px=self.tile_px,
            label_names=self.classifier.labels,
        )

    def clear_cache(self) -> None:
        self._cache.clear()
