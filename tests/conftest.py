"""Shared test fixtures: a fake colour-based "CLIP" plus synthetic farm frames."""
from __future__ import annotations

import zlib

import numpy as np
import pytest

from cortex.config import profile_from_dict

TILE = 16

# Each label has a distinct colour. The fake encoder embeds images by mean colour
# and prompts by looking up their colour, so zero-shot classification works the
# same way it would with real CLIP.
COLORS = {
    "grass": (60, 200, 60),
    "tilled_dry": (200, 150, 90),
    "crop_dry": (230, 230, 40),
    "crop_wet": (40, 120, 230),
    "crop_ripe": (230, 40, 40),
    "weed": (20, 90, 20),
    "stone": (120, 120, 120),
    "player": (240, 40, 240),
}
WALKABLE = {"grass", "tilled_dry", "crop_dry", "crop_wet", "crop_ripe", "player"}


def _embed_rgb(rgb) -> np.ndarray:
    v = np.asarray(rgb, np.float32) - 128.0
    v = np.concatenate([v, [40.0]])  # constant term so grey isn't a zero vector
    return v / np.linalg.norm(v)


class FakeEncoder:
    def __init__(self):
        self.image_calls = 0
        self.images_encoded = 0

    def encode_images(self, images: np.ndarray) -> np.ndarray:
        self.image_calls += 1
        self.images_encoded += len(images)
        means = images[..., :3].reshape(len(images), -1, 3).mean(axis=1)
        return np.stack([_embed_rgb(m) for m in means])

    def encode_text(self, texts):
        out = []
        for t in texts:
            name = t.split(":")[-1].strip()
            out.append(_embed_rgb(COLORS[name]) if name in COLORS else _embed_rgb((128, 128, 128)))
        return np.stack(out)


def make_profile(**overrides):
    raw = {
        "game": {"window_owner": "Test", "tile_px": TILE},
        "perception": {"prompt_template": "tile: {}", "scene_every_n_frames": 1},
        "labels": {n: {"walkable": n in WALKABLE, "prompts": [n]} for n in COLORS},
        "scenes": {"world": ["world"]},
        "tools": {"watering_can": "2", "scythe": "5", "pickaxe": "4"},
        "tasks": [
            {"name": "harvest", "targets": ["crop_ripe"], "action": "interact"},
            {"name": "water", "targets": ["crop_dry"], "action": "tool", "tool": "watering_can"},
            {"name": "weeds", "targets": ["weed"], "action": "tool", "tool": "scythe"},
        ],
        "controls": {"min_hold_s": 0.0, "jitter_ms": [0, 0]},
        "hud": {"min_energy": 0.1},
    }
    for k, v in overrides.items():
        raw[k] = v
    return profile_from_dict(raw, name="test")


def render_grid(names: list[list[str]], tile: int = TILE, phase=(0, 0), noise: int = 6) -> np.ndarray:
    """Render a frame where each tile is its label's colour plus a fixed per-label
    texture (so identical labels give identical pixels, like real tilesets)."""
    R, C = len(names), len(names[0])
    px, py = phase
    H, W = py + R * tile + 3, px + C * tile + 5  # ragged edges, like a real window
    frame = np.full((H, W, 3), 10, np.uint8)
    for r in range(R):
        for c in range(C):
            name = names[r][c]
            rng = np.random.default_rng(zlib.crc32(name.encode()))
            col = np.array(COLORS[name], np.int16)
            patch = col + rng.integers(-noise, noise + 1, size=(tile, tile, 3))
            frame[py + r * tile : py + (r + 1) * tile, px + c * tile : px + (c + 1) * tile] = np.clip(patch, 0, 255)
    return frame


@pytest.fixture
def profile():
    return make_profile()


@pytest.fixture
def encoder():
    return FakeEncoder()
