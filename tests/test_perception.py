import numpy as np
import pytest

from cortex.config import load_profile, profile_from_dict
from cortex.perception.hud import bar_fill
from cortex.perception.perceiver import Perceiver
from cortex.perception.tilegrid import TileGridPerceiver, estimate_grid_phase, slice_tiles
from cortex.perception.zeroshot import ZeroShotClassifier

from .conftest import COLORS, TILE, make_profile, render_grid

LAYOUT = [
    ["grass", "grass", "weed", "grass", "stone", "grass"],
    ["grass", "tilled_dry", "crop_dry", "crop_wet", "crop_ripe", "grass"],
    ["grass", "crop_dry", "player", "crop_ripe", "tilled_dry", "grass"],
    ["stone", "grass", "grass", "weed", "grass", "grass"],
]


def test_bundled_profile_loads():
    p = load_profile("stardew")
    assert p.tile_px > 0
    assert "crop_dry" in p.label_names
    assert p.tasks[0].name == "harvest"
    assert "water" not in p.walkable_labels


def test_profile_rejects_unknown_task_target(profile):
    raw = profile.raw | {"tasks": [{"name": "x", "targets": ["nope"], "action": "interact"}]}
    with pytest.raises(ValueError, match="unknown labels"):
        profile_from_dict(raw)


@pytest.mark.parametrize("phase", [(0, 0), (5, 11), (15, 1)])
def test_grid_phase_estimation(phase):
    frame = render_grid(LAYOUT, phase=phase)
    assert estimate_grid_phase(frame, TILE) == phase


def test_slice_tiles_is_whole_tiles_only():
    frame = render_grid(LAYOUT, phase=(5, 11))
    tiles, origin = slice_tiles(frame, TILE, (5, 11))
    assert origin == (5, 11)
    assert tiles.shape == (4, 6, TILE, TILE, 3)
    np.testing.assert_array_equal(tiles[1, 2], frame[11 + TILE : 11 + 2 * TILE, 5 + 2 * TILE : 5 + 3 * TILE])


def test_zero_shot_classifier(encoder):
    clf = ZeroShotClassifier(encoder, {n: [n] for n in COLORS})
    imgs = np.stack([np.full((8, 8, 3), COLORS[n], np.uint8) for n in COLORS])
    idx, p = clf.classify(imgs)
    assert [clf.labels[i] for i in idx] == list(COLORS)
    assert (p > 0.5).all()


def test_tile_perceiver_labels_and_cache(encoder):
    clf = ZeroShotClassifier(encoder, {n: [n] for n in COLORS})
    tp = TileGridPerceiver(clf, TILE)
    frame = render_grid(LAYOUT, phase=(3, 7))
    smap = tp.perceive(frame)
    got = [[smap.name_at(r, c) for c in range(smap.shape[1])] for r in range(smap.shape[0])]
    assert got == LAYOUT
    # Only one encode per *distinct* tile appearance.
    assert encoder.images_encoded == len({n for row in LAYOUT for n in row})

    # Shift the camera by a tile: every tile is still a cache hit.
    shifted = render_grid([row[1:] + row[:1] for row in LAYOUT], phase=(9, 2))
    before = encoder.images_encoded
    smap2 = tp.perceive(shifted)
    assert encoder.images_encoded == before
    assert smap2.name_at(0, 1) == "weed"


def test_semantic_map_geometry(encoder):
    clf = ZeroShotClassifier(encoder, {n: [n] for n in COLORS})
    smap = TileGridPerceiver(clf, TILE).perceive(render_grid(LAYOUT, phase=(4, 6)))
    x, y = smap.tile_center_px(2, 3)
    assert (x, y) == (4 + 3.5 * TILE, 6 + 2.5 * TILE)
    assert smap.tile_at_px(x, y) == (2, 3)
    assert smap.mask("crop_ripe").sum() == 2


def test_perceiver_builds_world_state(encoder, profile):
    frame = render_grid(LAYOUT)
    world = Perceiver(profile, encoder).perceive(frame)
    assert world.player == (2, 2)
    assert world.scene == "world"
    assert not world.walkable[0, 2]  # weed
    assert world.walkable[1, 2]      # crop
    assert set(world.map.label_names) == set(COLORS)


def test_bar_fill():
    frame = np.zeros((100, 20, 3), np.uint8)
    frame[40:, 5:15] = (40, 220, 40)  # bottom 60% filled with green
    assert bar_fill(frame, (0.25, 0.0, 0.75, 1.0)) == pytest.approx(0.6, abs=0.02)
    assert bar_fill(np.zeros((100, 20, 3), np.uint8), (0, 0, 1, 1)) == 0.0


def test_fixed_grid_phase_override(encoder):
    raw = make_profile().raw
    raw = raw | {"game": raw["game"] | {"grid_phase": [5, 11]}}
    world = Perceiver(profile_from_dict(raw), encoder).perceive(render_grid(LAYOUT, phase=(5, 11)))
    assert world.map.origin == (5, 11)
    assert world.map.name_at(2, 2) == "player"
