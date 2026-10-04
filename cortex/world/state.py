"""What the bot believes about the game right now."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cortex.perception.tilegrid import SemanticMap


@dataclass
class WorldState:
    map: SemanticMap
    player: tuple[int, int]          # (row, col) of the tile the player stands on
    walkable: np.ndarray             # (R, C) bool
    energy: float = 1.0              # 0..1
    scene: str = "world"             # "world" | "dialog" | "menu" | ...
    timestamp: float = 0.0


def locate_player(smap: SemanticMap, frame_size: tuple[int, int]) -> tuple[int, int]:
    """Find the player's tile.

    The camera follows the player, so the player is near the screen centre
    except at map edges. Prefer the 'player'-labelled tile closest to the centre.
    The sprite is about two tiles tall, so we take the lowest tile of that column
    (the feet). If no tile is labelled 'player', fall back to the centre tile.
    """
    W, H = frame_size
    cr, cc = smap.tile_at_px(W / 2, H / 2)
    R, C = smap.shape
    cr, cc = min(max(cr, 0), R - 1), min(max(cc, 0), C - 1)
    if "player" not in smap.label_names:
        return cr, cc
    rows, cols = np.nonzero(smap.mask("player"))
    if len(rows) == 0:
        return cr, cc
    d = np.abs(rows - cr) + np.abs(cols - cc)
    near = d <= d.min() + 1
    best = np.argmin(d)
    col = cols[best]
    feet = rows[near & (cols == col)].max()
    if abs(feet - cr) + abs(col - cc) > 4:  # far from centre: probably a false positive
        return cr, cc
    return int(feet), int(col)
