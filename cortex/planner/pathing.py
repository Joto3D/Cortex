"""Grid search on the semantic map."""
from __future__ import annotations

from collections import deque

import numpy as np

# Order matters for tie-breaking: orthogonal neighbours first.
NEIGHBOURS_8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
STEPS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def dilate8(mask: np.ndarray, include_self: bool = True) -> np.ndarray:
    """3x3 binary dilation without scipy.

    With ``include_self=False`` a tile is set only if one of its 8 *neighbours*
    is set ("touches another marked tile").
    """
    out = mask.copy() if include_self else np.zeros_like(mask)
    R, C = mask.shape
    for dr, dc in NEIGHBOURS_8:
        # out[r + dr, c + dc] |= mask[r, c]
        src = mask[max(0, -dr) : R - max(0, dr), max(0, -dc) : C - max(0, dc)]
        out[max(0, dr) : R - max(0, -dr), max(0, dc) : C - max(0, -dc)] |= src
    return out


def bfs_to_any(walkable: np.ndarray, start: tuple[int, int], goal: np.ndarray) -> list[tuple[int, int]] | None:
    """Shortest 4-connected path from ``start`` to the nearest ``goal`` tile.

    The start tile is always treated as walkable, because the player is standing
    on it. Returns the path including start and goal, or None.
    """
    R, C = walkable.shape
    sr, sc = start
    if not (0 <= sr < R and 0 <= sc < C):
        return None
    if goal[sr, sc]:
        return [start]
    parent = np.full((R, C, 2), -1, np.int32)
    seen = np.zeros((R, C), bool)
    seen[sr, sc] = True
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in STEPS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and not seen[nr, nc] and walkable[nr, nc]:
                seen[nr, nc] = True
                parent[nr, nc] = (r, c)
                if goal[nr, nc]:
                    path = [(nr, nc)]
                    while path[-1] != start:
                        pr, pc = parent[path[-1]]
                        path.append((int(pr), int(pc)))
                    return path[::-1]
                q.append((nr, nc))
    return None


def direction(a: tuple[int, int], b: tuple[int, int]) -> str:
    d = (b[0] - a[0], b[1] - a[1])
    for name, step in STEPS.items():
        if step == d:
            return name
    raise ValueError(f"{a} -> {b} is not a single orthogonal step")


def adjacent_target(stand: tuple[int, int], targets: np.ndarray) -> tuple[int, int] | None:
    """Pick a target tile next to ``stand`` (orthogonal neighbours preferred)."""
    R, C = targets.shape
    for dr, dc in NEIGHBOURS_8:
        r, c = stand[0] + dr, stand[1] + dc
        if 0 <= r < R and 0 <= c < C and targets[r, c]:
            return r, c
    return None
