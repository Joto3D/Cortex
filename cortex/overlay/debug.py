"""Debug overlay that draws the semantic map, the player and the current plan.

``render`` is pure numpy, so it can be tested and saved. ``show`` needs OpenCV.
"""
from __future__ import annotations

import colorsys
import zlib

import numpy as np

from cortex.planner.farm_planner import Plan
from cortex.world.state import WorldState


def label_color(name: str) -> tuple[int, int, int]:
    h = (zlib.crc32(name.encode()) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def render(frame: np.ndarray, world: WorldState, plan: Plan | None = None, alpha: float = 0.35) -> np.ndarray:
    """Return an RGB copy of ``frame`` with tile labels tinted in and the path drawn."""
    out = np.ascontiguousarray(frame[..., :3]).copy()
    smap = world.map
    t = smap.tile_px
    x0, y0 = smap.origin
    R, C = smap.shape
    colors = np.array([label_color(n) for n in smap.label_names] + [(0, 0, 0)], np.float32)
    tint = colors[smap.labels]  # UNKNOWN (-1) indexes the trailing black entry
    tint_px = np.repeat(np.repeat(tint, t, axis=0), t, axis=1)
    region = out[y0 : y0 + R * t, x0 : x0 + C * t].astype(np.float32)
    out[y0 : y0 + R * t, x0 : x0 + C * t] = (region * (1 - alpha) + tint_px * alpha).astype(np.uint8)

    def box(r: int, c: int, color, w: int = 3) -> None:
        ys, xs = y0 + r * t, x0 + c * t
        out[ys : ys + w, xs : xs + t] = color
        out[ys + t - w : ys + t, xs : xs + t] = color
        out[ys : ys + t, xs : xs + w] = color
        out[ys : ys + t, xs + t - w : xs + t] = color

    if plan is not None:
        for r, c in plan.path:
            box(r, c, (255, 255, 255), 2)
        box(*plan.target, (255, 0, 0), 5)
    box(*world.player, (0, 255, 255), 4)
    return out


def show(frame: np.ndarray, world: WorldState, plan: Plan | None, status: str, max_width: int = 1280) -> bool:
    """Display the overlay. Returns False if the user pressed q in the window."""
    import cv2

    img = render(frame, world, plan)
    if img.shape[1] > max_width:
        f = max_width / img.shape[1]
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("cortex debug", bgr)
    return (cv2.waitKey(1) & 0xFF) != ord("q")
