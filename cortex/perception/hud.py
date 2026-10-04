"""Cheap HUD readers that don't need a neural network."""
from __future__ import annotations

import numpy as np


def crop_frac(frame: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    """Crop a region given as fractions of the frame (x0, y0, x1, y1)."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = roi
    return frame[int(y0 * H) : int(y1 * H), int(x0 * W) : int(x1 * W), :3]


def bar_fill(frame: np.ndarray, roi: tuple[float, float, float, float], min_sat: int = 60, min_val: int = 80) -> float:
    """Estimate how full a vertical, bottom-anchored, coloured bar is (0..1).

    A row counts as "filled" when most of its pixels are strongly coloured. An
    empty bar is dark or grey, while the fill is green, yellow or red.
    """
    crop = crop_frac(frame, roi).astype(np.int16)
    if crop.size == 0:
        return 1.0
    mx = crop.max(axis=2)
    mn = crop.min(axis=2)
    coloured = ((mx - mn) >= min_sat) & (mx >= min_val)
    filled_rows = coloured.mean(axis=1) > 0.3
    if not filled_rows.any():
        return 0.0
    # Measure from the topmost filled row down to the bottom of the bar.
    top = int(np.argmax(filled_rows))
    return float(1.0 - top / len(filled_rows))
