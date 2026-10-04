"""Draw the Cortex app icon (a sprout on a green tile) and build Cortex.icns.

    python packaging/make_icon.py      # -> packaging/build/Cortex.icns (needs macOS iconutil)
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).parent / "build"
GREEN, LEAF = (47, 107, 58, 255), (233, 245, 208, 255)


def _leaf(s: float, angle: float) -> Image.Image:
    leaf = Image.new("RGBA", (round(12 * s), round(12 * s)), (0, 0, 0, 0))
    ImageDraw.Draw(leaf).ellipse([1 * s, 3.6 * s, 11 * s, 8.4 * s], fill=LEAF)
    return leaf.rotate(angle, resample=Image.BICUBIC)


def draw(size: int = 1024) -> Image.Image:
    s = size / 32  # 32-unit grid, like the website favicon
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([1 * s, 1 * s, 31 * s, 31 * s], radius=7 * s, fill=GREEN)
    d.rounded_rectangle([14.8 * s, 13 * s, 17.2 * s, 26 * s], radius=1.2 * s, fill=LEAF)  # stem
    left, right = _leaf(s, -35), _leaf(s, 35)
    img.alpha_composite(left, (round(5.5 * s), round(8.5 * s)))     # leaf tilted up-left
    img.alpha_composite(right, (round(14.5 * s), round(6.5 * s)))   # leaf tilted up-right
    return img


def main() -> None:
    iconset = OUT / "Cortex.iconset"
    iconset.mkdir(parents=True, exist_ok=True)
    big = draw(1024)
    big.save(OUT / "icon_1024.png")
    for px in (16, 32, 128, 256, 512):
        big.resize((px, px), Image.LANCZOS).save(iconset / f"icon_{px}x{px}.png")
        big.resize((px * 2, px * 2), Image.LANCZOS).save(iconset / f"icon_{px}x{px}@2x.png")
    if shutil.which("iconutil"):
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(OUT / "Cortex.icns")], check=True)
        print(f"wrote {OUT / 'Cortex.icns'}")
    else:
        print(f"wrote {iconset} (iconutil not found; .icns is only built on macOS)")


if __name__ == "__main__":
    main()
