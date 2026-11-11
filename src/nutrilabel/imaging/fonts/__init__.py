"""Bundled DejaVu Sans fonts (Bitstream Vera licence, see LICENSE-DejaVu.txt).

Shipping the font makes rendered test labels and visualisations identical on Linux,
macOS and Windows - system font fallbacks change glyph shapes and therefore OCR results.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

from PIL import ImageFont


@lru_cache(maxsize=64)
def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = resources.files("nutrilabel.imaging.fonts").joinpath(name)
    with resources.as_file(path) as p:
        return ImageFont.truetype(str(p), size)
