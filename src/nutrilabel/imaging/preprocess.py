"""Image loading and light preprocessing shared by all engines."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps

ImageInput = Union[str, Path, bytes, Image.Image, np.ndarray]


def load_image(image: ImageInput) -> Image.Image:
    """Load an image from a path, raw bytes, a PIL image or an array, as RGB.

    EXIF orientation is applied, so phone photos come out the right way up.
    """
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype(np.uint8))
    elif isinstance(image, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image))
    else:
        img = Image.open(Path(image))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def prepare(image: ImageInput, max_side: int = 1600, min_side: int = 400) -> tuple[np.ndarray, float]:
    """Return an RGB ``uint8`` array resized for OCR, and the applied scale factor.

    Large photos are shrunk (faster, and detectors are trained on ~1-2k px images);
    tiny crops are enlarged so that small digits stay readable.
    """
    img = load_image(image)
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
    elif min(w, h) < min_side:
        scale = 2.0  # integer factor: keeps glyph shapes crisp for the recogniser
    if scale != 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return np.asarray(img), scale


def rotate(array: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate counter-clockwise by a multiple of 90 degrees."""
    return np.ascontiguousarray(np.rot90(array, k=(degrees // 90) % 4))
