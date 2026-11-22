"""Common interface for OCR engines."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from nutrilabel.types import OCRResult, Word

# Languages found on European / American food labels. Adapters translate this list to
# each engine's own codes and keep only what the engine supports.
DEFAULT_LANGUAGES: tuple[str, ...] = ("en", "fr", "de", "es", "it", "nl", "pt")


class EngineNotAvailable(RuntimeError):  # noqa: N818 - public API name
    """Raised when an engine's Python package (or binary) is not installed."""


class OCREngine(ABC):
    """Wraps one OCR library and returns :class:`~nutrilabel.types.Word` objects.

    Subclasses implement :meth:`_run` and declare ``name`` and ``pip_extra``.
    Heavy models are loaded lazily on first use and then reused.
    """

    name: ClassVar[str] = "base"
    pip_extra: ClassVar[str] = ""

    def __init__(self, languages: tuple[str, ...] | list[str] | None = None) -> None:
        self.languages = tuple(languages or DEFAULT_LANGUAGES)

    @classmethod
    def is_available(cls) -> bool:
        """True when the engine's library (and binary, if any) can be imported."""
        try:
            cls._check_import()
        except Exception:  # noqa: BLE001 - any import failure means "not available"
            return False
        return True

    @classmethod
    @abstractmethod
    def _check_import(cls) -> None:
        """Import the backing library (raise if missing)."""

    @abstractmethod
    def _run(self, image: np.ndarray) -> list[Word]:
        """Run OCR on an RGB ``uint8`` image of shape ``(H, W, 3)``."""

    def read(self, image: np.ndarray) -> OCRResult:
        """Run the engine on an RGB (or grayscale) array; empty words are dropped."""
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        start = time.perf_counter()
        words = self._run(np.ascontiguousarray(image))
        elapsed = time.perf_counter() - start
        words = [w for w in words if w.text.strip()]
        return OCRResult(
            words=words,
            engine=self.name,
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            elapsed_s=elapsed,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(languages={list(self.languages)})"
