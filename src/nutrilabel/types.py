"""Core data structures shared by OCR engines, the extractor and the benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in pixel (or any consistent) coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def union(self, other: BBox) -> BBox:
        return BBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def scale(self, sx: float, sy: float) -> BBox:
        return BBox(self.x0 * sx, self.y0 * sy, self.x1 * sx, self.y1 * sy)

    @classmethod
    def from_points(cls, points: Any) -> BBox:
        """Build a box from a polygon ``[[x, y], ...]`` (as returned by most OCR engines)."""
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        return cls(min(xs), min(ys), max(xs), max(ys))

    def as_list(self) -> list[float]:
        return [round(self.x0, 1), round(self.y0, 1), round(self.x1, 1), round(self.y1, 1)]


@dataclass(frozen=True)
class Word:
    """A piece of recognised text with its location.

    Engines may return single words or whole text lines; the extractor splits
    multi-word boxes itself, so adapters do not need to.
    """

    text: str
    bbox: BBox
    confidence: float = 1.0


@dataclass
class OCRResult:
    """Output of an OCR engine for one image."""

    words: list[Word]
    engine: str
    width: int = 0
    height: int = 0
    elapsed_s: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "width": self.width,
            "height": self.height,
            "elapsed_s": round(self.elapsed_s, 4),
            "words": [
                {"text": w.text, "bbox": w.bbox.as_list(), "confidence": round(w.confidence, 4)}
                for w in self.words
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OCRResult:
        return cls(
            words=[
                Word(w["text"], BBox(*w["bbox"]), float(w.get("confidence", 1.0)))
                for w in data["words"]
            ],
            engine=data.get("engine", "unknown"),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            elapsed_s=float(data.get("elapsed_s", 0.0)),
        )


@dataclass
class NutrientValue:
    """One extracted nutrient value, e.g. ``fat_100g = 3.2 g``."""

    key: str
    value: float
    unit: str | None
    raw: str
    bbox: BBox | None = None
    confidence: float = 1.0
    qualifier: str = ""  # "<", ">", "~" when the label says e.g. "< 0.5 g"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bbox"] = self.bbox.as_list() if self.bbox else None
        return d


@dataclass
class ExtractionResult:
    """All nutrient values found on a label."""

    nutrients: dict[str, NutrientValue] = field(default_factory=dict)
    ocr: OCRResult | None = None
    warnings: list[str] = field(default_factory=list)

    def as_simple_dict(self) -> dict[str, float]:
        """``{"fat_100g": 3.2, ...}`` - handy for quick inspection and tests."""
        return {k: v.value for k, v in sorted(self.nutrients.items())}

    def to_dict(self, include_ocr: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "nutrients": {k: v.to_dict() for k, v in sorted(self.nutrients.items())},
            "warnings": list(self.warnings),
        }
        if self.ocr is not None:
            out["engine"] = self.ocr.engine
            out["ocr_elapsed_s"] = round(self.ocr.elapsed_s, 4)
            if include_ocr:
                out["ocr"] = self.ocr.to_dict()
        return out
