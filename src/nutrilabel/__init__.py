"""nutrilabel - read nutrition facts tables from product photos with any OCR engine."""

from nutrilabel.extraction.extractor import NutritionExtractor
from nutrilabel.types import BBox, ExtractionResult, NutrientValue, OCRResult, Word

__all__ = [
    "BBox",
    "ExtractionResult",
    "NutrientValue",
    "NutritionExtractor",
    "NutritionReader",
    "OCRResult",
    "Word",
    "available_engines",
    "get_engine",
]
__version__ = "0.1.0"


def __getattr__(name: str):  # lazy imports keep `import nutrilabel` light
    if name == "NutritionReader":
        from nutrilabel.pipeline import NutritionReader

        return NutritionReader
    if name in ("get_engine", "available_engines"):
        from nutrilabel import engines

        return getattr(engines, name)
    raise AttributeError(name)
