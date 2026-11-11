"""OCR engine registry.

>>> from nutrilabel.engines import get_engine, available_engines
>>> available_engines()            # doctest: +SKIP
['rapidocr', 'tesseract']
>>> engine = get_engine("auto")    # doctest: +SKIP
"""

from __future__ import annotations

from nutrilabel.engines.base import DEFAULT_LANGUAGES, EngineNotAvailable, OCREngine
from nutrilabel.engines.classic import (
    DocTREngine,
    EasyOCREngine,
    PaddleOCREngine,
    RapidOCREngine,
    SuryaEngine,
    TesseractEngine,
)
from nutrilabel.engines.vlm import (
    HF_VLM_ENGINES,
    HF_VLM_PRESETS,
    HFVLMEngine,
    PaddleOCRVLEngine,
    VLMEngine,
)

ENGINES: dict[str, type[OCREngine]] = {
    cls.name: cls
    for cls in (PaddleOCREngine, RapidOCREngine, DocTREngine, EasyOCREngine, TesseractEngine,
                SuryaEngine, PaddleOCRVLEngine, *HF_VLM_ENGINES, HFVLMEngine, VLMEngine)
}

#: Classic OCR engines (word/line boxes, CPU friendly). ``engine="auto"`` picks from
#: these, in this order - keep it in sync with the latest benchmark
#: (``python -m benchmark.report`` prints the recommended order).
PREFERENCE: list[str] = ["paddleocr", "rapidocr", "surya", "doctr", "easyocr", "tesseract"]

#: Vision-language OCR models (text output, GPU recommended). Never chosen by "auto"
#: because a first call downloads several GB of weights.
VLM_ENGINES: list[str] = ["paddleocr-vl", *HF_VLM_PRESETS, "vlm"]


def available_engines(include_vlm: bool = False) -> list[str]:
    """Names of the engines whose libraries are installed, in preference order."""
    names = PREFERENCE + (VLM_ENGINES if include_vlm else [])
    return [name for name in names if ENGINES[name].is_available()]


def get_engine(name: str = "auto", **kwargs) -> OCREngine:
    """Instantiate an engine by name (``"auto"`` picks the best installed one)."""
    key = name.lower().strip()
    if key == "auto":
        installed = available_engines()
        if not installed:
            raise EngineNotAvailable(
                "No OCR engine installed. Try: pip install 'nutrilabel-ocr[rapidocr]'"
            )
        key = installed[0]
    aliases = {"paddle": "paddleocr", "rapid": "rapidocr", "easy": "easyocr",
               "tess": "tesseract", "paddle-vl": "paddleocr-vl", "paddleocrvl": "paddleocr-vl"}
    key = aliases.get(key, key)
    if key not in ENGINES:
        raise ValueError(f"Unknown engine {name!r}. Choose from: auto, {', '.join(ENGINES)}")
    cls = ENGINES[key]
    if not cls.is_available() and not (key == "vlm" and kwargs.get("model")):
        hint = (f"pip install \"nutrilabel-ocr[{cls.pip_extra}]\"" if cls.pip_extra
                else "set NUTRILABEL_VLM_MODEL or pass model=...")
        raise EngineNotAvailable(f"Engine {key!r} is not available. Try: {hint}")
    return cls(**kwargs)


__all__ = [
    "DEFAULT_LANGUAGES",
    "ENGINES",
    "PREFERENCE",
    "VLM_ENGINES",
    "EngineNotAvailable",
    "DocTREngine",
    "EasyOCREngine",
    "HFVLMEngine",
    "PaddleOCREngine",
    "PaddleOCRVLEngine",
    "RapidOCREngine",
    "SuryaEngine",
    "TesseractEngine",
    "VLMEngine",
    "OCREngine",
    "available_engines",
    "get_engine",
]
