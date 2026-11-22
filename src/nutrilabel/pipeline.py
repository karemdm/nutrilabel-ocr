"""High-level API: image in, nutrient values out."""

from __future__ import annotations

from nutrilabel.engines import OCREngine, get_engine
from nutrilabel.extraction.extractor import NutritionExtractor
from nutrilabel.imaging.preprocess import ImageInput, prepare, rotate
from nutrilabel.types import BBox, ExtractionResult, OCRResult, Word


class NutritionReader:
    """Read the nutrition table of a product photo.

    >>> reader = NutritionReader(engine="rapidocr")            # doctest: +SKIP
    >>> result = reader.read("label.jpg")                      # doctest: +SKIP
    >>> result.as_simple_dict()                                # doctest: +SKIP
    {'energy_kcal_100g': 449.0, 'fat_100g': 25.0, ...}

    Parameters
    ----------
    engine:
        Engine name (``"auto"``, any name listed by ``nutrilabel engines``, for example
        ``"rapidocr"``, ``"tesseract"``, ``"olmocr"`` or ``"vlm"``) or an
        :class:`OCREngine` instance.
    languages:
        Language codes passed to the engine (default: ``en, fr, de, es, it, nl, pt``).
        Ignored when ``engine`` is an instance.
    extractor:
        A custom :class:`NutritionExtractor`; defaults to one with its default settings.
    min_fields:
        Minimum number of nutrients below which the rotation retry is tried.
    max_side:
        Longest image side fed to the OCR engine, in pixels.
    auto_rotate:
        When fewer than ``min_fields`` values are found, retry with the photo rotated
        by 90/270/180 degrees and keep the best reading.
    """

    def __init__(
        self,
        engine: str | OCREngine = "auto",
        *,
        languages: list[str] | None = None,
        extractor: NutritionExtractor | None = None,
        max_side: int = 1600,
        auto_rotate: bool = True,
        min_fields: int = 2,
        **engine_kwargs,
    ) -> None:
        if isinstance(engine, OCREngine):
            self.engine = engine
        else:
            if languages is not None:
                engine_kwargs["languages"] = languages
            self.engine = get_engine(engine, **engine_kwargs)
        self.extractor = extractor or NutritionExtractor()
        self.max_side = max_side
        self.auto_rotate = auto_rotate
        self.min_fields = min_fields

    def ocr(self, image: ImageInput) -> OCRResult:
        """Run OCR only; boxes are returned in original-image coordinates."""
        array, scale = prepare(image, max_side=self.max_side)
        return _rescale(self.engine.read(array), scale)

    def read(self, image: ImageInput) -> ExtractionResult:
        """OCR the image and extract nutrients; boxes are in original-image pixels."""
        array, scale = prepare(image, max_side=self.max_side)
        ocr = self.engine.read(array)
        result = self.extractor.extract(ocr)
        # VLM engines read rotated text themselves and are slow: no rotation retries.
        text_only = getattr(self.engine, "boxes_are_synthetic", False)
        if self.auto_rotate and not text_only and len(result.nutrients) < self.min_fields:
            for degrees in (90, 270, 180):
                rotated = rotate(array, degrees)
                r_ocr = self.engine.read(rotated)
                candidate = self.extractor.extract(r_ocr)
                if len(candidate.nutrients) > len(result.nutrients):
                    r_ocr.elapsed_s += ocr.elapsed_s
                    candidate.ocr = _unrotate(r_ocr, degrees, array.shape[1], array.shape[0])
                    for nv in candidate.nutrients.values():
                        if nv.bbox is not None:
                            nv.bbox = _unrotate_box(nv.bbox, degrees, array.shape[1], array.shape[0])
                    candidate.warnings.append(f"image rotated by {degrees} degrees")
                    result = candidate
                if len(result.nutrients) >= self.min_fields:
                    break
        result.ocr = _rescale(result.ocr, scale) if result.ocr else None
        if scale != 1.0:
            for nv in result.nutrients.values():
                if nv.bbox is not None:
                    nv.bbox = nv.bbox.scale(1 / scale, 1 / scale)
        return result


def _rescale(ocr: OCRResult, scale: float) -> OCRResult:
    if scale == 1.0:
        return ocr
    inv = 1.0 / scale
    return OCRResult(
        words=[Word(w.text, w.bbox.scale(inv, inv), w.confidence) for w in ocr.words],
        engine=ocr.engine,
        width=round(ocr.width * inv),
        height=round(ocr.height * inv),
        elapsed_s=ocr.elapsed_s,
    )


def _unrotate_box(b: BBox, degrees: int, w: int, h: int) -> BBox:
    """Map a box from an image rotated CCW by ``degrees`` back to the original (w x h)."""
    if degrees == 90:  # rotated image is h x w; original (x, y) -> (y, w - x)
        return BBox(w - b.y1, b.x0, w - b.y0, b.x1)
    if degrees == 270:  # original (x, y) -> (h - y, x)
        return BBox(b.y0, h - b.x1, b.y1, h - b.x0)
    if degrees == 180:
        return BBox(w - b.x1, h - b.y1, w - b.x0, h - b.y0)
    return b


def _unrotate(ocr: OCRResult, degrees: int, w: int, h: int) -> OCRResult:
    return OCRResult(
        words=[Word(x.text, _unrotate_box(x.bbox, degrees, w, h), x.confidence) for x in ocr.words],
        engine=ocr.engine,
        width=w,
        height=h,
        elapsed_s=ocr.elapsed_s,
    )
