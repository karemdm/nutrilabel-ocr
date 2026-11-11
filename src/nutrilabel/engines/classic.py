"""Adapters for classic OCR engines (text detection + recognition, word/line boxes).

Every adapter converts the library's native output into a list of
:class:`~nutrilabel.types.Word`. Libraries are imported lazily so that installing
one engine is enough to use the package.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from nutrilabel.engines.base import EngineNotAvailable, OCREngine
from nutrilabel.types import BBox, Word

# ---------------------------------------------------------------------------- Tesseract

_TESSERACT_LANGS = {"en": "eng", "fr": "fra", "de": "deu", "es": "spa", "it": "ita",
                    "nl": "nld", "pt": "por"}


class TesseractEngine(OCREngine):
    """Tesseract 4/5 (LSTM) through ``pytesseract``.

    ``psm=11`` ("sparse text") works best on nutrition tables, whose cells are
    short, widely spaced tokens; see ``docs/engines.md``.
    """

    name: ClassVar[str] = "tesseract"
    pip_extra: ClassVar[str] = "tesseract"

    def __init__(self, languages=None, psm: int = 11, oem: int = 1) -> None:
        super().__init__(languages)
        self.psm = psm
        self.oem = oem
        self._lang: str | None = None

    @classmethod
    def _check_import(cls) -> None:
        import pytesseract

        pytesseract.get_tesseract_version()

    def _lang_string(self) -> str:
        if self._lang is None:
            import pytesseract

            try:
                installed = set(pytesseract.get_languages(config=""))
            except Exception:  # noqa: BLE001
                installed = {"eng"}
            wanted = [_TESSERACT_LANGS.get(lang, lang) for lang in self.languages]
            chosen = [code for code in wanted if code in installed] or ["eng"]
            self._lang = "+".join(dict.fromkeys(chosen))
        return self._lang

    def _run(self, image: np.ndarray) -> list[Word]:
        try:
            import pytesseract
        except ImportError as e:  # pragma: no cover - guarded by is_available
            raise EngineNotAvailable("pip install 'nutrilabel-ocr[tesseract]'") from e
        data = pytesseract.image_to_data(
            image,
            lang=self._lang_string(),
            config=f"--oem {self.oem} --psm {self.psm}",
            output_type=pytesseract.Output.DICT,
        )
        return tesseract_data_to_words(data)


def tesseract_data_to_words(data: dict[str, list[Any]]) -> list[Word]:
    words = []
    for text, conf, x, y, w, h in zip(
        data["text"], data["conf"], data["left"], data["top"], data["width"], data["height"]
    ):
        text = str(text).strip()
        try:
            c = float(conf)
        except (TypeError, ValueError):
            c = -1.0
        if not text or c < 0:
            continue
        words.append(Word(text, BBox(float(x), float(y), float(x + w), float(y + h)), c / 100.0))
    return words


# ---------------------------------------------------------------------------- EasyOCR


class EasyOCREngine(OCREngine):
    """EasyOCR (CRAFT detector + CRNN recogniser). All Latin languages share one model."""

    name: ClassVar[str] = "easyocr"
    pip_extra: ClassVar[str] = "easyocr"

    def __init__(self, languages=None, gpu: bool = False) -> None:
        super().__init__(languages)
        self.gpu = gpu
        self._reader = None

    @classmethod
    def _check_import(cls) -> None:
        import easyocr  # noqa: F401

    def _run(self, image: np.ndarray) -> list[Word]:
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(list(self.languages), gpu=self.gpu, verbose=False)
        result = self._reader.readtext(image, detail=1, paragraph=False)
        return easyocr_to_words(result)


def easyocr_to_words(result: list[Any]) -> list[Word]:
    return [Word(str(text), BBox.from_points(pts), float(conf)) for pts, text, conf in result]


# ---------------------------------------------------------------------------- PaddleOCR


class PaddleOCREngine(OCREngine):
    """PaddleOCR (PP-OCR). Supports both the 2.x (``.ocr``) and 3.x (``.predict``) APIs.

    ``lang="latin"`` loads the multilingual Latin-script recogniser (accents included);
    ``lang="en"`` is faster but drops accents.
    """

    name: ClassVar[str] = "paddleocr"
    pip_extra: ClassVar[str] = "paddle"

    def __init__(self, languages=None, lang: str = "latin", **paddle_kwargs: Any) -> None:
        super().__init__(languages)
        self.lang = lang
        self.paddle_kwargs = paddle_kwargs
        self._ocr = None
        self._v3 = False
        # oneDNN (MKL-DNN) crashes some PaddlePaddle 3.x CPU builds on Windows.
        import sys

        self._no_mkldnn = sys.platform == "win32" and "enable_mkldnn" not in paddle_kwargs

    @classmethod
    def _check_import(cls) -> None:
        import paddleocr  # noqa: F401

    def _load(self) -> None:
        import paddleocr
        from paddleocr import PaddleOCR

        major = int(str(getattr(paddleocr, "__version__", "2")).split(".")[0] or 2)
        self._v3 = major >= 3
        if self._v3:
            kwargs = {
                "lang": self.lang,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": True,
            }
            if self._no_mkldnn:
                kwargs["enable_mkldnn"] = False
        else:
            kwargs = {"lang": self.lang, "use_angle_cls": True, "show_log": False}
        kwargs.update(self.paddle_kwargs)
        # Model availability per language differs between PaddleOCR releases (e.g. some
        # 3.x versions ship no PP-OCRv5 "latin" model). Fall back gracefully.
        attempts = [kwargs]
        if self._v3 and not any(k.startswith("text_detection") for k in self.paddle_kwargs):
            # The default "server" detector is ~10x slower on CPU than the mobile one,
            # with little accuracy gain on short label text.
            attempts.insert(0, {**kwargs, "text_detection_model_name": "PP-OCRv5_mobile_det"})
        if self._v3 and "ocr_version" not in self.paddle_kwargs:
            attempts.append({**kwargs, "ocr_version": "PP-OCRv3"})
        attempts.append({**kwargs, "lang": "en"})
        error: Exception | None = None
        for attempt in attempts:
            try:
                self._ocr = PaddleOCR(**attempt)
                self.lang = attempt["lang"]
                return
            except (ValueError, TypeError) as e:  # "No models are available for lang=..."
                error = e
        raise EngineNotAvailable(f"PaddleOCR could not load a model: {error}") from error

    def _run(self, image: np.ndarray) -> list[Word]:
        if self._ocr is None:
            self._load()
        bgr = image[:, :, ::-1].copy()  # PaddleOCR follows the OpenCV (BGR) convention
        try:
            return self._predict(bgr)
        except NotImplementedError as e:
            # Known PaddlePaddle 3.x CPU bug (oneDNN + PIR, frequent on Windows):
            # "ConvertPirAttribute2RuntimeAttribute not support ...". Retry without oneDNN.
            if self._no_mkldnn or not self._v3:
                raise
            self._no_mkldnn = True
            self._load()
            try:
                return self._predict(bgr)
            except NotImplementedError:
                raise e from None

    def _predict(self, bgr: np.ndarray) -> list[Word]:
        if self._v3:
            return paddle_v3_to_words(self._ocr.predict(bgr))
        return paddle_v2_to_words(self._ocr.ocr(bgr, cls=True))


def paddle_v2_to_words(result: list[Any] | None) -> list[Word]:
    words: list[Word] = []
    for page in result or []:
        for item in page or []:
            pts, (text, score) = item[0], item[1]
            words.append(Word(str(text), BBox.from_points(pts), float(score)))
    return words


def paddle_v3_to_words(result: list[Any] | None) -> list[Word]:
    words: list[Word] = []
    for page in result or []:
        get = page.get if hasattr(page, "get") else lambda k, d=None, p=page: getattr(p, k, d)
        texts = get("rec_texts", []) or []
        scores = get("rec_scores", []) or []
        polys = get("rec_polys", None)
        if polys is None or len(polys) == 0:
            polys = get("dt_polys", None)
        boxes = get("rec_boxes", None)
        for i, text in enumerate(texts):
            if polys is not None and i < len(polys):
                bbox = BBox.from_points(np.asarray(polys[i]).reshape(-1, 2))
            elif boxes is not None and i < len(boxes):
                bbox = BBox(*map(float, boxes[i][:4]))
            else:
                continue
            score = float(scores[i]) if i < len(scores) else 1.0
            words.append(Word(str(text), bbox, score))
    return words


# ---------------------------------------------------------------------------- RapidOCR


class RapidOCREngine(OCREngine):
    """RapidOCR: PP-OCR models exported to ONNX - no Paddle install, fast on CPU.

    Works with both ``rapidocr`` (>=2) and the older ``rapidocr_onnxruntime`` package.
    """

    name: ClassVar[str] = "rapidocr"
    pip_extra: ClassVar[str] = "rapidocr"

    def __init__(self, languages=None, **rapid_kwargs: Any) -> None:
        super().__init__(languages)
        self.rapid_kwargs = rapid_kwargs
        self._engine = None

    @classmethod
    def _check_import(cls) -> None:
        try:
            import rapidocr  # noqa: F401
        except ImportError:
            import rapidocr_onnxruntime  # noqa: F401

    def _run(self, image: np.ndarray) -> list[Word]:
        if self._engine is None:
            try:
                from rapidocr import RapidOCR
            except ImportError:
                from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR(**self.rapid_kwargs)
        bgr = image[:, :, ::-1].copy()
        return rapidocr_to_words(self._engine(bgr))


def rapidocr_to_words(output: Any) -> list[Word]:
    words: list[Word] = []
    # rapidocr >= 2: an object with .boxes / .txts / .scores
    if hasattr(output, "txts"):
        boxes = output.boxes if output.boxes is not None else []
        txts = output.txts or ()
        scores = output.scores or ()
        for pts, text, score in zip(boxes, txts, scores):
            words.append(Word(str(text), BBox.from_points(pts), float(score)))
        return words
    # rapidocr_onnxruntime: (result, elapse) with result = [[pts, text, score], ...] | None
    result = output[0] if isinstance(output, tuple) else output
    for pts, text, score in result or []:
        words.append(Word(str(text), BBox.from_points(pts), float(score)))
    return words


# ---------------------------------------------------------------------------- docTR


class DocTREngine(OCREngine):
    """docTR (Mindee): DBNet/FAST detector + CRNN/PARSeq recogniser, word-level output."""

    name: ClassVar[str] = "doctr"
    pip_extra: ClassVar[str] = "doctr"

    def __init__(self, languages=None, det_arch: str = "fast_base",
                 reco_arch: str = "crnn_vgg16_bn") -> None:
        super().__init__(languages)
        self.det_arch = det_arch
        self.reco_arch = reco_arch
        self._model = None

    @classmethod
    def _check_import(cls) -> None:
        import doctr  # noqa: F401

    def _run(self, image: np.ndarray) -> list[Word]:
        if self._model is None:
            from doctr.models import ocr_predictor

            self._model = ocr_predictor(
                det_arch=self.det_arch, reco_arch=self.reco_arch, pretrained=True
            )
        doc = self._model([image])
        return doctr_export_to_words(doc.export(), image.shape[1], image.shape[0])


def doctr_export_to_words(export: dict[str, Any], width: int, height: int) -> list[Word]:
    words: list[Word] = []
    for page in export.get("pages", []):
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                for w in line.get("words", []):
                    geom = w["geometry"]
                    if len(geom) == 2:  # ((xmin, ymin), (xmax, ymax)), relative
                        (x0, y0), (x1, y1) = geom
                        bbox = BBox(x0 * width, y0 * height, x1 * width, y1 * height)
                    else:  # rotated polygon
                        pts = [(x * width, y * height) for x, y in geom]
                        bbox = BBox.from_points(pts)
                    words.append(Word(str(w["value"]), bbox, float(w.get("confidence", 1.0))))
    return words


# ---------------------------------------------------------------------------- Surya


class SuryaEngine(OCREngine):
    """Surya OCR (Datalab): transformer detector + recogniser, 90+ languages, line boxes.

    Code GPL-3.0, weights under a modified OpenRAIL licence (free for research and
    personal use). Supports the ``surya-ocr`` >= 0.14 predictor API and the older
    ``run_ocr`` API.
    """

    name: ClassVar[str] = "surya"
    pip_extra: ClassVar[str] = "surya"

    def __init__(self, languages=None) -> None:
        super().__init__(languages)
        self._rec = None
        self._det = None
        self._legacy = False

    @classmethod
    def _check_import(cls) -> None:
        import surya  # noqa: F401

    def _load(self) -> None:
        try:
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor

            try:
                from surya.foundation import FoundationPredictor

                self._rec = RecognitionPredictor(FoundationPredictor())
            except ImportError:
                self._rec = RecognitionPredictor()
            self._det = DetectionPredictor()
        except ImportError:  # surya < 0.10
            self._legacy = True

    def _run(self, image: np.ndarray) -> list[Word]:
        from PIL import Image

        if self._rec is None and not self._legacy:
            self._load()
        img = Image.fromarray(image)
        if self._legacy:
            from surya.model.detection.model import load_model as load_det
            from surya.model.detection.model import load_processor as load_det_proc
            from surya.model.recognition.model import load_model as load_rec
            from surya.model.recognition.processor import load_processor as load_rec_proc
            from surya.ocr import run_ocr

            preds = run_ocr([img], [list(self.languages)], load_det(), load_det_proc(),
                            load_rec(), load_rec_proc())
        else:
            try:
                preds = self._rec([img], det_predictor=self._det)
            except TypeError:  # 0.10-0.13 also wanted languages
                preds = self._rec([img], [list(self.languages)], self._det)
        return surya_to_words(preds)


def surya_to_words(preds: list[Any]) -> list[Word]:
    words: list[Word] = []
    for page in preds or []:
        for line in getattr(page, "text_lines", []) or []:
            text = getattr(line, "text", "")
            bbox = getattr(line, "bbox", None)
            if bbox is None and getattr(line, "polygon", None) is not None:
                box = BBox.from_points(line.polygon)
            elif bbox is not None:
                box = BBox(*map(float, bbox[:4]))
            else:
                continue
            words.append(Word(str(text), box, float(getattr(line, "confidence", 1.0) or 1.0)))
    return words
