"""Unit tests for the OCR engine adapters, registry and the OCREngine base class.

Only mocked raw engine outputs are used: no OCR library is imported here.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from nutrilabel.engines import ENGINES, PREFERENCE, VLM_ENGINES, available_engines, get_engine
from nutrilabel.engines.base import EngineNotAvailable, OCREngine
from nutrilabel.engines.classic import (
    TesseractEngine,
    UltralyticsEngine,
    doctr_export_to_words,
    easyocr_to_words,
    paddle_v2_to_words,
    paddle_v3_to_words,
    rapidocr_to_words,
    tesseract_data_to_words,
    ultralytics_to_words,
)
from nutrilabel.types import BBox, Word

QUAD = [[10, 20], [50, 20], [50, 40], [10, 40]]
QUAD_BBOX = BBox(10.0, 20.0, 50.0, 40.0)


# ------------------------------------------------------------------ Tesseract


def test_tesseract_basic_conversion():
    """Confidence is scaled to 0-1 and bbox is (l, t, l+w, t+h)."""
    data = {"text": ["Fat"], "conf": [96], "left": [10], "top": [20], "width": [40], "height": [15]}
    words = tesseract_data_to_words(data)
    assert words == [Word("Fat", BBox(10.0, 20.0, 50.0, 35.0), 0.96)]


def test_tesseract_skips_empty_text_and_negative_conf():
    """Blank/whitespace text and conf -1 (int, float or string) are dropped."""
    data = {
        "text": ["", "  ", "ok", "bad", "bad2", "bad3"],
        "conf": [90, 90, 80, -1, "-1", "-1.0"],
        "left": [0] * 6,
        "top": [0] * 6,
        "width": [5] * 6,
        "height": [5] * 6,
    }
    words = tesseract_data_to_words(data)
    assert [w.text for w in words] == ["ok"]
    assert words[0].confidence == pytest.approx(0.8)


def test_tesseract_string_conf_and_strip():
    """String confidences are parsed and the text is stripped."""
    data = {"text": [" Salt "], "conf": ["55.5"], "left": [1], "top": [2], "width": [3], "height": [4]}
    (w,) = tesseract_data_to_words(data)
    assert w.text == "Salt"
    assert w.confidence == pytest.approx(0.555)
    assert w.bbox == BBox(1.0, 2.0, 4.0, 6.0)


def test_tesseract_unparseable_conf_is_skipped():
    """A non-numeric confidence is treated like -1."""
    data = {"text": ["x"], "conf": ["n/a"], "left": [0], "top": [0], "width": [1], "height": [1]}
    assert tesseract_data_to_words(data) == []


# ------------------------------------------------------------------ EasyOCR


def test_easyocr_to_words():
    """EasyOCR tuples (polygon, text, conf) become Words with axis-aligned boxes."""
    result = [(QUAD, "Protein", 0.91), ([[0, 0], [5, 1], [5, 9], [0, 8]], "g", 0.5)]
    words = easyocr_to_words(result)
    assert words[0] == Word("Protein", QUAD_BBOX, 0.91)
    assert words[1].bbox == BBox(0.0, 0.0, 5.0, 9.0)
    assert easyocr_to_words([]) == []


# ------------------------------------------------------------------ PaddleOCR 2.x


def test_paddle_v2_to_words():
    """Paddle 2.x nests results as [page][item] = [pts, (text, score)]."""
    result = [[[QUAD, ("Energy", 0.97)], [[[0, 0], [4, 0], [4, 4], [0, 4]], ("kcal", 0.8)]]]
    words = paddle_v2_to_words(result)
    assert words[0] == Word("Energy", QUAD_BBOX, 0.97)
    assert words[1].text == "kcal"
    assert words[1].bbox == BBox(0.0, 0.0, 4.0, 4.0)


def test_paddle_v2_empty_pages():
    """None results and None pages yield no words."""
    assert paddle_v2_to_words(None) == []
    assert paddle_v2_to_words([None]) == []
    assert paddle_v2_to_words([[]]) == []


# ------------------------------------------------------------------ PaddleOCR 3.x


def _poly(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def test_paddle_v3_dict_page_with_rec_polys():
    """rec_polys (numpy (4,2) arrays) are the primary geometry source."""
    page = {
        "rec_texts": ["Fat", "Sugar"],
        "rec_scores": [0.9, 0.8],
        "rec_polys": [_poly(1, 2, 11, 12), _poly(20, 30, 40, 50)],
    }
    words = paddle_v3_to_words([page])
    assert words == [
        Word("Fat", BBox(1.0, 2.0, 11.0, 12.0), pytest.approx(0.9)),
        Word("Sugar", BBox(20.0, 30.0, 40.0, 50.0), pytest.approx(0.8)),
    ]


def test_paddle_v3_empty_rec_polys_falls_back_to_dt_polys():
    """An empty rec_polys falls back to dt_polys before looking at rec_boxes."""
    page = {
        "rec_texts": ["Fat"],
        "rec_scores": [0.9],
        "rec_polys": np.zeros((0, 4, 2)),
        "dt_polys": [_poly(5, 6, 7, 8)],
        "rec_boxes": np.array([[100, 100, 200, 200]]),
    }
    (w,) = paddle_v3_to_words([page])
    assert w.bbox == BBox(5.0, 6.0, 7.0, 8.0)


def test_paddle_v3_rec_boxes_fallback():
    """With no polygons at all, rec_boxes ([x0, y0, x1, y1]) are used."""
    page = {
        "rec_texts": ["Salt"],
        "rec_scores": [0.7],
        "rec_polys": [],
        "rec_boxes": np.array([[3, 4, 30, 40]]),
    }
    (w,) = paddle_v3_to_words([page])
    assert w.bbox == BBox(3.0, 4.0, 30.0, 40.0)
    assert w.confidence == pytest.approx(0.7)


def test_paddle_v3_skips_text_without_geometry_and_defaults_score():
    """Texts beyond the available geometry are skipped; a missing score is 1.0."""
    page = {"rec_texts": ["a", "b"], "rec_polys": [_poly(0, 0, 1, 1)]}
    words = paddle_v3_to_words([page])
    assert [w.text for w in words] == ["a"]
    assert words[0].confidence == 1.0


def test_paddle_v3_object_page_and_empty_inputs():
    """Attribute-style pages work like dict pages; None/empty input gives []."""
    page = SimpleNamespace(rec_texts=["Fibre"], rec_scores=[0.6], rec_polys=[_poly(0, 0, 9, 9)])
    (w,) = paddle_v3_to_words([page])
    assert w == Word("Fibre", BBox(0.0, 0.0, 9.0, 9.0), pytest.approx(0.6))
    assert paddle_v3_to_words(None) == []
    assert paddle_v3_to_words([]) == []
    assert paddle_v3_to_words([{}]) == []


# ------------------------------------------------------------------ RapidOCR


def test_rapidocr_new_api():
    """rapidocr >= 2 returns an object with boxes/txts/scores."""
    out = SimpleNamespace(
        boxes=np.array([_poly(1, 2, 3, 4), _poly(10, 10, 20, 30)]),
        txts=("Carbs", "12 g"),
        scores=(0.95, 0.85),
    )
    words = rapidocr_to_words(out)
    assert [w.text for w in words] == ["Carbs", "12 g"]
    assert words[0].bbox == BBox(1.0, 2.0, 3.0, 4.0)
    assert words[1].confidence == pytest.approx(0.85)


def test_rapidocr_new_api_nothing_found():
    """New API with boxes/txts/scores all None gives []."""
    assert rapidocr_to_words(SimpleNamespace(boxes=None, txts=None, scores=None)) == []


def test_rapidocr_legacy_tuple():
    """rapidocr_onnxruntime returns (result_list, elapse)."""
    out = ([[QUAD, "Fat", 0.9], [QUAD, "Salt", 0.8]], [0.1, 0.2, 0.3])
    words = rapidocr_to_words(out)
    assert [w.text for w in words] == ["Fat", "Salt"]
    assert words[0].bbox == QUAD_BBOX


def test_rapidocr_legacy_none():
    """Legacy (None, elapse) means no text found."""
    assert rapidocr_to_words((None, [0.1])) == []


# ------------------------------------------------------------------ docTR


def _export(*words):
    return {"pages": [{"blocks": [{"lines": [{"words": list(words)}]}]}]}


def test_doctr_relative_box_scaled_by_size():
    """Relative ((x0,y0),(x1,y1)) geometry is scaled by width and height."""
    export = _export({"value": "Fat", "confidence": 0.9, "geometry": ((0.1, 0.2), (0.5, 0.4))})
    (w,) = doctr_export_to_words(export, 200, 100)
    assert w.text == "Fat"
    assert w.bbox == BBox(pytest.approx(20.0), pytest.approx(20.0), pytest.approx(100.0), pytest.approx(40.0))
    assert w.confidence == pytest.approx(0.9)


def test_doctr_polygon_geometry_and_default_confidence():
    """A 4-point polygon is scaled then reduced to its enclosing box."""
    geom = ((0.1, 0.1), (0.5, 0.15), (0.5, 0.35), (0.1, 0.3))
    (w,) = doctr_export_to_words(_export({"value": "Salt", "geometry": geom}), 100, 200)
    assert w.bbox.x0 == pytest.approx(10.0)
    assert w.bbox.y0 == pytest.approx(20.0)
    assert w.bbox.x1 == pytest.approx(50.0)
    assert w.bbox.y1 == pytest.approx(70.0)
    assert w.confidence == 1.0


def test_doctr_empty_export():
    """Missing or empty pages give no words."""
    assert doctr_export_to_words({}, 10, 10) == []
    assert doctr_export_to_words({"pages": [{"blocks": []}]}, 10, 10) == []


# ------------------------------------------------------------------ Ultralytics


def _ult(text, line, poly, conf=0.9):
    return {"text": text, "confidence": conf, "detection_index": 0, "line": line,
            "polygon": poly, "detection_confidence": 0.8}


def test_ultralytics_polygon_becomes_enclosing_box():
    """A rotated 4-point polygon is reduced to its axis-aligned box; confidence is the reading's."""
    poly = [[12, 22], [52, 18], [50, 40], [10, 42]]
    (w,) = ultralytics_to_words([_ult("Protein", 0, poly, 0.77)])
    assert w.text == "Protein"
    assert w.bbox == BBox(10.0, 18.0, 52.0, 42.0)
    assert w.confidence == pytest.approx(0.77)


def test_ultralytics_keeps_reading_order_by_line():
    """Words come out ordered by line, and by input order within a line."""
    items = [_ult("c", 1, QUAD), _ult("a", 0, QUAD), _ult("b", 0, QUAD)]
    assert [w.text for w in ultralytics_to_words(items)] == ["a", "b", "c"]


def test_ultralytics_skips_blank_and_empty():
    """None/empty results and entries without text or polygon give no words."""
    assert ultralytics_to_words(None) == []
    assert ultralytics_to_words([]) == []
    assert ultralytics_to_words([_ult("", 0, QUAD), _ult("x", 0, [])]) == []


def _mocked_engine(texts, **kwargs):
    engine = UltralyticsEngine(**kwargs)
    engine._ocr = mock.Mock(return_value=SimpleNamespace(texts=texts))
    return engine


def test_ultralytics_engine_read_with_mocked_solution():
    """The engine passes BGR pixels to the solution and returns words with image size."""
    engine = _mocked_engine([_ult("Salt", 0, QUAD)])
    image = np.zeros((60, 80, 3), np.uint8)
    image[..., 0] = 255  # pure red in RGB
    res = engine.read(image)
    assert [w.text for w in res.words] == ["Salt"]
    assert (res.width, res.height, res.engine) == (80, 60, "ultralytics")
    sent = engine._ocr.call_args.args[0]
    assert sent[0, 0].tolist() == [0, 0, 255]  # BGR


def test_ultralytics_engine_empty_result():
    assert _mocked_engine([]).read(np.zeros((10, 10, 3), np.uint8)).words == []


def test_ultralytics_options_and_solution_kwargs():
    """Size selects the weights, imgsz/recognizer/device are forwarded, display and logging are off."""
    fake = mock.Mock()
    with mock.patch.dict("sys.modules", {"ultralytics": mock.Mock(),
                                         "ultralytics.solutions": SimpleNamespace(OCR=fake)}):
        engine = UltralyticsEngine(size="n", imgsz=640, recognizer="korean_PP-OCRv5_mobile_rec")
        engine._load()
    fake.assert_called_once_with(model="yolo26n-text.pt", imgsz=640, show=False, verbose=False,
                                 recognizer="korean_PP-OCRv5_mobile_rec")
    assert UltralyticsEngine().model == "yolo26x-text.pt"
    assert UltralyticsEngine(model="my.pt").model == "my.pt"
    with pytest.raises(ValueError):
        UltralyticsEngine(size="xl")


def test_ultralytics_variant_name_and_unavailable():
    """'ultralytics:n' selects the size; a missing package raises EngineNotAvailable."""
    with mock.patch.object(UltralyticsEngine, "is_available", return_value=True):
        engine = get_engine("Ultralytics:s")
    assert isinstance(engine, UltralyticsEngine) and engine.model == "yolo26s-text.pt"
    with mock.patch.object(UltralyticsEngine, "_check_import", side_effect=ImportError):
        assert UltralyticsEngine.is_available() is False
        with pytest.raises(EngineNotAvailable, match="ultralytics"):
            get_engine("ultralytics")


# ------------------------------------------------------------------ Registry


def test_get_engine_unknown_raises_value_error():
    """Unknown engine names are rejected."""
    with pytest.raises(ValueError):
        get_engine("nope")


@pytest.mark.skipif(not TesseractEngine.is_available(), reason="tesseract not available")
def test_get_engine_alias_and_case():
    """'tess' and 'Tesseract ' resolve to the Tesseract engine."""
    assert isinstance(get_engine("tess"), TesseractEngine)
    assert isinstance(get_engine(" Tesseract "), TesseractEngine)


def test_available_engines_subset_of_preference_in_order():
    """available_engines() keeps PREFERENCE order and only lists known engines."""
    avail = available_engines()
    assert set(avail) <= set(PREFERENCE)
    assert avail == [n for n in PREFERENCE if n in avail]
    # every registered engine is either a classic one (PREFERENCE) or a VLM one
    assert set(PREFERENCE) | set(VLM_ENGINES) | {"hf-vlm"} == set(ENGINES)
    assert not set(PREFERENCE) & set(VLM_ENGINES)


def test_get_engine_unavailable_raises():
    """An engine whose is_available() is False raises EngineNotAvailable."""
    with mock.patch.object(TesseractEngine, "is_available", classmethod(lambda cls: False)):
        with pytest.raises(EngineNotAvailable):
            get_engine("tesseract")


def test_get_engine_auto_without_engines_raises():
    """'auto' with nothing installed raises EngineNotAvailable."""
    with mock.patch("nutrilabel.engines.available_engines", return_value=[]):
        with pytest.raises(EngineNotAvailable):
            get_engine("auto")


# ------------------------------------------------------------------ OCREngine.read


class _FakeEngine(OCREngine):
    name = "fake"

    def __init__(self, words):
        super().__init__()
        self.words = words
        self.seen_shape = None

    @classmethod
    def _check_import(cls):
        return None

    def _run(self, image):
        self.seen_shape = image.shape
        return self.words


def _w(text):
    return Word(text, BBox(0, 0, 1, 1), 0.9)


def test_read_converts_grayscale_to_three_channels():
    """A (H, W) image reaches _run as (H, W, 3)."""
    eng = _FakeEngine([_w("a")])
    eng.read(np.zeros((7, 11), dtype=np.uint8))
    assert eng.seen_shape == (7, 11, 3)


def test_read_keeps_rgb_shape():
    """An (H, W, 3) image is passed through unchanged."""
    eng = _FakeEngine([])
    eng.read(np.zeros((5, 6, 3), dtype=np.uint8))
    assert eng.seen_shape == (5, 6, 3)


def test_read_filters_whitespace_and_fills_metadata():
    """Whitespace-only words are dropped; size, engine name and timing are set."""
    eng = _FakeEngine([_w("Fat"), _w("   "), _w(""), _w("Salt")])
    res = eng.read(np.zeros((30, 40), dtype=np.uint8))
    assert [w.text for w in res.words] == ["Fat", "Salt"]
    assert (res.width, res.height) == (40, 30)
    assert res.engine == "fake"
    assert res.elapsed_s >= 0.0
