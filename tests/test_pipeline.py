"""End-to-end tests on rendered labels: image -> OCR -> nutrients.

They need the Tesseract binary (``apt install tesseract-ocr``) and are skipped otherwise.
"""

import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

from nutrilabel import NutritionReader
from nutrilabel.cli import main as cli_main
from nutrilabel.engines import TesseractEngine
from nutrilabel.evaluation.synthetic import render_label
from nutrilabel.imaging.preprocess import prepare, rotate
from nutrilabel.imaging.viz import draw_result

HAS_TESSERACT = TesseractEngine.is_available()
needs_tesseract = pytest.mark.skipif(not HAS_TESSERACT, reason="tesseract not installed")


def _accuracy(truth, got):
    ok = sum(1 for k, v in truth.items() if k in got and abs(got[k] - v) < 0.011)
    return ok / len(truth)


def test_prepare_resizes_and_reports_scale():
    big = np.zeros((3000, 2000, 3), dtype=np.uint8)
    arr, scale = prepare(big, max_side=1500)
    assert arr.shape[:2] == (1500, 1000) and scale == 0.5
    small = np.zeros((100, 300, 3), dtype=np.uint8)
    arr, scale = prepare(small)
    assert arr.shape[:2] == (200, 600) and scale == 2.0


def test_rotate_is_counter_clockwise():
    a = np.arange(6, dtype=np.uint8).reshape(2, 3)
    assert rotate(a, 90).shape == (3, 2)
    assert rotate(a, 90)[0, 0] == a[0, 2]


@needs_tesseract
def test_end_to_end_on_synthetic_labels():
    reader = NutritionReader(engine="tesseract")
    scores = []
    for lang in ("en", "fr", "de", "es"):
        label = render_label(lang=lang)
        result = reader.read(label.image)
        scores.append(_accuracy(label.truth, result.as_simple_dict()))
        assert not (set(result.nutrients) - set(label.truth)), "no invented fields"
    assert min(scores) >= 0.75, scores


@needs_tesseract
def test_auto_rotate_recovers_sideways_photo():
    label = render_label(lang="en", serving_g=None)
    sideways = label.image.rotate(90, expand=True)
    reader = NutritionReader(engine="tesseract")
    result = reader.read(sideways)
    assert _accuracy(label.truth, result.as_simple_dict()) >= 0.7
    # boxes are mapped back onto the sideways photo
    w, h = sideways.size
    for nv in result.nutrients.values():
        assert 0 <= nv.bbox.x0 <= w and 0 <= nv.bbox.y1 <= h


@needs_tesseract
def test_visualisation_and_cli():
    label = render_label(lang="en")
    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "label.png"
        label.image.save(img_path)
        result = NutritionReader(engine="tesseract").read(img_path)
        annotated = draw_result(img_path, result)
        assert annotated.width > label.image.width  # side panel added

        out_png = Path(tmp) / "annotated.png"
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli_main(["read", str(img_path), "--engine", "tesseract", "--json",
                             "--annotate", str(out_png)])
        assert code == 0 and out_png.exists()
        payload = json.loads(buf.getvalue())
        assert payload["nutrients"]["fat_100g"]["value"] == pytest.approx(25.0)


def test_cli_annotate_rejects_several_inputs(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    for path in (a, b):
        path.write_bytes(b"x")
    assert cli_main(["read", str(a), str(b), "--annotate", str(tmp_path / "o.png")]) == 2


def test_cli_lists_engines():
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert cli_main(["engines"]) == 0
    text = buf.getvalue()
    for name in ("paddleocr", "rapidocr", "doctr", "easyocr", "tesseract"):
        assert name in text
