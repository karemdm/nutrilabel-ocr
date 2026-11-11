"""VLM text output, sanity checks, the web UI server and the folder CLI."""

import csv
import io
import json
import tempfile
import threading
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nutrilabel import NutritionExtractor
from nutrilabel.cli import collect_images
from nutrilabel.cli import main as cli_main
from nutrilabel.engines import ENGINES, TesseractEngine, get_engine
from nutrilabel.engines.vlm import HF_VLM_PRESETS, VLMEngine, paddle_vl_to_text
from nutrilabel.evaluation.synthetic import render_label
from nutrilabel.extraction.markup import text_to_words
from nutrilabel.extraction.validation import check
from nutrilabel.types import ExtractionResult, NutrientValue, OCRResult

MARKDOWN = """## Nutrition information
| | Per 100 g | Per serving (30 g) | %RI* |
|---|---|---|---|
| Energy | 1873 kJ / 449 kcal | 562 kJ / 135 kcal | 7% |
| Fat | 25 g | 7.5 g | 11% |
| of which saturates | 9.5 g | 2.9 g | 15% |
| Carbohydrate | 51 g | 15 g | 6% |
| Protein | 6.4 g | 1.9 g | 4% |
| Salt | 0.45 g | 0.14 g | 2% |
"""
EXPECTED = {
    "energy_kj_100g": 1873, "energy_kcal_100g": 449, "energy_kj_serving": 562,
    "energy_kcal_serving": 135, "fat_100g": 25, "fat_serving": 7.5,
    "saturated_fat_100g": 9.5, "saturated_fat_serving": 2.9, "carbohydrates_100g": 51,
    "carbohydrates_serving": 15, "proteins_100g": 6.4, "proteins_serving": 1.9,
    "salt_100g": 0.45, "salt_serving": 0.14,
}


# ----------------------------------------------------------------- VLM text output
def test_markdown_table_goes_through_the_same_extractor():
    words = text_to_words(MARKDOWN)
    assert NutritionExtractor().extract(OCRResult(words, "vlm")).as_simple_dict() == EXPECTED


def test_html_table_and_plain_text():
    html = ("<table><tr><td></td><td>pour 100 g</td></tr>"
            "<tr><td>Énergie</td><td>1513 kJ / 361 kcal</td></tr>"
            "<tr><td>Matières grasses</td><td>15 g</td></tr>"
            "<tr><td>Sel</td><td>0,47 g</td></tr></table>")
    got = NutritionExtractor().extract(text_to_words(html)).as_simple_dict()
    assert got == {"energy_kj_100g": 1513, "energy_kcal_100g": 361, "fat_100g": 15,
                   "salt_100g": 0.47}
    text = "Energy 1500 kJ - 353 kcal; fat 1,0 g; of which saturates 0,3 g; salt 0 g"
    got = NutritionExtractor().extract(text_to_words(text)).as_simple_dict()
    assert got["fat_100g"] == 1.0 and got["saturated_fat_100g"] == 0.3


def test_paddle_vl_output_collection():
    md = SimpleNamespace(markdown={"markdown_texts": "| Fat | 3 g |"})
    blocks = {"parsing_res_list": [{"block_content": "Salt 0.2 g"}]}
    assert paddle_vl_to_text([md, blocks]) == "| Fat | 3 g |\nSalt 0.2 g"
    assert paddle_vl_to_text(None) == ""


def test_hf_presets_are_registered_with_real_model_ids():
    for name, (model_id, _) in HF_VLM_PRESETS.items():
        assert ENGINES[name].model_id == model_id
        assert "/" in model_id


class _FakeChatServer(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible endpoint answering with the markdown table."""

    received: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).received.append(body)
        out = json.dumps({"choices": [{"message": {"content": MARKDOWN}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def _serve(handler):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def test_vlm_engine_against_local_openai_compatible_server():
    httpd = _serve(_FakeChatServer)
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
        engine = get_engine("vlm", model="test-model", base_url=url)
        ocr = engine.read(np.full((60, 80, 3), 255, dtype=np.uint8))
        assert NutritionExtractor().extract(ocr).as_simple_dict() == EXPECTED
        body = _FakeChatServer.received[-1]
        assert body["model"] == "test-model" and body["temperature"] == 0
        content = body["messages"][0]["content"]
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    finally:
        httpd.shutdown()


def test_vlm_engine_is_not_available_without_configuration():
    assert not VLMEngine.is_available() or VLMEngine().model


# ----------------------------------------------------------------- sanity checks
def _result(**values):
    return ExtractionResult(nutrients={
        k: NutrientValue(k, v, "g", raw=f"{v:g} g") for k, v in values.items()
    })


def test_lost_decimal_separator_is_repaired():
    res = check(_result(fat_100g=12.5, saturated_fat_100g=95.0, carbohydrates_100g=429.0))
    assert res.nutrients["saturated_fat_100g"].value == 9.5
    assert res.nutrients["carbohydrates_100g"].value == 42.9
    assert len(res.warnings) == 2


def test_values_with_a_separator_are_only_flagged():
    res = check(_result(fat_100g=1.5, saturated_fat_100g=2.5))
    assert res.nutrients["saturated_fat_100g"].value == 2.5
    assert res.warnings and res.nutrients["saturated_fat_100g"].confidence < 1


def test_energy_mismatch_is_flagged():
    res = ExtractionResult(nutrients={
        "energy_kj_100g": NutrientValue("energy_kj_100g", 1873, "kJ", "1873"),
        "energy_kcal_100g": NutrientValue("energy_kcal_100g", 149, "kcal", "149"),
    })
    assert any("do not match" in w for w in check(res).warnings)


# ----------------------------------------------------------------- web UI
def test_ui_page_and_snapshot_injection():
    from nutrilabel.ui import page_html

    html = page_html()
    assert "Open folder" in html and "webkitdirectory" in html
    snap = page_html({"engine": "x", "items": [], "active": 0})
    assert "window.__PRELOADED__" in snap


@pytest.mark.skipif(not TesseractEngine.is_available(), reason="tesseract not installed")
def test_ui_server_reads_an_uploaded_image():
    from nutrilabel.ui.server import _Handler

    handler = type("H", (_Handler,), {"default_engine": "tesseract"})
    httpd = _serve(handler)
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        engines = json.loads(opener.open(f"{base}/api/engines").read())
        assert "tesseract" in engines["engines"]
        buf = io.BytesIO()
        render_label(lang="en").image.save(buf, format="PNG")
        req = urllib.request.Request(f"{base}/api/read?engine=tesseract&name=a.png",
                                     data=buf.getvalue(), method="POST")
        data = json.loads(opener.open(req).read())
        assert data["name"] == "a.png" and data["engine"] == "tesseract"
        assert data["nutrients"]["fat_100g"]["value"] == 25
        assert data["annotated"].startswith("data:image/jpeg;base64,")
    finally:
        httpd.shutdown()


# ----------------------------------------------------------------- folder CLI
def test_collect_images_walks_folders():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "sub").mkdir()
        for name in ("a.jpg", "sub/b.PNG", "notes.txt"):
            (root / name).write_bytes(b"x")
        found = [p.name for p in collect_images([tmp])]
        assert found == ["a.jpg", "b.PNG"]


@pytest.mark.skipif(not TesseractEngine.is_available(), reason="tesseract not installed")
def test_cli_folder_to_csv_and_annotated_images():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "photos").mkdir()
        render_label(lang="en").image.save(root / "photos" / "a.png")
        render_label(lang="de", serving_g=None).image.save(root / "photos" / "b.jpg")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = cli_main(["read", str(root / "photos"), "--engine", "tesseract", "-q",
                             "--csv", str(root / "out.csv"), "--out-dir", str(root / "ann")])
        assert code == 0
        rows = list(csv.DictReader((root / "out.csv").open(encoding="utf-8")))
        assert len(rows) == 2 and float(rows[0]["fat_100g"]) == 25
        assert sorted(p.name for p in (root / "ann").iterdir()) == [
            "a.annotated.png", "b.annotated.png"]
