# OCR engines

The extractor is engine-agnostic: every engine is an `OCREngine` subclass that turns an RGB image into a list of `Word(text, bbox, confidence)`. Engines are registered by name in `ENGINES` (`src/nutrilabel/engines/__init__.py`). Libraries are imported lazily, so installing one extra is enough to use that engine.

```python
from nutrilabel import NutritionReader

reader = NutritionReader(engine="rapidocr")       # or "auto", an engine name, or an OCREngine instance
result = reader.read("label.jpg")
```

Licences below are the ones stated in the code. Where the code says nothing, check the library or model card before use; this page does not assert a licence.

## Classic engines

Classic engines return word or line boxes in image coordinates. They are the only engines `engine="auto"` can select.

| Name | pip extra | Library / model | Licence as stated in code | Hardware note |
|---|---|---|---|---|
| `paddleocr` | `paddle` | PaddleOCR (PP-OCR), supports the 2.x and 3.x APIs | not stated; check the project | CPU by default in this adapter (options are passed through `**paddle_kwargs`) |
| `rapidocr` | `rapidocr` | RapidOCR: PP-OCR models exported to ONNX; works with `rapidocr` >= 2 and `rapidocr_onnxruntime` | not stated; check the project | "No Paddle install, fast on CPU" (docstring) |
| `doctr` | `doctr` | docTR (Mindee): DBNet/FAST detector + CRNN/PARSeq recogniser, word-level output | not stated; check the project | `[torch]` extra; no GPU switch in the adapter |
| `easyocr` | `easyocr` | EasyOCR: CRAFT detector + CRNN recogniser; all Latin languages share one model | not stated; check the project | `gpu=False` by default |
| `tesseract` | `tesseract` | Tesseract 4/5 (LSTM) via `pytesseract` | not stated; check the project | Needs the Tesseract binary on `PATH` (availability check calls `get_tesseract_version()`); CPU |
| `surya` | `surya` | Surya OCR (Datalab): transformer detector + recogniser, 90+ languages, line boxes; `surya-ocr` >= 0.14 predictor API and the older `run_ocr` API | Code GPL-3.0, weights under a modified OpenRAIL licence (free for research and personal use) | Not stated in code |

Constructor options (all engines also accept `languages`, default `("en", "fr", "de", "es", "it", "nl", "pt")`; adapters keep only what the library supports):

| Engine | Options |
|---|---|
| `paddleocr` | `lang="latin"` (multilingual Latin recogniser, keeps accents; `"en"` is faster but drops accents), `**paddle_kwargs` forwarded to `PaddleOCR`. On 3.x the adapter sets `use_doc_orientation_classify=False`, `use_doc_unwarping=False`, `use_textline_orientation=True`; on 2.x `use_angle_cls=True` |
| `rapidocr` | `**rapid_kwargs` forwarded to `RapidOCR` |
| `doctr` | `det_arch="fast_base"`, `reco_arch="crnn_vgg16_bn"` |
| `easyocr` | `gpu=False` |
| `tesseract` | `psm=11`, `oem=1` |
| `surya` | none beyond `languages` |

### Tesseract page segmentation mode

The adapter defaults to `--psm 11` ("sparse text") with `--oem 1` (LSTM). The adapter's docstring gives the reason: nutrition tables consist of short, widely spaced tokens, which is what sparse-text mode targets. Other modes can be set with `get_engine("tesseract", psm=6)`; compare the result with the benchmark before changing the default. Language codes are mapped (`en` to `eng`, `fr` to `fra`, ...) and only installed language packs are used, falling back to `eng`.

## Vision-language (VLM) engines

VLM engines receive the image and a prompt (`VLM_PROMPT`: transcribe the nutrition table as markdown, copy labels and numbers as printed, keep column headers, answer `NONE` if there is no table). Their text is converted to a word grid by `markup.text_to_words` (see [architecture.md](architecture.md#text-only-vlm-output)). Boxes are synthetic (`boxes_are_synthetic = True`), so there are no image positions to draw. A GPU is recommended for all of them.

| Name | pip extra | Model | Licence as stated in code | Hardware note |
|---|---|---|---|---|
| `paddleocr-vl` | `paddle-vl` (install separately, not part of `all`) | PaddleOCR-VL via `paddleocr[doc-parser]>=3.3` | not stated; check the model card | "Experimental; a GPU is strongly recommended" |
| `olmocr` | `vlm` | `allenai/olmOCR-2-7B-1025` | Apache-2.0 (Qwen2.5-VL fine-tune) | 7B model; GPU recommended |
| `glm-ocr` | `vlm` | `zai-org/GLM-OCR` | not stated; check the model card | Needs a recent `transformers` release |
| `chandra` | `vlm` | `datalab-to/chandra-ocr-2` | OpenRAIL (code comment: "non-commercial use is fine") | GPU recommended |
| `nanonets-ocr` | `vlm` | `nanonets/Nanonets-OCR2-3B` | not stated; check the model card | GPU recommended |
| `deepseek-ocr` | `vlm` | `deepseek-ai/DeepSeek-OCR-2` | not stated; check the model card | Runs custom model code (`trust_remote_code=True`); "vLLM recommended" |
| `qwen-vl` | `vlm` | `Qwen/Qwen2.5-VL-3B-Instruct` (general VLM baseline) | not stated; check the model card | GPU recommended |
| `hf-vlm` | `vlm` | Any image-text-to-text checkpoint, default `Qwen/Qwen2.5-VL-3B-Instruct` | depends on the model; check its card | "7B models take minutes per image on CPU" (docstring) |
| `vlm` | none | Any model behind an OpenAI-compatible endpoint | depends on the model | Runs on the server; see below |

The `vlm` extra installs `transformers>=4.57`, `torch>=2.3`, `accelerate>=0.30` and `torchvision`. The first call of a transformers-based engine downloads the weights (several GB).

`hf-vlm` and the six presets (`olmocr`, `glm-ocr`, `chandra`, `nanonets-ocr`, `deepseek-ocr`, `qwen-vl`) share the `HFVLMEngine` class. The presets only set `model_id` and `trust_remote_code`, except `deepseek-ocr`, which has its own subclass (`DeepSeekOCREngine`): it loads the model with `AutoModel` and calls its `infer` method, and it counts as available only when CUDA is present.

`HFVLMEngine` options: `model_id` (overrides the checkpoint, for example `HFVLMEngine(model_id="org/model")`), `prompt=VLM_PROMPT`, `max_new_tokens=1024`, `max_side=1280` (longest side of the image sent to the model), `device_map="auto"`, `torch_dtype="auto"`. Generation is greedy (`do_sample=False`).

`PaddleOCRVLEngine` options: `languages` and `**pipeline_kwargs` forwarded to `PaddleOCRVL`.

### The `vlm` engine (OpenAI-compatible server)

`VLMEngine` posts the image (JPEG, base64) to `<base_url>/chat/completions` with `temperature=0`. Options and environment variables:

| Option | Env var | Default |
|---|---|---|
| `model` | `NUTRILABEL_VLM_MODEL` | none; the engine is "available" only if this variable is set |
| `base_url` | `NUTRILABEL_VLM_URL` | `http://localhost:11434/v1` (Ollama) |
| `api_key` | `NUTRILABEL_VLM_API_KEY` | empty; sent as `Authorization: Bearer ...` only if set |
| `prompt` | | `VLM_PROMPT` |
| `timeout` | | `180.0` seconds |
| `max_side` | | `1280` |

Passing `model=` explicitly also works without the environment variable (`get_engine("vlm", model="...")`).

Ollama example (a locally pulled model):

```bash
ollama pull qwen2.5vl:7b
export NUTRILABEL_VLM_URL=http://localhost:11434/v1
export NUTRILABEL_VLM_MODEL=qwen2.5vl:7b
nutrilabel read label.jpg --engine vlm
```

vLLM example:

```bash
vllm serve allenai/olmOCR-2-7B-1025 --port 8000
export NUTRILABEL_VLM_URL=http://localhost:8000/v1
export NUTRILABEL_VLM_MODEL=allenai/olmOCR-2-7B-1025
nutrilabel read label.jpg --engine vlm
```

Product photos are sent to whatever server you configure. The docstring recommends keeping the server local.

## Choosing an engine

### `engine="auto"`

`get_engine("auto")` (the default of `NutritionReader`) takes the first installed engine in `PREFERENCE`:

```python
PREFERENCE = ["paddleocr", "rapidocr", "surya", "doctr", "easyocr", "tesseract"]
```

Only classic engines are in this list. VLM engines are never picked automatically: a first call downloads several GB of weights, and the `vlm` engine would contact a server. If no classic engine is installed, `EngineNotAvailable` is raised with an install hint. The comment in the code says to keep `PREFERENCE` in sync with the latest benchmark; `python -m benchmark.report` prints the recommended order.

### `available_engines`

```python
from nutrilabel.engines import available_engines
available_engines()                    # installed classic engines, in preference order
available_engines(include_vlm=True)    # plus installed VLM engines
```

With `include_vlm=True` the names in `VLM_ENGINES` are appended: `paddleocr-vl`, `olmocr`, `glm-ocr`, `chandra`, `nanonets-ocr`, `deepseek-ocr`, `qwen-vl`, `vlm`. `hf-vlm` is registered in `ENGINES` and can be requested by name, but is not in `VLM_ENGINES`, so it is not listed or used by the benchmark's `all-vlm`. A transformers-based engine counts as available when `torch` and `transformers` import; that does not mean a GPU or the weights are present.

### Aliases

Names are lower-cased and stripped, then mapped:

| Alias | Engine |
|---|---|
| `paddle` | `paddleocr` |
| `rapid` | `rapidocr` |
| `easy` | `easyocr` |
| `tess` | `tesseract` |
| `paddle-vl`, `paddleocrvl` | `paddleocr-vl` |

An unknown name raises `ValueError` listing the valid names; an engine whose library is missing raises `EngineNotAvailable` with a `pip install 'nutrilabel-ocr[<extra>]'` hint.

## Adding an engine

1. Subclass `OCREngine` in `engines/classic.py` (or `vlm.py`). Set `name` (the registry key) and `pip_extra` as class variables.
2. Implement `_check_import` as a classmethod that imports the backing library (or checks the binary) and raises if it is missing. `is_available()` calls it and treats any exception as "not installed".
3. Implement `_run(self, image) -> list[Word]`. The image is an RGB `uint8` array of shape `(H, W, 3)`. Load heavy models lazily on first use and cache them on the instance. Return boxes in the image's pixel coordinates as `BBox(x0, y0, x1, y1)` (`BBox.from_points` converts polygons). Whole-line boxes are fine; `split_words` splits them. For text-only models, return `text_to_words(text)` and set `boxes_are_synthetic = True`.
4. Register the class in the tuple that builds `ENGINES` in `engines/__init__.py`, add its name to `PREFERENCE` (classic) or `VLM_ENGINES` (VLM) if it should be listed, and add an alias if useful. For a VLM that runs through transformers, add an entry to `HF_VLM_PRESETS` instead; the class is generated.
5. Add the extra to `[project.optional-dependencies]` in `pyproject.toml`.
6. Keep the conversion from the library's native output to `list[Word]` in a module-level function (like `paddle_v3_to_words` or `doctr_export_to_words`) and add a parser test in `tests/test_engines.py` that feeds a hand-written native result and checks the words. Parser tests need no OCR library installed.
7. Run the benchmark for the new engine (see [benchmark.md](benchmark.md)) and update `PREFERENCE` if it changes the recommendation.

## Why keep several engines

No single engine wins on every axis, and the right choice depends on the deployment:

- **Accuracy.** Engines differ in how well they read small, low-contrast or curved print, and in whether they keep decimal separators and unit letters, which the extractor depends on.
- **Latency.** VLMs generate text token by token; the code notes that 7B models take minutes per image on CPU. Classic engines are designed for CPU use. Measure both with the benchmark.
- **Install size and dependencies.** RapidOCR runs on ONNX Runtime without a deep-learning framework; PaddleOCR, docTR, EasyOCR and Surya pull in Paddle or PyTorch; VLM engines add multi-GB weights. Tesseract needs a system binary.
- **GPU.** Classic engines are usable on CPU. VLM engines are impractical without a GPU or a remote server.
- **Licence.** Surya's code is GPL-3.0 with restricted-use weights, Chandra uses OpenRAIL, olmOCR is Apache-2.0. Check each licence against your use case.
- **Privacy and data flow.** The `vlm` engine sends images to a server you configure.

The benchmark (`python -m benchmark.run`, see [benchmark.md](benchmark.md)) measures F1 and per-image OCR latency for each engine on the same images. Its output decides the order of `PREFERENCE`, which is what `engine="auto"` uses; override it per deployment by passing an engine name.
