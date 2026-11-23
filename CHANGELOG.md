# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ultralytics` engine: the Ultralytics OCR solution (YOLO26 text detector + PP-OCRv6 recogniser), with detector size selectable as `ultralytics:n|s|m|l|x`, a new `ultralytics` extra and tests. No benchmark results yet.

## [0.1.0] - 2026-11-11

### Added

- Layout-aware nutrient extractor for EU and US nutrition facts tables, supporting rotated and
  horizontal layouts, 10 nutrients, and both per-100g and per-serving values.
- Six classic OCR engine adapters (PaddleOCR, RapidOCR, Surya, docTR, EasyOCR, Tesseract) behind one `engine=` parameter.
- Benchmark on the Open Food Facts nutrient dataset.
- Command-line interface (`nutrilabel`).
- Local web interface (`nutrilabel ui`): open one image or a whole folder, review annotated results, export CSV/JSON.
- Folder mode in the CLI (`nutrilabel read photos/ --csv results.csv --out-dir annotated/`).
- Vision-language OCR engines: PaddleOCR-VL, olmOCR-2, GLM-OCR, Chandra, Nanonets-OCR2, DeepSeek-OCR-2, Qwen2.5-VL, plus any local OpenAI-compatible server (Ollama, vLLM).
- Nutritional sanity checks that repair lost decimal separators.
- Brazilian (ANVISA) tables: *Quantidade por porção*, *%VD* column, *Gorduras saturadas/trans*.
- Web interface: per-engine results, "Run <engine>" button, cross-engine comparison, original photo shown on errors.
- Bundled DejaVu Sans font so rendered labels and tests are identical on every OS.
- Windows helpers: `scripts/run_all.bat` and `scripts/publish.bat`; PaddleOCR oneDNN workaround and model fallbacks.
- Docker image.

[0.1.0]: https://github.com/karemdm/nutrilabel-ocr/releases/tag/v0.1.0
