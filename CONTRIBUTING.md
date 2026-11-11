# Contributing to nutrilabel-ocr

Thanks for your interest in improving nutrilabel-ocr.

## Development setup

```bash
git clone https://github.com/karemdm/nutrilabel-ocr.git
cd nutrilabel-ocr
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # or: make install
```

Tesseract must be installed on your system for the Tesseract engine and its tests
(e.g. `sudo apt-get install tesseract-ocr tesseract-ocr-fra tesseract-ocr-deu tesseract-ocr-spa`).

## Tests and lint

```bash
ruff check .      # lint
pytest            # tests and doctests (tests/ and src/)
```

Optional: `pre-commit install` to run ruff and basic file checks on each commit.

## Adding a new OCR engine

1. Subclass `OCREngine` in `src/nutrilabel/engines/classic.py` (or `vlm.py`); see [docs/engines.md](docs/engines.md#adding-an-engine).
2. Implement `_check_import` (raise a clear error if the optional dependency is missing) and
   `_run`, which returns `list[Word]`.
3. Register the engine in `ENGINES` and add it to `PREFERENCE` in `src/nutrilabel/engines/__init__.py`.
4. Add a parsing unit test that feeds a mocked raw output of the engine into the adapter.
5. Run the benchmark (see below) and include the numbers in your PR.

## Adding a language

1. Add the language's nutrient label regexes in `src/nutrilabel/extraction/nutrients.py`.
2. Add test cases covering typical labels in that language.
3. Run the benchmark and report the results.

## Running the benchmark

```bash
make benchmark    # python -m benchmark.run
make report       # python -m benchmark.report --gallery 6
```

Results are written under `benchmark/results/`; the report is written to `docs/`.

## Pull request checklist

- [ ] Tests pass (`pytest`).
- [ ] `ruff check .` is clean.
- [ ] If the extractor changed, the benchmark numbers in the docs are updated.
- [ ] No proprietary or confidential data, product photos you are not allowed to share,
      or credentials (API keys, tokens, passwords) are committed.
