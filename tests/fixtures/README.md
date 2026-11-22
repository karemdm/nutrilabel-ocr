# Test fixtures

`off_sample.jsonl`: 40 annotated documents (OCR tokens, boxes and nutrient labels) taken from the
`openfoodfacts/nutrient-detection-layout` dataset (Open Food Facts contributors, CC BY-SA 3.0).
It is used by the regression tests and by the offline benchmark smoke run
(`python -m benchmark.run --engines gcv --jsonl tests/fixtures/off_sample.jsonl`).
