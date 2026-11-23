# Documentation assets

Sources of the files in this folder.

| File | Source |
|---|---|
| `examples/*.jpg` | Photos from the test split of the `openfoodfacts/nutrient-detection-layout` dataset (Open Food Facts contributors, CC BY-SA 3.0), annotated by this tool with the `rapidocr` engine (`python -m benchmark.report --gallery 6`). The number in each name is the image index, followed by the product barcode. |
| `real_examples.gif` | Animation of the `examples/` images, built by `benchmark.report --gallery`. |
| `real_examples_ultralytics.gif` | Same for `ultralytics:s`: 6 test-split photos from its cached OCR (`benchmark.report --gallery 6 --gallery-engine ultralytics:s --gallery-gif real_examples_ultralytics.gif --gallery-skip 12,22,38`). Photos: Open Food Facts contributors (CC BY-SA 3.0). |
| `demo.gif`, `demo_annotated.png` | Package photos rendered by `nutrilabel.evaluation.synthetic` (no real photos), read and drawn by `scripts/make_demo.py`. |
| `demo_ui.gif` | Recording of the web interface reading six photos from the dataset above. |
| `ui.png` | Screenshot of the web interface rendered by `scripts/make_ui_snapshot.py` on four synthetic sample labels read with the `paddleocr` engine. |
| `f1_vs_latency.png`, `per_nutrient_f1.png` | Charts generated from `benchmark/results/summary.json` by `python -m benchmark.report`. |
