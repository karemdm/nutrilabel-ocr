# Benchmark results

| Engine | F1 | Precision | Recall | Perfect labels | OCR value recall | OCR s/img (median) |
|---|---:|---:|---:|---:|---:|---:|
| Google Cloud Vision (reference*) | **0.834** | 0.887 | 0.787 | 41.9% | 98.0% | - |

*198 images, `test` split. \*The reference row uses the dataset's own Google Cloud Vision tokens, so it measures the extractor with near-perfect OCR.*
