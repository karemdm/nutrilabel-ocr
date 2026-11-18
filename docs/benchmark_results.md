# Benchmark results

| Engine | F1 | Precision | Recall | Perfect labels | OCR value recall | OCR s/img (median) |
|---|---:|---:|---:|---:|---:|---:|
| Google Cloud Vision (reference*) | **0.834** | 0.887 | 0.787 | 41.9% | 98.0% | - |
| RapidOCR (ONNX) | **0.730** | 0.856 | 0.636 | 17.6% | 88.3% | 5.661 |
| docTR | **0.699** | 0.808 | 0.617 | 18.1% | 80.6% | 4.068 |
| EasyOCR | **0.544** | 0.677 | 0.454 | 6.0% | 52.3% | 8.644 |

*198 images, `test` split. \*The reference row uses the dataset's own Google Cloud Vision tokens, so it measures the extractor with near-perfect OCR.*
