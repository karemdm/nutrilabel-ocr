"""Benchmark OCR engines end-to-end on the Open Food Facts nutrient dataset.

Usage
-----
    # every installed classic engine + the Google Cloud Vision reference, full test split
    python -m benchmark.run

    # vision-language OCR models (GPU recommended)
    python -m benchmark.run --engines paddleocr-vl,olmocr,glm-ocr

    # quick check: two engines, 30 images
    python -m benchmark.run --engines rapidocr,tesseract --limit 30

    # offline: reference OCR only, from a token dump (no images needed)
    python -m benchmark.run --engines gcv --jsonl tests/fixtures/off_sample.jsonl

For each engine and image the script runs OCR (cached under ``<out>/ocr``),
extracts nutrients, scores every field against the annotations and writes:

* ``<out>/<engine>/predictions.jsonl`` - per-image predictions and errors
* ``<out>/summary.json`` - metrics per engine (used by ``benchmark.report``)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

from nutrilabel.engines import ENGINES, PREFERENCE, VLM_ENGINES, get_engine
from nutrilabel.evaluation import Report, gold_from_bio, score_document, words_from_dataset
from nutrilabel.extraction.extractor import NutritionExtractor
from nutrilabel.imaging.preprocess import prepare
from nutrilabel.types import OCRResult

from .data import Sample, iter_hf, iter_jsonl

REFERENCE = "gcv"  # the dataset's own Google Cloud Vision tokens


def _ocr_cached(engine_name: str, engine, sample: Sample, cache_dir: Path, max_side: int) -> OCRResult:
    path = cache_dir / engine_name / f"{sample.idx:05d}.json"
    if path.exists():
        return OCRResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
    array, scale = prepare(sample.image, max_side=max_side)
    ocr = engine.read(array)
    # store boxes in original-image coordinates
    inv = 1.0 / scale
    ocr = OCRResult(
        words=[type(w)(w.text, w.bbox.scale(inv, inv), w.confidence) for w in ocr.words],
        engine=ocr.engine,
        width=round(ocr.width * inv),
        height=round(ocr.height * inv),
        elapsed_s=ocr.elapsed_s,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ocr.to_dict(), ensure_ascii=False), encoding="utf-8")
    return ocr


def evaluate_engine(name: str, samples: list[Sample], out: Path, max_side: int) -> dict:
    extractor = NutritionExtractor()
    engine = None if name == REFERENCE else get_engine(name)
    report = Report()
    latencies: list[float] = []
    errors = 0
    pred_path = out / name / "predictions.jsonl"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with pred_path.open("w", encoding="utf-8") as fh:
        for i, s in enumerate(samples, 1):
            gold = gold_from_bio(s.tokens, s.labels)
            record = {"idx": s.idx, "barcode": s.meta.get("barcode"), "gold": {
                k: [g.value, g.unit] for k, g in gold.items()}}
            try:
                if engine is None:
                    ocr = words_from_dataset(s.tokens, s.bboxes)
                else:
                    ocr = _ocr_cached(name, engine, s, out / "ocr", max_side)
                    latencies.append(ocr.elapsed_s)
                result = extractor.extract(ocr)
                score = score_document(result, gold)
                report.add_value_recall(ocr.text, gold)
                record.update(
                    pred={k: [v.value, v.unit] for k, v in result.nutrients.items()},
                    tp=score.tp, fp=score.fp, fn=score.fn, ocr_s=round(ocr.elapsed_s, 3),
                    warnings=result.warnings,
                )
            except Exception as e:  # noqa: BLE001 - one bad image must not stop the run
                errors += 1
                score = score_document({}, gold)
                record.update(error=f"{type(e).__name__}: {e}", fn=list(gold))
                traceback.print_exc(file=sys.stderr)
            report.add(score)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            if i % 20 == 0:
                print(f"  [{name}] {i}/{len(samples)}  F1 so far {report.overall.f1:.3f}",
                      file=sys.stderr)
    summary = report.to_dict()
    summary.update(
        engine=name,
        errors=errors,
        wall_time_s=round(time.perf_counter() - t0, 1),
        ocr_latency_median_s=round(statistics.median(latencies), 3) if latencies else None,
        ocr_latency_p90_s=round(sorted(latencies)[int(0.9 * (len(latencies) - 1))], 3)
        if latencies else None,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engines", default="all",
                   help=f"comma list from: {REFERENCE},{','.join(PREFERENCE + VLM_ENGINES)} | "
                        "all (installed classic engines) | all-vlm (installed VLM engines)")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--limit", type=int, default=None, help="only the first N images")
    p.add_argument("--jsonl", default=None, help="token-only dump instead of the HF dataset")
    p.add_argument("--max-side", type=int, default=1600)
    p.add_argument("--out", default="benchmark/results")
    args = p.parse_args(argv)

    if args.engines == "all":
        names = [REFERENCE] + [n for n in PREFERENCE if ENGINES[n].is_available()]
    elif args.engines == "all-vlm":
        names = [n for n in VLM_ENGINES if ENGINES[n].is_available()]
    else:
        names = [n.strip() for n in args.engines.split(",") if n.strip()]
    if args.jsonl and any(n != REFERENCE for n in names):
        p.error("--jsonl has no images: only the 'gcv' reference can run on it")

    samples = list(iter_jsonl(args.jsonl, args.limit) if args.jsonl else iter_hf(args.split, args.limit))
    print(f"{len(samples)} samples, engines: {', '.join(names)}", file=sys.stderr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summaries = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for name in names:
        print(f"== {name}", file=sys.stderr)
        summaries[name] = evaluate_engine(name, samples, out, args.max_side)
        summaries[name]["split"] = "jsonl" if args.jsonl else args.split
        summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        s = summaries[name]
        print(f"   F1={s['f1']:.3f}  P={s['precision']:.3f}  R={s['recall']:.3f}  "
              f"OCR median={s['ocr_latency_median_s'] if s['ocr_latency_median_s'] is not None else '-'}s",
              file=sys.stderr)
    print(f"summary written to {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
