"""Compare two OCR engine variants on the documents where the base engine did worst.

    python -m benchmark.compare_hard --base ultralytics:s --candidate ultralytics:x --n 10

The base engine's ``predictions.jsonl`` (written by ``benchmark.run``) is read, a
per-document F1 is computed from its stored tp/fp/fn key lists, and the N documents
with the lowest F1 are selected (ties by idx; documents without gold fields are
skipped). Only those images are run through the candidate engine, using the same OCR
cache, extractor and scoring as ``benchmark.run``, so the candidate's OCR results are
cached under ``<out>/ocr/<candidate folder>/`` and can be reused later.

Output: ``<out>/compare_<base>_vs_<candidate>.json`` plus a Markdown table on stdout.
The numbers describe a small sample chosen as the base engine's worst cases; they are
not comparable to full-split results.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from nutrilabel.evaluation import gold_from_bio, score_document
from nutrilabel.evaluation.metrics import Counts

from .run import _folder, _ocr_cached


def doc_counts(record: dict) -> Counts:
    """Counts for one predictions.jsonl record (tp/fp/fn are lists of keys)."""
    return Counts(tp=len(record.get("tp") or []), fp=len(record.get("fp") or []),
                  fn=len(record.get("fn") or []))


def select_hard(records: list[dict], n: int) -> list[dict]:
    """The n records with the lowest per-document F1 (ties by idx), skipping documents without gold."""
    rows = [r for r in records if r.get("gold")]
    rows.sort(key=lambda r: (doc_counts(r).f1, r["idx"]))
    return rows[:n]


def doc_metrics(counts: Counts, ocr_s: float | None) -> dict:
    return {"f1": round(counts.f1, 4), "precision": round(counts.precision, 4),
            "recall": round(counts.recall, 4), "ocr_s": ocr_s}


def aggregate(rows: list[dict]) -> dict:
    """Mean per-document F1/precision/recall and mean OCR seconds (over documents that have one)."""
    out: dict = {"documents": len(rows)}
    for side in ("base", "candidate"):
        agg = {}
        for k in ("f1", "precision", "recall"):
            vals = [r[side][k] for r in rows]
            agg[f"mean_{k}"] = round(sum(vals) / len(vals), 4) if vals else None
        secs = [r[side]["ocr_s"] for r in rows if r[side]["ocr_s"] is not None]
        agg["mean_ocr_s"] = round(sum(secs) / len(secs), 3) if secs else None
        out[side] = agg
    return out


def markdown_table(rows: list[dict], agg: dict, base: str, candidate: str) -> str:
    def s(v):
        return "-" if v is None else f"{v:.3f}"

    lines = [f"| idx | F1 {base} | F1 {candidate} | P {base} | P {candidate} | R {base} | R {candidate} |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        b, c = r["base"], r["candidate"]
        lines.append(f"| {r['idx']} | {s(b['f1'])} | {s(c['f1'])} | {s(b['precision'])} | "
                     f"{s(c['precision'])} | {s(b['recall'])} | {s(c['recall'])} |")
    b, c = agg["base"], agg["candidate"]
    lines.append(f"| **mean** | **{s(b['mean_f1'])}** | **{s(c['mean_f1'])}** | {s(b['mean_precision'])} | "
                 f"{s(c['mean_precision'])} | {s(b['mean_recall'])} | {s(c['mean_recall'])} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="engine whose worst documents are selected, e.g. ultralytics:s")
    p.add_argument("--candidate", required=True, help="engine run on those documents, e.g. ultralytics:x")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--max-side", type=int, default=1600)
    p.add_argument("--out", default="benchmark/results")
    args = p.parse_args(argv)

    from nutrilabel.engines import get_engine
    from nutrilabel.extraction.extractor import NutritionExtractor

    from .data import iter_hf

    out = Path(args.out)
    pred_path = out / _folder(args.base) / "predictions.jsonl"
    lines = pred_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    wanted = {r["idx"]: r for r in select_hard(records, args.n)}
    print(f"{len(wanted)} hardest documents of {args.base}: {sorted(wanted)}", file=sys.stderr)

    extractor = NutritionExtractor()
    engine = get_engine(args.candidate)
    rows: list[dict] = []
    for s in iter_hf(args.split):
        if s.idx not in wanted:
            continue
        base_rec = wanted[s.idx]
        gold = gold_from_bio(s.tokens, s.labels)
        ocr_s = None
        try:
            ocr = _ocr_cached(args.candidate, engine, s, out / "ocr", args.max_side)
            ocr_s = round(ocr.elapsed_s, 3)
            sc = score_document(extractor.extract(ocr), gold)
            cand = Counts(len(sc.tp), len(sc.fp), len(sc.fn))
        except Exception:  # noqa: BLE001 - one bad image must not stop the comparison
            traceback.print_exc(file=sys.stderr)
            cand = Counts(0, 0, len(gold))
        base_m = doc_metrics(doc_counts(base_rec), base_rec.get("ocr_s"))
        rows.append({"idx": s.idx, "base": base_m, "candidate": doc_metrics(cand, ocr_s)})
        print(f"  idx {s.idx}: {base_m['f1']:.3f} -> {rows[-1]['candidate']['f1']:.3f}", file=sys.stderr)
        if len(rows) == len(wanted):
            break

    rows.sort(key=lambda r: (r["base"]["f1"], r["idx"]))
    agg = aggregate(rows)
    result = {"base": args.base, "candidate": args.candidate, "split": args.split, "n": args.n,
              "selection": "lowest per-document F1 of the base engine (ties by idx)",
              "aggregate": agg, "documents": rows}
    target = out / f"compare_{_folder(args.base)}_vs_{_folder(args.candidate)}.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(markdown_table(rows, agg, args.base, args.candidate))
    print(f"\nwritten to {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
