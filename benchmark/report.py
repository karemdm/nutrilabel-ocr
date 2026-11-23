"""Turn ``benchmark/results/summary.json`` into tables, charts and an example gallery.

    python -m benchmark.report                   # markdown table + charts
    python -m benchmark.report --gallery 6       # + annotated real examples and a GIF

Outputs go to ``docs/`` so the README can embed them:

* ``docs/benchmark_results.md`` - results table (paste-ready for the README)
* ``docs/assets/f1_vs_latency.png`` / ``docs/assets/per_nutrient_f1.png``
* ``docs/assets/examples/*.jpg`` and ``docs/assets/real_examples.gif`` (``--gallery``)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ENGINE_LABELS = {
    "gcv": "Google Cloud Vision (reference*)",
    "paddleocr": "PaddleOCR",
    "rapidocr": "RapidOCR (ONNX)",
    "doctr": "docTR",
    "easyocr": "EasyOCR",
    "tesseract": "Tesseract 5",
    "ultralytics": "Ultralytics OCR (YOLO26 + PP-OCRv6)",
    "surya": "Surya",
    "paddleocr-vl": "PaddleOCR-VL (VLM)",
    "olmocr": "olmOCR-2 7B (VLM)",
    "glm-ocr": "GLM-OCR (VLM)",
    "chandra": "Chandra OCR 2 (VLM)",
    "nanonets-ocr": "Nanonets-OCR2 3B (VLM)",
    "deepseek-ocr": "DeepSeek-OCR-2 (VLM)",
    "qwen-vl": "Qwen2.5-VL 3B (VLM)",
    "vlm": "VLM via local server",
}


def _table(summaries: dict) -> str:
    rows = sorted(summaries.values(), key=lambda s: s["f1"], reverse=True)
    lines = [
        "| Engine | F1 | Precision | Recall | Perfect labels | OCR value recall | OCR s/img (median) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in rows:
        lat = s.get("ocr_latency_median_s")
        lines.append(
            f"| {ENGINE_LABELS.get(s['engine'], s['engine'])} | **{s['f1']:.3f}** | "
            f"{s['precision']:.3f} | {s['recall']:.3f} | {s['perfect_document_rate']:.1%} | "
            f"{s['ocr_value_recall']:.1%} | {lat if lat is not None else '-'} |"
        )
    n = rows[0]["documents"] if rows else 0
    split = rows[0].get("split", "test") if rows else "test"
    lines.append("")
    lines.append(f"*{n} images, `{split}` split. \\*The reference row uses the dataset's own "
                 "Google Cloud Vision tokens, so it measures the extractor with near-perfect OCR.*")
    return "\n".join(lines)


def _charts(summaries: dict, assets: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assets.mkdir(parents=True, exist_ok=True)
    made = []
    engines = [s for s in summaries.values() if s.get("ocr_latency_median_s") is not None]
    if engines:
        fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
        for s in engines:
            ax.scatter(s["ocr_latency_median_s"], s["f1"], s=80)
            ax.annotate(ENGINE_LABELS.get(s["engine"], s["engine"]),
                        (s["ocr_latency_median_s"], s["f1"]), textcoords="offset points",
                        xytext=(6, 4), fontsize=9)
        ref = summaries.get("gcv")
        if ref:
            ax.axhline(ref["f1"], ls="--", lw=1, color="grey")
            ax.text(ax.get_xlim()[1], ref["f1"], " GCV reference", va="center", fontsize=8,
                    color="grey")
        ax.set_xlabel("OCR time per image, median (s, CPU)")
        ax.set_ylabel("Field-level F1")
        ax.set_title("Accuracy vs speed")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = assets / "f1_vs_latency.png"
        fig.savefig(p)
        plt.close(fig)
        made.append(p)

    names = sorted({k for s in summaries.values() for k in s.get("per_nutrient_f1", {})})
    if names:
        order = sorted(summaries, key=lambda e: summaries[e]["f1"], reverse=True)
        data = [[summaries[e]["per_nutrient_f1"].get(n, 0.0) for n in names] for e in order]
        fig, ax = plt.subplots(figsize=(8, 0.55 * len(order) + 1.6), dpi=150)
        im = ax.imshow(data, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
        ax.set_yticks(range(len(order)), [ENGINE_LABELS.get(e, e) for e in order])
        for i, row in enumerate(data):
            for j, v in enumerate(row):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v < 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.03)
        ax.set_title("F1 per nutrient")
        fig.tight_layout()
        p = assets / "per_nutrient_f1.png"
        fig.savefig(p)
        plt.close(fig)
        made.append(p)
    return made


def _gallery(engine: str, results: Path, assets: Path, n: int, split: str) -> None:
    from PIL import Image

    from nutrilabel.extraction.extractor import NutritionExtractor
    from nutrilabel.imaging.viz import SYNTHETIC_BOX_ENGINES, draw_result
    from nutrilabel.types import OCRResult

    from .data import iter_hf

    out_dir = assets / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = [json.loads(line) for line in (results / engine / "predictions.jsonl").open()]
    good = {p["idx"] for p in preds if p.get("tp") and len(p["tp"]) >= 6 and not p.get("fp")}
    if engine in SYNTHETIC_BOX_ENGINES:
        print(f"{engine} returns text only (no boxes): gallery skipped")
        return
    frames = []
    extractor = NutritionExtractor()
    for s in iter_hf(split):
        if s.idx not in good:
            continue
        cache = results / "ocr" / engine / f"{s.idx:05d}.json"
        if not cache.exists():
            continue
        ocr = OCRResult.from_dict(json.loads(cache.read_text(encoding="utf-8")))
        img = draw_result(s.image, extractor.extract(ocr))
        img.thumbnail((1400, 900))
        path = out_dir / f"{engine}_{s.idx:03d}_{s.meta.get('barcode', '')}.jpg"
        img.save(path, quality=88)
        frames.append(img)
        if len(frames) >= n:
            break
    if frames:
        w = max(f.width for f in frames)
        h = max(f.height for f in frames)
        canvas = [Image.new("RGB", (w, h), "white") for _ in frames]
        for c, f in zip(canvas, frames):
            c.paste(f, (0, 0))
        canvas[0].save(assets / "real_examples.gif", save_all=True, append_images=canvas[1:],
                       duration=2200, loop=0, optimize=True)
        print(f"gallery: {len(frames)} images in {out_dir}, GIF in {assets / 'real_examples.gif'}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="benchmark/results")
    p.add_argument("--docs", default="docs")
    p.add_argument("--gallery", type=int, default=0, help="number of annotated real examples")
    p.add_argument("--gallery-engine", default=None, help="defaults to the best open-source engine")
    args = p.parse_args(argv)

    results = Path(args.results)
    summaries = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    docs = Path(args.docs)
    table = _table(summaries)
    (docs / "benchmark_results.md").write_text("# Benchmark results\n\n" + table + "\n",
                                                 encoding="utf-8")
    print(table)
    for path in _charts(summaries, docs / "assets"):
        print(f"chart: {path}")

    ranking = [e for e in sorted(summaries, key=lambda e: summaries[e]["f1"], reverse=True)
               if e != "gcv"]
    if ranking:
        print(f"\nRecommended PREFERENCE (src/nutrilabel/engines/__init__.py): {ranking}")
    if args.gallery and ranking:
        engine = args.gallery_engine or ranking[0]
        split = summaries[engine].get("split", "test")
        _gallery(engine, results, docs / "assets", args.gallery, split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
