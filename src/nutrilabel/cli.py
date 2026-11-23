"""Command line interface (no GUI needed).

Exit status of ``read``: 0 when every image gave at least one value, 1 when an image failed
or had no values, 2 when no image was found or the options are invalid.

    nutrilabel read label.jpg                       # best installed engine
    nutrilabel read label.jpg --engine tesseract --json
    nutrilabel read label.jpg --annotate out.png    # save a visual check
    nutrilabel read photos/ --csv results.csv --out-dir annotated/   # a whole folder
    nutrilabel ui                                   # local web interface
    nutrilabel engines                              # what is installed?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nutrilabel import __version__


def _cmd_engines(_: argparse.Namespace) -> int:
    """Print every engine, whether its library is installed, and how to install it."""
    from nutrilabel.engines import ENGINES, PREFERENCE, VLM_ENGINES

    print(f"{'engine':<13} {'installed':<10} install with")
    for name in PREFERENCE + VLM_ENGINES:
        if name == VLM_ENGINES[0]:
            print("-- vision-language OCR (GPU recommended) --")
        cls = ENGINES[name]
        ok = "yes" if cls.is_available() else "no"
        how = (f"pip install \"nutrilabel-ocr[{cls.pip_extra}]\"" if cls.pip_extra
               else "set NUTRILABEL_VLM_MODEL / NUTRILABEL_VLM_URL (local model server)")
        print(f"{name:<13} {ok:<10} {how}")
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    """Start the local web interface."""
    from nutrilabel.ui import serve

    serve(host=args.host, port=args.port, engine=args.engine, open_browser=not args.no_browser)
    return 0


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def collect_images(paths: list[str]) -> list[Path]:
    """Expand files and folders (recursively) into a sorted list of image paths."""
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out.extend(sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS))
        elif p.exists():
            out.append(p)
        else:
            print(f"warning: {p} not found", file=sys.stderr)
    return out


def _cmd_read(args: argparse.Namespace) -> int:
    """Read one or more images/folders; print tables or JSON and optionally write CSV/PNGs."""
    from nutrilabel.pipeline import NutritionReader

    images = collect_images(args.images)
    if not images:
        print("no images found", file=sys.stderr)
        return 2
    if args.annotate and len(images) > 1:
        print("error: --annotate saves one image; use --out-dir for several inputs",
              file=sys.stderr)
        return 2
    reader = NutritionReader(
        engine=args.engine,
        languages=args.lang.split(",") if args.lang else None,
        max_side=args.max_side,
        auto_rotate=not args.no_rotate,
    )
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    status = 0
    for i, path in enumerate(images, 1):
        try:
            result = reader.read(path)
        except Exception as e:  # noqa: BLE001 - keep going through a folder
            print(f"error: {path}: {type(e).__name__}: {e}", file=sys.stderr)
            status = 1
            continue
        if args.json:
            payload = result.to_dict(include_ocr=args.with_ocr)
            payload["image"] = str(path)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        elif not args.quiet:
            _print_table(str(path), result)
        if len(images) > 1:
            print(f"[{i}/{len(images)}] {path.name}: {len(result.nutrients)} fields",
                  file=sys.stderr)
        if args.annotate or out_dir:
            from nutrilabel.imaging.viz import draw_result

            target = (out_dir / f"{path.stem}.annotated.png") if out_dir else Path(args.annotate)
            draw_result(path, result).save(target)
        row = {"image": str(path), "engine": result.ocr.engine if result.ocr else ""}
        row.update({k: v.value for k, v in result.nutrients.items()})
        row["warnings"] = " | ".join(result.warnings)
        rows.append(row)
        if not result.nutrients:
            status = 1
    if args.csv:
        _write_csv(args.csv, rows)
        print(f"results written to {args.csv}", file=sys.stderr)
    return status


def _write_csv(path: str, rows: list[dict]) -> None:
    import csv

    from nutrilabel.extraction.nutrients import SUPPORTED_KEYS

    fields = ["image", "engine", *SUPPORTED_KEYS, "warnings"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _print_table(path: str, result) -> None:
    """Print one image's extracted fields as a plain-text table."""
    engine = result.ocr.engine if result.ocr else "?"
    secs = result.ocr.elapsed_s if result.ocr else 0.0
    print(f"\n{path}  (engine: {engine}, OCR {secs:.2f}s)")
    if not result.nutrients:
        print("  no nutrition values found", *result.warnings, sep="\n  - ")
        return
    print(f"  {'field':<24}{'value':>10}  unit")
    for key, nv in sorted(result.nutrients.items()):
        print(f"  {key:<24}{nv.qualifier + format(nv.value, 'g'):>10}  {nv.unit or ''}")
    for w in result.warnings:
        print(f"  note: {w}")


def main(argv: list[str] | None = None) -> int:
    """Run the ``nutrilabel`` command; returns the process exit status."""
    parser = argparse.ArgumentParser(
        prog="nutrilabel", description="Read nutrition facts tables from product photos."
    )
    parser.add_argument("--version", action="version", version=f"nutrilabel {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_read = sub.add_parser("read", help="extract nutrient values from images or folders")
    p_read.add_argument("images", nargs="+", help="image files and/or folders (recursive)")
    p_read.add_argument("-e", "--engine", default="auto",
                        help="auto | paddleocr | rapidocr | surya | doctr | easyocr | tesseract | "
                             "ultralytics[:n|s|m|l|x] | paddleocr-vl | olmocr | glm-ocr | chandra | nanonets-ocr | "
                             "deepseek-ocr | qwen-vl | vlm  (see `nutrilabel engines`)")
    p_read.add_argument("--lang", default=None,
                        help="comma separated languages for engines that need them, e.g. en,fr,de "
                             "(default: en,fr,de,es,it,nl,pt)")
    p_read.add_argument("--max-side", type=int, default=1600,
                        help="longest image side sent to the OCR, in pixels (default: 1600)")
    p_read.add_argument("--no-rotate", action="store_true", help="disable the rotation retry")
    p_read.add_argument("--json", action="store_true", help="print JSON instead of a table")
    p_read.add_argument("--with-ocr", action="store_true", help="include raw OCR words in JSON")
    p_read.add_argument("--annotate", metavar="PNG", help="save an annotated image (single input only)")
    p_read.add_argument("--out-dir", metavar="DIR", help="save one annotated PNG per image here")
    p_read.add_argument("--csv", metavar="FILE", help="write one row per image to a CSV file")
    p_read.add_argument("-q", "--quiet", action="store_true", help="no per-image table")
    p_read.set_defaults(func=_cmd_read)

    p_ui = sub.add_parser("ui", help="open the local web interface (single image or folder)")
    p_ui.add_argument("-e", "--engine", default="auto",
                      help="engine preselected in the page (default: auto, best installed classic engine)")
    p_ui.add_argument("--host", default="127.0.0.1",
                      help="address to bind (default: 127.0.0.1; use 0.0.0.0 only in a container)")
    p_ui.add_argument("--port", type=int, default=8765, help="port (default: 8765)")
    p_ui.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    p_ui.set_defaults(func=_cmd_ui)

    p_eng = sub.add_parser("engines", help="list OCR engines and whether they are installed")
    p_eng.set_defaults(func=_cmd_engines)

    args = parser.parse_args(argv)
    return int(args.func(args))


def main_ui() -> int:
    """Entry point of the ``nutrilabel-ui`` command."""
    return main(["ui", *sys.argv[1:]])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
