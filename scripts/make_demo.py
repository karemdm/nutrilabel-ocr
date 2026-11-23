"""Build the README animation from rendered package photos (no dataset download needed).

    python scripts/make_demo.py                    # best installed engine
    python scripts/make_demo.py --engine paddleocr

Each label is shown in three steps: input photo -> OCR boxes -> extracted values.
Output: docs/assets/demo.gif and docs/assets/demo_annotated.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from nutrilabel import NutritionReader
from nutrilabel.evaluation.synthetic import as_package_photo, render_label
from nutrilabel.imaging.fonts import get_font
from nutrilabel.imaging.viz import draw_result


def _caption(img: Image.Image, text: str) -> Image.Image:
    fs = max(18, img.height // 22)
    out = Image.new("RGB", (img.width, img.height + fs * 2), "white")
    out.paste(img, (0, fs * 2))
    ImageDraw.Draw(out).text((fs // 2, fs // 2), text, fill=(25, 25, 25), font=get_font(fs))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", default="auto")
    p.add_argument("--out", default="docs/assets")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    reader = NutritionReader(engine=args.engine)
    samples = [
        ("French pack, photographed at an angle",
         as_package_photo(render_label(lang="fr", seed=3), lang="fr", seed=3)),
        ("German pack, single column",
         as_package_photo(render_label(lang="de", serving_g=None,
                                       values={"proteins": 11.0, "salt": 0.08}),
                          lang="de", seed=2, color=(38, 104, 64), brand="MÜSLI CLASSIC")),
        ("English pack, two columns",
         as_package_photo(render_label(lang="en"), lang="en", seed=1, color=(34, 82, 150),
                          brand="GRANOLA BITES")),
    ]
    frames: list[Image.Image] = []
    for title, label in samples:
        result = reader.read(label.image)
        ok = sum(1 for k, v in label.truth.items()
                 if k in result.nutrients and abs(result.nutrients[k].value - v) < 0.011)
        steps = [
            (label.image, f"1. Input photo - {title}"),
            (draw_result(label.image, result, panel=False), f"2. OCR ({result.ocr.engine})"),
            (draw_result(label.image, result),
             f"3. Extracted {len(result.nutrients)} fields - {ok}/{len(label.truth)} correct"),
        ]
        for img, caption in steps:
            frames.append(_caption(img, caption))
        print(f"{title}: {ok}/{len(label.truth)} correct, {result.ocr.elapsed_s:.2f}s OCR")

    w = max(f.width for f in frames)
    h = max(f.height for f in frames)
    canvas = []
    for f in frames:
        c = Image.new("RGB", (w, h), "white")
        c.paste(f, (0, 0))
        c.thumbnail((960, 960))
        canvas.append(c)
    durations = [1200, 1200, 2600] * len(samples)
    canvas[0].save(out / "demo.gif", save_all=True, append_images=canvas[1:],
                   duration=durations, loop=0, optimize=True)
    frames[2].save(out / "demo_annotated.png")
    print(f"wrote {out / 'demo.gif'} and {out / 'demo_annotated.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
