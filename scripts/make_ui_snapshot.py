"""Render a static snapshot of the web UI with real pipeline output (for the docs).

    python scripts/make_ui_snapshot.py --images path/to/photos/*.jpg --engine paddleocr
    python scripts/make_ui_snapshot.py            # rendered sample labels, best engine

Writes docs/assets/ui_snapshot.html - open it in a browser and take a screenshot
(docs/assets/ui.png is the one shown in the README).
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from nutrilabel.engines import available_engines
from nutrilabel.evaluation.synthetic import as_package_photo, render_label
from nutrilabel.ui import analyse_image, page_html


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", nargs="*", default=[])
    p.add_argument("--engine", default=None)
    p.add_argument("--out", default="docs/assets/ui_snapshot.html")
    p.add_argument("--zoom", type=float, default=1.0,
                   help="CSS zoom, to fit the desktop layout in a narrow screenshot window")
    args = p.parse_args(argv)
    installed = available_engines()
    # RapidOCR is the fastest CPU engine: prefer it for a quick snapshot.
    engine = args.engine or ("rapidocr" if "rapidocr" in installed else installed[0])

    items = []
    if args.images:
        for path in args.images:
            items.append(analyse_image(Path(path).read_bytes(), engine, Path(path).name))
    else:
        samples = [
            ("photos/crunchy_oats_fr.jpg", as_package_photo(
                render_label(lang="fr", seed=3), lang="fr", seed=3)),
            ("photos/granola_bites_en.jpg", as_package_photo(
                render_label(lang="en", values={"fat": 18.0, "sugars": 21.0}), lang="en",
                seed=1, color=(34, 82, 150), brand="GRANOLA BITES")),
            ("photos/muesli_de.jpg", as_package_photo(
                render_label(lang="de", serving_g=None, values={"proteins": 11.0, "salt": 0.08}),
                lang="de", seed=2, color=(38, 104, 64), brand="MÜSLI CLASSIC")),
            ("photos/galletas_es.jpg", as_package_photo(
                render_label(lang="es", values={"fat": 12.0, "salt": 0.6}), lang="es", seed=6,
                color=(214, 140, 30), brand="GALLETAS MARÍA")),
        ]
        for name, label in samples:
            buf = io.BytesIO()
            label.image.save(buf, format="JPEG", quality=90)
            items.append(analyse_image(buf.getvalue(), engine, name))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    html = page_html({"engine": engine, "items": items, "active": 0})
    if args.zoom != 1.0:
        # force the desktop layout regardless of the screenshot window width
        css = (f"html {{ zoom: {args.zoom}; }} main {{ grid-template-columns: 260px 1fr !important; }}"
               " #detail { grid-template-columns: minmax(0,2fr) minmax(280px,1fr) !important; }"
               " #files { position: static !important; }")
        html = html.replace("</style>", css + "\n</style>", 1)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
