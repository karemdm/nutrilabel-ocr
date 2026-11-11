"""Visualisation helpers: draw OCR boxes and extracted values on the photo."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from nutrilabel.extraction.nutrients import NUTRIENT_NAMES
from nutrilabel.imaging.fonts import get_font
from nutrilabel.imaging.preprocess import ImageInput, load_image
from nutrilabel.types import ExtractionResult

# One colour per nutrient (colour-blind friendly palette, Okabe-Ito + extras).
_PALETTE = [
    (230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66), (0, 114, 178),
    (213, 94, 0), (204, 121, 167), (120, 94, 240), (100, 143, 255), (254, 97, 0),
]
COLORS = {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(NUTRIENT_NAMES)}

_DISPLAY = {
    "energy_kj": "Energy (kJ)",
    "energy_kcal": "Energy (kcal)",
    "fat": "Fat",
    "saturated_fat": "- of which saturates",
    "carbohydrates": "Carbohydrates",
    "sugars": "- of which sugars",
    "fiber": "Fibre",
    "proteins": "Protein",
    "salt": "Salt",
    "sodium": "Sodium",
}
_ORDER = list(_DISPLAY)
#: Engines returning text only; their word boxes live on a synthetic grid.
SYNTHETIC_BOX_ENGINES = {"paddleocr-vl", "vlm", "hf-vlm", "olmocr", "glm-ocr", "chandra",
                         "nanonets-ocr", "deepseek-ocr", "qwen-vl"}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return get_font(size, bold)


def draw_result(image: ImageInput, result: ExtractionResult, show_ocr: bool = True,
                panel: bool = True) -> Image.Image:
    """Return a copy of the image with OCR boxes (grey) and extracted values (coloured),
    plus an optional side panel listing the values."""
    img = load_image(image).copy()
    draw = ImageDraw.Draw(img, "RGBA")
    scale = max(img.size) / 1000
    lw = max(2, round(3 * scale))
    synthetic = result.ocr is not None and result.ocr.engine in SYNTHETIC_BOX_ENGINES
    if synthetic:  # text-only VLM output: no image positions to draw, panel only
        return _with_panel(img, result) if panel else img
    if show_ocr and result.ocr is not None:
        for w in result.ocr.words:
            b = w.bbox
            draw.rectangle([b.x0, b.y0, b.x1, b.y1], outline=(120, 120, 120, 160), width=1)
    for key, nv in result.nutrients.items():
        if nv.bbox is None:
            continue
        name = key.rsplit("_", 1)[0]
        color = COLORS.get(name, (255, 0, 0))
        b = nv.bbox
        draw.rectangle([b.x0 - 2, b.y0 - 2, b.x1 + 2, b.y1 + 2], outline=color + (255,),
                       fill=color + (60,), width=lw)
    if not panel:
        return img
    return _with_panel(img, result)


def _with_panel(img: Image.Image, result: ExtractionResult) -> Image.Image:
    rows = []
    for name in _ORDER:
        v100 = result.nutrients.get(f"{name}_100g")
        vsv = result.nutrients.get(f"{name}_serving")
        if v100 is None and vsv is None:
            continue
        cells = [f"{nv.qualifier}{nv.value:g} {nv.unit or ''}".strip() if nv else "-"
                 for nv in (v100, vsv)]
        rows.append((name, _DISPLAY[name], cells))

    h = img.height
    fs = max(15, round(h / 32))
    font, bold = _font(fs), _font(round(fs * 1.15), bold=True)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def tw(text: str, f=font) -> float:
        return probe.textlength(text, font=f)

    gap = fs * 1.2
    name_w = max([tw(r[1]) for r in rows] + [tw("Nutrient")]) + fs * 1.2
    col_w = max([tw(c) for r in rows for c in r[2]] + [tw("serving")]) + gap
    engine = result.ocr.engine if result.ocr else "?"
    title = f"Extracted ({engine})"
    width = round(max(name_w + 2 * col_w, tw(title, bold)) + 2 * fs)
    height = max(h, round(fs * (4.5 + 1.5 * len(rows))))
    out = Image.new("RGB", (img.width + width, height), "white")
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    x = img.width + fs
    y = fs
    d.text((x, y), title, fill=(20, 20, 20), font=bold)
    y += round(fs * 2)
    d.text((x + name_w, y), "100 g", fill=(90, 90, 90), font=font)
    d.text((x + name_w + col_w, y), "serving", fill=(90, 90, 90), font=font)
    y += round(fs * 1.6)
    for name, label, cells in rows:
        color = COLORS.get(name, (0, 0, 0))
        d.rectangle([x, y + fs * 0.25, x + fs * 0.6, y + fs * 0.85], fill=color)
        d.text((x + fs, y), label, fill=(30, 30, 30), font=font)
        for i, txt in enumerate(cells):
            d.text((x + name_w + i * col_w, y), txt, fill=(30, 30, 30), font=font)
        y += round(fs * 1.5)
    return out
