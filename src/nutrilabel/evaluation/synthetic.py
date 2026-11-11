"""Render synthetic nutrition labels with known values.

Used by the test-suite (end-to-end checks without downloading data) and by
``scripts/make_demo.py`` to build the README animation. Real-world accuracy is
measured on the Open Food Facts dataset, not on these images.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFilter, ImageFont

LABELS = {
    "en": {
        "title": "Nutrition information", "per100": "Per 100 g", "serving": "Per serving (30 g)",
        "energy": "Energy", "fat": "Fat", "saturated_fat": "of which saturates",
        "carbohydrates": "Carbohydrate", "sugars": "of which sugars", "fiber": "Fibre",
        "proteins": "Protein", "salt": "Salt",
    },
    "fr": {
        "title": "Valeurs nutritionnelles", "per100": "Pour 100 g", "serving": "Par portion (30 g)",
        "energy": "Énergie", "fat": "Matières grasses", "saturated_fat": "dont acides gras saturés",
        "carbohydrates": "Glucides", "sugars": "dont sucres", "fiber": "Fibres alimentaires",
        "proteins": "Protéines", "salt": "Sel",
    },
    "de": {
        "title": "Nährwerte", "per100": "pro 100 g", "serving": "pro Portion (30 g)",
        "energy": "Brennwert", "fat": "Fett", "saturated_fat": "davon gesättigte Fettsäuren",
        "carbohydrates": "Kohlenhydrate", "sugars": "davon Zucker", "fiber": "Ballaststoffe",
        "proteins": "Eiweiß", "salt": "Salz",
    },
    "es": {
        "title": "Información nutricional", "per100": "Por 100 g", "serving": "Por ración (30 g)",
        "energy": "Valor energético", "fat": "Grasas", "saturated_fat": "de las cuales saturadas",
        "carbohydrates": "Hidratos de carbono", "sugars": "de los cuales azúcares",
        "fiber": "Fibra alimentaria", "proteins": "Proteínas", "salt": "Sal",
    },
}

DEFAULT_VALUES = {
    "energy_kj": 1873.0, "energy_kcal": 449.0, "fat": 25.0, "saturated_fat": 9.5,
    "carbohydrates": 51.0, "sugars": 32.0, "fiber": 3.1, "proteins": 6.4, "salt": 0.45,
}


@dataclass
class SyntheticLabel:
    image: Image.Image
    truth: dict[str, float]  # e.g. {"fat_100g": 25.0, "fat_serving": 7.5, ...}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    from nutrilabel.imaging.fonts import get_font

    return get_font(size, bold)


def _fmt(value: float, lang: str) -> str:
    text = f"{value:g}"
    return text.replace(".", ",") if lang in ("fr", "de", "es") else text


def render_label(
    values: dict[str, float] | None = None,
    lang: str = "en",
    serving_g: float | None = 30.0,
    scale: float = 1.0,
    noise: bool = False,
    seed: int = 0,
) -> SyntheticLabel:
    """Draw an EU-style table. ``serving_g=None`` draws a single "per 100 g" column."""
    vals = dict(DEFAULT_VALUES, **(values or {}))
    lab = LABELS[lang]
    rows = ["energy", "fat", "saturated_fat", "carbohydrates", "sugars", "fiber", "proteins", "salt"]
    fs = round(26 * scale)
    font, bold = _font(fs), _font(fs, bold=True)
    pad = round(24 * scale)
    row_h = round(fs * 1.9)
    col1 = round(520 * scale)
    col2 = round(760 * scale)
    width = round((1020 if serving_g else 800) * scale)
    height = pad * 2 + row_h * (len(rows) + 3)
    img = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([4, 4, width - 5, height - 5], outline=(0, 0, 0), width=max(2, round(3 * scale)))
    y = pad
    d.text((pad, y), lab["title"], font=bold, fill=(0, 0, 0))
    y += row_h
    d.text((col1, y), lab["per100"], font=bold, fill=(0, 0, 0))
    if serving_g:
        d.text((col2, y), lab["serving"].replace("30", f"{serving_g:g}"), font=bold, fill=(0, 0, 0))
    y += row_h
    d.line([pad, y - row_h * 0.25, width - pad, y - row_h * 0.25], fill=(0, 0, 0), width=2)

    truth: dict[str, float] = {}
    factor = (serving_g or 0) / 100.0
    for row in rows:
        indent = round(30 * scale) if row in ("saturated_fat", "sugars") else 0
        d.text((pad + indent, y), lab[row], font=font, fill=(0, 0, 0))
        if row == "energy":
            kj, kcal = vals["energy_kj"], vals["energy_kcal"]
            d.text((col1, y), f"{kj:g} kJ / {kcal:g} kcal", font=font, fill=(0, 0, 0))
            truth["energy_kj_100g"], truth["energy_kcal_100g"] = kj, kcal
            if serving_g:
                skj, skcal = round(kj * factor), round(kcal * factor)
                d.text((col2, y), f"{skj:g} kJ / {skcal:g} kcal", font=font, fill=(0, 0, 0))
                truth["energy_kj_serving"], truth["energy_kcal_serving"] = skj, skcal
        else:
            v = vals[row]
            d.text((col1, y), f"{_fmt(v, lang)} g", font=font, fill=(0, 0, 0))
            truth[f"{row}_100g"] = v
            if serving_g:
                sv = round(v * factor, 2)
                d.text((col2, y), f"{_fmt(sv, lang)} g", font=font, fill=(0, 0, 0))
                truth[f"{row}_serving"] = sv
        y += row_h

    if noise:
        rnd = random.Random(seed)
        img = img.rotate(rnd.uniform(-2.5, 2.5), resample=Image.BICUBIC, expand=True,
                         fillcolor=(235, 230, 220))
        img = img.filter(ImageFilter.GaussianBlur(radius=0.6 * scale))
    return SyntheticLabel(img, truth)


_INGREDIENTS = {
    "en": "INGREDIENTS: wholegrain oat flakes (58%), sugar, sunflower oil, honey (4%), "
          "rice flour, almonds (3%), salt, natural flavouring. May contain traces of milk.",
    "fr": "INGRÉDIENTS : flocons d'avoine complète (58 %), sucre, huile de tournesol, "
          "miel (4 %), farine de riz, amandes (3 %), sel, arôme naturel. Traces de lait.",
    "de": "ZUTATEN: Vollkornhaferflocken (58 %), Zucker, Sonnenblumenöl, Honig (4 %), "
          "Reismehl, Mandeln (3 %), Salz, natürliches Aroma. Kann Spuren von Milch enthalten.",
    "es": "INGREDIENTES: copos de avena integral (58 %), azúcar, aceite de girasol, miel (4 %), "
          "harina de arroz, almendras (3 %), sal, aroma natural. Puede contener leche.",
}


def as_package_photo(label: SyntheticLabel, lang: str = "en", seed: int = 0,
                     color: tuple[int, int, int] = (196, 54, 46),
                     brand: str = "CRUNCHY OATS") -> SyntheticLabel:
    """Place a rendered table on a printed package and photograph it badly: coloured
    carton, brand and ingredient text, barcode, perspective, uneven light, JPEG noise.

    The nutrient values (``truth``) are unchanged.
    """
    import io

    import numpy as np

    rnd = random.Random(seed)
    tab = label.image
    W, H = tab.width + 260, tab.height + 520
    pack = Image.new("RGB", (W, H), color)
    d = ImageDraw.Draw(pack)
    # white print area with ingredients, the table and a barcode
    d.rounded_rectangle([60, 60, W - 60, H - 60], radius=28, fill=(250, 248, 242))
    big = _font(64, bold=True)
    d.text((100, 90), brand, font=big, fill=color)
    small = _font(22)
    words, line, y = _INGREDIENTS[lang].split(), "", 185
    for w in words:
        if d.textlength(line + " " + w, font=small) > W - 220:
            d.text((100, y), line.strip(), font=small, fill=(60, 60, 60))
            line, y = "", y + 30
        line += " " + w
    d.text((100, y), line.strip(), font=small, fill=(60, 60, 60))
    pack.paste(tab, (130, y + 60))
    bx, by = W - 380, H - 150
    for i in range(70):
        if rnd.random() < 0.55:
            d.rectangle([bx + i * 4, by, bx + i * 4 + rnd.choice((1, 2, 3)), by + 70], fill=(20, 20, 20))
    d.text((bx, by + 74), "5 012345 678900", font=_font(18), fill=(20, 20, 20))

    # perspective: the photo is taken slightly from the side
    dx, dy = rnd.uniform(0.03, 0.06) * W, rnd.uniform(0.02, 0.05) * H
    canvas = Image.new("RGB", (int(W * 1.12), int(H * 1.12)), (92, 84, 74))
    ox, oy = int(W * 0.06), int(H * 0.06)
    canvas.paste(pack, (ox, oy))
    src = [(ox, oy), (ox + W, oy), (ox + W, oy + H), (ox, oy + H)]
    dst = [(ox + dx, oy), (ox + W - dx * 0.3, oy + dy), (ox + W, oy + H - dy), (ox, oy + H)]
    coeffs = _perspective_coeffs(dst, src)
    canvas = canvas.transform(canvas.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC,
                              fillcolor=(92, 84, 74))
    canvas = canvas.rotate(rnd.uniform(-3, 3), resample=Image.BICUBIC, fillcolor=(92, 84, 74))
    # uneven lighting + a soft glare
    arr = np.asarray(canvas).astype(np.float32)
    yy, xx = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
    light = 0.82 + 0.25 * (xx / arr.shape[1]) - 0.08 * (yy / arr.shape[0])
    gx, gy = rnd.uniform(0.55, 0.8) * arr.shape[1], rnd.uniform(0.2, 0.4) * arr.shape[0]
    glare = 38 * np.exp(-(((xx - gx) / (0.18 * arr.shape[1])) ** 2 + ((yy - gy) / (0.12 * arr.shape[0])) ** 2))
    arr = np.clip(arr * light[..., None] + glare[..., None], 0, 255).astype(np.uint8)
    photo = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=0.7))
    buf = io.BytesIO()
    photo.save(buf, format="JPEG", quality=72)
    return SyntheticLabel(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), label.truth)


def _perspective_coeffs(src, dst):
    """Coefficients for PIL's PERSPECTIVE transform mapping ``dst`` points to ``src``."""
    import numpy as np

    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    a = np.array(rows, dtype=float)
    b = np.array([c for p in dst for c in p], dtype=float)
    return np.linalg.solve(a, b).tolist()
