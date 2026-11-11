"""Layout scenarios for the rule-based extractor, built from hand-placed words."""

from nutrilabel import NutritionExtractor
from nutrilabel.types import BBox, OCRResult, Word

CHAR_W, LINE_H = 12, 20


def words(*rows):
    """rows: (y, [(x, text), ...]) -> list[Word] with simple monospace boxes."""
    out = []
    for y, cells in rows:
        for x, text in cells:
            out.append(Word(text, BBox(x, y, x + CHAR_W * len(text), y + LINE_H)))
    return out


def extract(ws):
    return NutritionExtractor().extract(OCRResult(ws, engine="test")).as_simple_dict()


EU_TWO_COLUMNS = words(
    (0, [(0, "Nutrition"), (120, "information")]),
    (40, [(300, "Per"), (345, "100g"), (500, "Per"), (545, "serving")]),
    (80, [(0, "Energy"), (300, "1873kJ/449kcal"), (500, "562kJ/135kcal")]),
    (120, [(0, "Fat"), (300, "25"), (330, "g"), (500, "7.5"), (545, "g")]),
    (160, [(20, "of"), (50, "which"), (120, "saturates"), (300, "9.5g"), (500, "2.9g")]),
    (200, [(0, "Carbohydrate"), (300, "51g"), (500, "15g")]),
    (240, [(20, "of"), (50, "which"), (120, "sugars"), (300, "32g"), (500, "9.6g")]),
    (280, [(0, "Protein"), (300, "6.4g"), (500, "1.9g")]),
    (320, [(0, "Salt"), (300, "0.45g"), (500, "0.14g")]),
)


def test_eu_table_two_columns():
    got = extract(EU_TWO_COLUMNS)
    assert got == {
        "energy_kj_100g": 1873, "energy_kcal_100g": 449,
        "energy_kj_serving": 562, "energy_kcal_serving": 135,
        "fat_100g": 25, "fat_serving": 7.5,
        "saturated_fat_100g": 9.5, "saturated_fat_serving": 2.9,
        "carbohydrates_100g": 51, "carbohydrates_serving": 15,
        "sugars_100g": 32, "sugars_serving": 9.6,
        "proteins_100g": 6.4, "proteins_serving": 1.9,
        "salt_100g": 0.45, "salt_serving": 0.14,
    }


def test_header_decides_column_order():
    """Serving column printed first: the header, not the position, decides the basis."""
    ws = words(
        (40, [(300, "per"), (345, "portion"), (500, "per"), (545, "100"), (590, "g")]),
        (80, [(0, "Fat"), (300, "3g"), (500, "10g")]),
        (120, [(0, "Protein"), (300, "1.5g"), (500, "5g")]),
    )
    got = extract(ws)
    assert got["fat_100g"] == 10 and got["fat_serving"] == 3
    assert got["proteins_100g"] == 5 and got["proteins_serving"] == 1.5


def test_french_single_column_with_stacked_energy():
    ws = words(
        (0, [(0, "Valeurs"), (100, "nutritionnelles"), (300, "pour"), (360, "100"), (400, "g")]),
        (40, [(0, "Énergie"), (300, "1513"), (360, "kJ")]),
        (62, [(300, "361"), (350, "kcal")]),
        (100, [(0, "Matières"), (110, "grasses"), (300, "15"), (330, "g")]),
        (140, [(10, "dont"), (60, "acides"), (140, "gras"), (200, "saturés"), (300, "9"), (320, "g")]),
        (180, [(0, "Glucides"), (300, "50"), (330, "g")]),
        (220, [(10, "dont"), (60, "sucres"), (300, "34"), (330, "g")]),
        (260, [(0, "Protéines"), (300, "4,47"), (360, "g")]),
        (300, [(0, "Sel"), (300, "0,47"), (360, "g")]),
    )
    got = extract(ws)
    assert got == {
        "energy_kj_100g": 1513, "energy_kcal_100g": 361, "fat_100g": 15,
        "saturated_fat_100g": 9, "carbohydrates_100g": 50, "sugars_100g": 34,
        "proteins_100g": 4.47, "salt_100g": 0.47,
    }


def test_us_nutrition_facts_is_per_serving_and_ignores_percent():
    ws = words(
        (0, [(0, "Nutrition"), (120, "Facts")]),
        (30, [(0, "Serving"), (100, "size"), (160, "1"), (180, "cup"), (240, "(30g)")]),
        (60, [(0, "Amount"), (90, "per"), (140, "serving")]),
        (90, [(0, "Calories"), (300, "120")]),
        (120, [(400, "%"), (420, "Daily"), (490, "Value*")]),
        (150, [(0, "Total"), (70, "Fat"), (120, "2g"), (450, "3%")]),
        (180, [(20, "Saturated"), (140, "Fat"), (190, "0.5g"), (450, "2%")]),
        (210, [(0, "Sodium"), (100, "160mg"), (450, "7%")]),
        (240, [(0, "Total"), (70, "Carbohydrate"), (230, "22g"), (450, "8%")]),
        (270, [(20, "Total"), (90, "Sugars"), (180, "9g")]),
        (300, [(0, "Protein"), (110, "3g")]),
    )
    got = extract(ws)
    assert got == {
        "energy_kcal_serving": 120, "fat_serving": 2, "saturated_fat_serving": 0.5,
        "sodium_serving": 160, "carbohydrates_serving": 22, "sugars_serving": 9,
        "proteins_serving": 3,
    }


def test_running_text_label():
    """Nutrition info printed as a sentence, several nutrients per line."""
    ws = words(
        (0, [(0, "Valeurs"), (100, "nutritionnelles"), (300, "pour"), (360, "100"), (400, "g:")]),
        (30, [(0, "Énergie:"), (110, "521"), (160, "kJ"), (200, "(124"), (260, "kcal)"),
              (330, "-"), (350, "Matières"), (460, "grasses:"), (570, "2,5"), (610, "g")]),
        (60, [(0, "Glucides:"), (120, "17,1"), (175, "g"), (200, "-"), (220, "Protéines:"),
              (350, "6"), (370, "g"), (400, "-"), (420, "Sel:"), (480, "0,55"), (540, "g")]),
    )
    got = extract(ws)
    assert got == {
        "energy_kj_100g": 521, "energy_kcal_100g": 124, "fat_100g": 2.5,
        "carbohydrates_100g": 17.1, "proteins_100g": 6, "salt_100g": 0.55,
    }


def test_rotated_photo_words():
    """Text running top-to-bottom (photo taken sideways) is read in a rotated frame."""
    upright = EU_TWO_COLUMNS
    w_max = max(w.bbox.x1 for w in upright)
    # inverse of the extractor's 90 degree rotation: (x, y) -> (W - y, x)
    rotated = [Word(w.text, BBox(w_max - w.bbox.y1, w.bbox.x0, w_max - w.bbox.y0, w.bbox.x1))
               for w in upright]
    res = NutritionExtractor().extract(OCRResult(rotated, engine="test"))
    assert res.as_simple_dict() == extract(upright)
    assert any("rotated" in w for w in res.warnings)


def test_horizontal_table():
    """Nutrient names side by side, values underneath (common on small packs)."""
    ws = words(
        (0, [(0, "Energy"), (150, "Fat"), (260, "Carbohydrate"), (440, "Protein"), (560, "Salt")]),
        (30, [(0, "1600kJ"), (150, "12g"), (260, "60g"), (440, "8g"), (560, "1.2g")]),
    )
    res = NutritionExtractor().extract(OCRResult(ws, engine="test"))
    assert res.as_simple_dict() == {
        "energy_kj_100g": 1600, "fat_100g": 12, "carbohydrates_100g": 60,
        "proteins_100g": 8, "salt_100g": 1.2,
    }


def test_percent_column_without_percent_signs_is_skipped():
    ws = words(
        (0, [(300, "100"), (345, "g"), (500, "AR")]),
        (40, [(0, "Fat"), (300, "4,9g"), (500, "7")]),
        (80, [(0, "Carbohydrate"), (300, "7,4g"), (500, "3")]),
        (120, [(0, "Protein"), (300, "0,6g"), (500, "1")]),
    )
    got = extract(ws)
    assert got == {"fat_100g": 4.9, "carbohydrates_100g": 7.4, "proteins_100g": 0.6}


def test_ingredient_mentions_do_not_steal_values():
    ws = words(
        (0, [(0, "Ingredients:"), (150, "sugar,"), (230, "salt,"), (300, "palm"), (360, "fat")]),
        (60, [(300, "per"), (350, "100g")]),
        (100, [(0, "Fat"), (300, "20g")]),
        (140, [(0, "Sugars"), (300, "40g")]),
        (180, [(0, "Salt"), (300, "0.3g")]),
    )
    assert extract(ws) == {"fat_100g": 20, "sugars_100g": 40, "salt_100g": 0.3}


def test_empty_and_unrelated_input():
    assert NutritionExtractor().extract([]).nutrients == {}
    res = NutritionExtractor().extract(words((0, [(0, "Hello"), (80, "world"), (200, "42")])))
    assert res.nutrients == {} and res.warnings


def test_result_serialisation():
    res = NutritionExtractor().extract(OCRResult(EU_TWO_COLUMNS, engine="test"))
    d = res.to_dict(include_ocr=True)
    assert d["engine"] == "test"
    assert d["nutrients"]["fat_100g"]["value"] == 25
    assert d["nutrients"]["fat_100g"]["unit"] == "g"
    assert len(d["ocr"]["words"]) == len(EU_TWO_COLUMNS)
    again = OCRResult.from_dict(d["ocr"])
    assert NutritionExtractor().extract(again).as_simple_dict() == res.as_simple_dict()


def test_values_glued_without_spaces():
    """Some recognisers (e.g. RapidOCR) drop spaces: "1873kJ/449kcal562kJ"."""
    ws = [
        Word("Por 100 g", BBox(300, 0, 390, 20)), Word("Por ración (30 g)", BBox(450, 0, 620, 20)),
        Word("Valor energético", BBox(0, 40, 160, 60)),
        Word("1873kJ/449kcal562kJ", BBox(300, 40, 490, 60)), Word("135kcal", BBox(495, 40, 565, 60)),
        Word("Grasas", BBox(0, 80, 60, 100)), Word("12 g", BBox(300, 80, 340, 100)),
        Word("3,6 g", BBox(460, 80, 510, 100)),
    ]
    assert extract(ws) == {
        "energy_kj_100g": 1873, "energy_kcal_100g": 449, "energy_kj_serving": 562,
        "energy_kcal_serving": 135, "fat_100g": 12, "fat_serving": 3.6,
    }


def test_brazilian_label_with_percent_daily_value_column():
    """ANVISA-style table: Portuguese names, "kcal = kJ", "%VD" column without signs."""
    ws = words(
        (0, [(80, "Porção"), (160, "de"), (200, "30"), (240, "g")]),
        (30, [(250, "Quantidade"), (380, "por"), (420, "porção"), (520, "%VD(*)")]),
        (60, [(0, "Valor"), (70, "energético"), (250, "471"), (300, "kcal"), (360, "="),
              (380, "1978"), (440, "kJ"), (530, "24")]),
        (90, [(0, "Carboidratos"), (280, "64"), (310, "g"), (530, "21")]),
        (120, [(0, "Proteínas"), (280, "11"), (310, "g"), (530, "15")]),
        (150, [(0, "Gorduras"), (110, "totais"), (280, "19"), (310, "g"), (530, "35")]),
        (180, [(0, "Gorduras"), (110, "saturadas"), (280, "8,1"), (320, "g"), (530, "37")]),
        (210, [(0, "Gorduras"), (110, "trans"), (280, "0"), (300, "g")]),
        (240, [(0, "Fibra"), (70, "alimentar"), (280, "2,8"), (320, "g"), (530, "11")]),
        (270, [(0, "Sódio"), (280, "1405"), (340, "mg"), (530, "59")]),
    )
    assert extract(ws) == {
        "energy_kcal_serving": 471, "energy_kj_serving": 1978, "carbohydrates_serving": 64,
        "proteins_serving": 11, "fat_serving": 19, "saturated_fat_serving": 8.1,
        "fiber_serving": 2.8, "sodium_serving": 1405,
    }
