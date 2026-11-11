"""Value / unit parsing, including OCR slips seen on real labels."""

from nutrilabel.extraction.layout import split_words
from nutrilabel.extraction.parsing import first_number, parse_unit, parse_value, to_base_unit
from nutrilabel.types import BBox, Word


def _pv(text):
    v = parse_value(text)
    return None if v is None else (v.value, v.unit, v.qualifier)


def test_basic_values_and_units():
    assert _pv("4,5 g") == (4.5, "g", "")
    assert _pv("4.5g") == (4.5, "g", "")
    assert _pv("1873kJ") == (1873.0, "kJ", "")
    assert _pv("449 kcal") == (449.0, "kcal", "")
    assert _pv("390mg") == (390.0, "mg", "")
    assert _pv("12") == (12.0, None, "")


def test_qualifiers():
    assert _pv("<0,5 g") == (0.5, "g", "<")
    assert _pv("< 0.1g") == (0.1, "g", "<")
    assert _pv("≤0,01g") == (0.01, "g", "<")


def test_ocr_confusions():
    assert _pv("Og") == (0.0, "g", "")  # letter O read instead of zero
    assert _pv("O,5g") == (0.5, "g", "")
    assert _pv("521 kl") == (521.0, "kJ", "")  # "kJ" read as "kl"
    assert _pv("(124") == (124.0, None, "")
    assert _pv("0,3 g;") == (0.3, "g", "")


def test_thousands_separator_for_energy():
    assert _pv("1.873 kJ") == (1873.0, "kJ", "")
    assert _pv("2.000 kcal") == (2000.0, "kcal", "")


def test_rejects_non_values():
    for text in ("Fat", "E407", "www.example.com", "", "12345678", "g"):
        assert parse_value(text) is None, text


def test_parse_unit():
    assert parse_unit("g") == "g"
    assert parse_unit("kcal)") == "kcal"
    assert parse_unit("KJ") == "kJ"
    assert parse_unit("%") == "%"
    assert parse_unit("fat") is None


def test_first_number_in_free_text():
    v = first_number("kJ/126 kcal")
    assert (v.value, v.unit) == (126.0, "kcal")
    v = first_number("Energía:1840,87 KJ")
    assert (v.value, v.unit) == (1840.87, "kJ")
    assert first_number("no digits here") is None


def test_to_base_unit():
    assert to_base_unit(390, "mg", "mass") == (0.39, "g")
    assert to_base_unit(2, "g", "mass") == (2, "g")
    assert to_base_unit(449, "kcal", "energy_kcal") == (449, "kcal")


def test_split_words_interpolates_boxes():
    """A line-level box ("Fat 3.2 g") is split into word atoms with partial boxes."""
    atoms = split_words([Word("Fat 3.2 g", BBox(0, 0, 90, 10))])
    assert [a.text for a in atoms] == ["Fat", "3.2", "g"]
    assert atoms[0].bbox.x0 == 0 and atoms[0].bbox.x1 == 30
    assert atoms[1].bbox.x0 == 40 and atoms[2].bbox.x1 == 90


def test_split_words_on_slash_and_glued_labels():
    atoms = split_words([Word("1873kJ/449kcal", BBox(0, 0, 140, 10)),
                         Word("Energía:1840", BBox(0, 20, 120, 30))])
    assert [a.text for a in atoms] == ["1873kJ", "449kcal", "Energía", "1840"]
