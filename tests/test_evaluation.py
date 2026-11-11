"""Ground truth building, scoring, and a regression guard on real annotated labels.

``tests/fixtures/off_sample.jsonl`` holds 40 annotated labels (tokens, boxes, BIO tags)
from the *train* split of ``openfoodfacts/nutrient-detection-layout``
(CC-BY-SA 3.0, Open Food Facts contributors). No images are included.
"""

import json
from pathlib import Path

from nutrilabel import NutritionExtractor
from nutrilabel.evaluation import (
    Report,
    gold_from_bio,
    numbers_in_text,
    score_document,
    values_match,
    words_from_dataset,
)

FIXTURE = Path(__file__).parent / "fixtures" / "off_sample.jsonl"


def test_gold_from_bio_groups_spans_and_parses_values():
    tokens = ["Fat ", "3,2 ", "g ", "Energy ", "1873 ", "kJ ", "/449 ", "kcal ", "Iron ", "2mg"]
    labels = ["O", "B-FAT_100G", "I-FAT_100G", "O", "B-ENERGY_KJ_100G", "I-ENERGY_KJ_100G",
              "B-ENERGY_KCAL_100G", "I-ENERGY_KCAL_100G", "O", "B-IRON_100G"]
    gold = gold_from_bio(tokens, labels)
    assert set(gold) == {"fat_100g", "energy_kj_100g", "energy_kcal_100g"}  # iron unsupported
    assert (gold["fat_100g"].value, gold["fat_100g"].unit) == (3.2, "g")
    assert gold["energy_kcal_100g"].value == 449


def test_gold_undoes_g_read_as_9():
    gold = gold_from_bio(["0,49"], ["B-SUGARS_100G"])
    assert (gold["sugars_100g"].value, gold["sugars_100g"].unit) == (0.4, "g")
    gold = gold_from_bio(["0,49 ", "g"], ["B-SUGARS_100G", "I-SUGARS_100G"])
    assert gold["sugars_100g"].value == 0.49  # explicit unit: keep as printed


def test_values_match_units_and_tolerance():
    gold = gold_from_bio(["390mg"], ["B-SODIUM_SERVING"])["sodium_serving"]
    assert values_match("sodium_serving", 0.39, "g", gold)
    assert values_match("sodium_serving", 390, "mg", gold)
    assert not values_match("sodium_serving", 39, "mg", gold)


def test_score_document_counts_wrong_value_as_fp_and_fn():
    gold = gold_from_bio(["3g", "5g"], ["B-FAT_100G", "B-SUGARS_100G"])
    s = score_document({"fat_100g": (3.0, "g"), "sugars_100g": (6.0, "g"),
                        "salt_100g": (1.0, "g")}, gold)
    assert s.tp == ["fat_100g"]
    assert sorted(s.fp) == ["salt_100g", "sugars_100g"]
    assert s.fn == ["sugars_100g"]
    assert not s.perfect


def test_report_metrics():
    r = Report()
    gold = gold_from_bio(["3g"], ["B-FAT_100G"])
    r.add(score_document({"fat_100g": (3.0, "g")}, gold))
    r.add(score_document({}, gold))
    r.add_value_recall("Fat 3 g", gold)
    d = r.to_dict()
    assert d["precision"] == 1.0 and d["recall"] == 0.5
    assert d["perfect_document_rate"] == 0.5
    assert d["ocr_value_recall"] == 1.0


def test_numbers_in_text():
    assert {0.0, 4.5, 1873.0} <= numbers_in_text("Og fat 4,5 g 1873 kJ")


def test_regression_on_real_annotated_labels():
    """With the dataset's reference OCR, the extractor must keep F1 >= 0.80 on the
    fixture (it scores ~0.85 on 210 train labels and ~0.83 on the test split)."""
    report = Report()
    extractor = NutritionExtractor()
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        gold = gold_from_bio(d["tokens"], d["labels"])
        result = extractor.extract(words_from_dataset(d["tokens"], d["bboxes"]))
        report.add(score_document(result, gold))
    assert report.documents == 40
    assert report.overall.f1 >= 0.80, report.to_dict()
    assert report.overall.precision >= 0.85, report.to_dict()
