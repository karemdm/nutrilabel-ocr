"""Ground truth construction and metrics for the Open Food Facts nutrient dataset.

The dataset (``openfoodfacts/nutrient-detection-layout``, CC-BY-SA 3.0) annotates
OCR tokens with BIO tags such as ``B-FAT_100G`` / ``I-FAT_100G``. We turn every
tagged span into a ground-truth field, e.g. ``{"fat_100g": (3.2, "g")}``, and
score predictions field by field.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

from nutrilabel.extraction.nutrients import SUPPORTED_KEYS, nutrient_kind
from nutrilabel.extraction.parsing import first_number, to_base_unit
from nutrilabel.types import BBox, ExtractionResult, OCRResult, Word


@dataclass(frozen=True)
class GoldField:
    key: str
    value: float
    unit: str | None
    raw: str


def gold_from_bio(tokens: list[str], labels: list[str]) -> dict[str, GoldField]:
    """Build ``{key: GoldField}`` from BIO-tagged tokens.

    Only keys in :data:`SUPPORTED_KEYS` are returned; when the same key is
    annotated twice (rare duplicates) the first occurrence wins.
    """
    spans: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] | None = None
    for tok, lab in zip(tokens, labels):
        if lab == "O":
            current = None
            continue
        prefix, _, name = lab.partition("-")
        if prefix == "B" or current is None or current[0] != name:
            current = (name, [tok])
            spans.append(current)
        else:
            current[1].append(tok)
    gold: dict[str, GoldField] = {}
    for name, toks in spans:
        key = name.lower()
        if key not in SUPPORTED_KEYS or key in gold:
            continue
        raw = "".join(toks).strip()
        pv = first_number(raw)
        if pv is None:
            continue
        # The dataset's tokens come from Google Cloud Vision and keep its OCR slips,
        # e.g. "0,4 g" transcribed as "0,49". Undo that one so that engines which read
        # the label correctly are not penalised (see docs/benchmark.md).
        m = _G_AS_9.match(raw.strip(" ;,.\n"))
        if pv.unit is None and m:
            fixed = first_number(m.group(1))
            if fixed is not None:
                pv = type(pv)(fixed.value, "g", fixed.qualifier)
        gold[key] = GoldField(key, pv.value, pv.unit, raw)
    return _physically_plausible(gold)


def _physically_plausible(gold: dict[str, GoldField]) -> dict[str, GoldField]:
    """Apply the extractor's sanity rules to the annotations too.

    The reference OCR sometimes lost a decimal separator ("42,9 g" -> "429") and the
    annotators tagged that token as is. 429 g of fat per 100 g is impossible, so such
    fields are corrected the same way (42.9) - otherwise an engine that reads the
    label correctly would be scored as wrong. See docs/benchmark.md.
    """
    from nutrilabel.extraction.validation import check
    from nutrilabel.types import ExtractionResult, NutrientValue

    res = ExtractionResult(nutrients={
        k: NutrientValue(k, g.value, g.unit, g.raw) for k, g in gold.items()
    })
    check(res, repair=True)
    return {k: GoldField(k, nv.value, nv.unit, gold[k].raw) for k, nv in res.nutrients.items()}


_G_AS_9 = re.compile(r"^([<>]?\d+[.,]\d+)9$")


def words_from_dataset(tokens: list[str], bboxes: list[list[int]]) -> OCRResult:
    """Wrap the dataset's own (Google Cloud Vision) tokens as an :class:`OCRResult`.

    Used as the *reference OCR* run: it isolates extraction errors from OCR errors.
    Boxes are in the dataset's normalised 0-1000 space.
    """
    words = [
        Word(t.strip(), BBox(*map(float, b)), 1.0) for t, b in zip(tokens, bboxes) if t.strip()
    ]
    return OCRResult(words=words, engine="gcv-reference", width=1000, height=1000)


def _key_kind(key: str) -> str:
    name = key.rsplit("_", 1)[0]
    return nutrient_kind(name)


def values_match(key: str, pred_value: float, pred_unit: str | None, gold: GoldField) -> bool:
    """Numeric comparison after unit normalisation (``0.39 g`` == ``390 mg``)."""
    kind = _key_kind(key)
    pu = pred_unit if pred_unit not in ("kJ", "kcal") or kind != "mass" else None
    gu = gold.unit if gold.unit not in ("kJ", "kcal") or kind != "mass" else None
    p, _ = to_base_unit(pred_value, pu, kind)
    g, _ = to_base_unit(gold.value, gu, kind)
    return math.isclose(p, g, rel_tol=0.01, abs_tol=0.0051)


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class DocScore:
    tp: list[str] = field(default_factory=list)
    fp: list[str] = field(default_factory=list)  # predicted key absent from gold, or wrong value
    fn: list[str] = field(default_factory=list)  # gold key missed or wrong value

    @property
    def perfect(self) -> bool:
        return not self.fp and not self.fn


def score_document(pred: ExtractionResult | dict[str, tuple[float, str | None]],
                   gold: dict[str, GoldField]) -> DocScore:
    """Field-level scoring. A wrong value counts as one FP *and* one FN."""
    if isinstance(pred, ExtractionResult):
        pmap = {k: (v.value, v.unit) for k, v in pred.nutrients.items()}
    else:
        pmap = dict(pred)
    s = DocScore()
    for key, (val, unit) in pmap.items():
        if key in gold and values_match(key, val, unit, gold[key]):
            s.tp.append(key)
        else:
            s.fp.append(key)
    for key in gold:
        if key not in s.tp:
            s.fn.append(key)
    return s


@dataclass
class Report:
    overall: Counts = field(default_factory=Counts)
    per_nutrient: dict[str, Counts] = field(default_factory=lambda: defaultdict(Counts))
    documents: int = 0
    perfect_documents: int = 0
    value_recall_hits: int = 0
    value_recall_total: int = 0

    def add(self, s: DocScore) -> None:
        self.documents += 1
        self.perfect_documents += int(s.perfect)
        self.overall.tp += len(s.tp)
        self.overall.fp += len(s.fp)
        self.overall.fn += len(s.fn)
        for k in s.tp:
            self.per_nutrient[k.rsplit("_", 1)[0]].tp += 1
        for k in s.fp:
            self.per_nutrient[k.rsplit("_", 1)[0]].fp += 1
        for k in s.fn:
            self.per_nutrient[k.rsplit("_", 1)[0]].fn += 1

    def add_value_recall(self, ocr_text: str, gold: dict[str, GoldField]) -> None:
        """OCR-only metric: is each gold number present somewhere in the OCR text?"""
        found = numbers_in_text(ocr_text)
        for g in gold.values():
            self.value_recall_total += 1
            if any(math.isclose(g.value, x, rel_tol=1e-6, abs_tol=1e-6) for x in found):
                self.value_recall_hits += 1

    @property
    def value_recall(self) -> float:
        return self.value_recall_hits / self.value_recall_total if self.value_recall_total else 0.0

    def to_dict(self) -> dict:
        return {
            "documents": self.documents,
            "precision": round(self.overall.precision, 4),
            "recall": round(self.overall.recall, 4),
            "f1": round(self.overall.f1, 4),
            "tp": self.overall.tp,
            "fp": self.overall.fp,
            "fn": self.overall.fn,
            "perfect_document_rate": round(self.perfect_documents / self.documents, 4)
            if self.documents
            else 0.0,
            "ocr_value_recall": round(self.value_recall, 4),
            "per_nutrient_f1": {
                k: round(c.f1, 4) for k, c in sorted(self.per_nutrient.items())
            },
        }


_NUM_RX = re.compile(r"\d{1,5}(?:[.,]\d{1,3})?")


def numbers_in_text(text: str) -> set[float]:
    out: set[float] = set()
    for m in _NUM_RX.finditer(text.replace("O g", "0 g").replace("Og", "0g")):
        try:
            out.add(float(m.group(0).replace(",", ".")))
        except ValueError:
            pass
    return out
