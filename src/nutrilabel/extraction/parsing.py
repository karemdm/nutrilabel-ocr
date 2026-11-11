"""Parsing of numeric values such as ``"4,5 g"``, ``"<0.5g"``, ``"1873kJ"`` or ``"Og"``."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Canonical unit spellings.
_UNIT_ALIASES = {
    "g": "g",
    "gr": "g",
    "grs": "g",
    "gram": "g",
    "grams": "g",
    "gramm": "g",
    "mg": "mg",
    "µg": "ug",
    "μg": "ug",
    "ug": "ug",
    "mcg": "ug",
    "kj": "kJ",
    "kl": "kJ",  # frequent OCR confusion
    "k)": "kJ",
    "k": "kJ",  # OCR often drops the "j": "1513 k/361 kcal"
    "kcal": "kcal",
    "kca": "kcal",
    "kcai": "kcal",
    "cal": "kcal",
    "calories": "kcal",
    "ml": "ml",
    "%": "%",
}
_UNIT_RX = r"(?P<unit>kcal|kcai|kca|kj|kl|k\)|mg|mcg|µg|μg|ug|grs|gr|g|ml|cal|%|k)"

_VALUE_RX = re.compile(
    rf"^(?P<qual>[<>≤≥~]|less than|moins de|menos de|meno di)?\s*"
    rf"(?P<num>\d{{1,5}}(?:[.,]\d{{1,3}})?|[.,]\d{{1,3}})\s*{_UNIT_RX}?\.?$"
)
_UNIT_ONLY_RX = re.compile(rf"^{_UNIT_RX}\.?$")

# Characters stripped from the edges of a token before parsing.
_EDGE_PUNCT = "()[]{}:;,*'\"|-–—=_!?"

# OCR confusions in a numeric context: "O" -> 0, "l"/"I" -> 1 ... only applied when the
# token otherwise looks like "<digits><unit>".
_OCR_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "l": "1", "I": "1", "S": "5"})


@dataclass(frozen=True)
class ParsedValue:
    value: float
    unit: str | None
    qualifier: str = ""


def clean_token(text: str) -> str:
    return text.strip().strip(_EDGE_PUNCT).strip()


def parse_unit(text: str) -> str | None:
    """Return the canonical unit if ``text`` is a bare unit (``"g"``, ``"kcal"``...)."""
    t = clean_token(text).lower()
    m = _UNIT_ONLY_RX.match(t)
    if not m:
        return None
    return _UNIT_ALIASES.get(m.group("unit"))


def _to_float(num: str, unit: str | None) -> float:
    num = num.replace(" ", "")
    if "," in num and "." in num:  # "1.234,5" or "1,234.5"
        if num.rfind(",") > num.rfind("."):
            num = num.replace(".", "").replace(",", ".")
        else:
            num = num.replace(",", "")
    elif "," in num:
        num = num.replace(",", ".")
    # "1.873 kJ" is a thousands separator, not 1.873 kJ.
    if unit in ("kJ", "kcal") and re.fullmatch(r"\d\.\d{3}", num):
        num = num.replace(".", "")
    if num.startswith("."):
        num = "0" + num
    return float(num)


def parse_value(text: str) -> ParsedValue | None:
    """Parse a single value token.

    >>> parse_value("4,5 g")
    ParsedValue(value=4.5, unit='g', qualifier='')
    >>> parse_value("<0.5g")
    ParsedValue(value=0.5, unit='g', qualifier='<')
    >>> parse_value("Og")
    ParsedValue(value=0.0, unit='g', qualifier='')
    """
    raw = clean_token(text)
    if not raw:
        return None
    t = raw.lower().replace(" ", " ")
    m = _VALUE_RX.match(t)
    if m is None:
        # Try common OCR digit confusions, but only for short "number+unit" shapes.
        fixed = re.sub(
            r"^[<>≤≥~]?[OoDlIS\d.,]{1,6}(?=\s*(?:mg|g|kj|kcal)?\.?$)",
            lambda mm: mm.group(0).translate(_OCR_DIGIT_FIX),
            raw,
        ).lower()
        m = _VALUE_RX.match(fixed)
        if m is None:
            return None
    unit = _UNIT_ALIASES.get(m.group("unit")) if m.group("unit") else None
    qual = m.group("qual") or ""
    if qual and not qual.startswith(("<", ">", "≤", "≥", "~")):
        qual = "<"
    qual = {"≤": "<", "≥": ">"}.get(qual, qual)
    try:
        value = _to_float(m.group("num"), unit)
    except ValueError:
        return None
    return ParsedValue(value=value, unit=unit, qualifier=qual)


def first_number(text: str) -> ParsedValue | None:
    """Find the first ``number [unit]`` inside free text (used for ground-truth spans)."""
    t = text.replace("\n", " ")
    m = re.search(
        r"([<>≤≥~]?)\s*(\d{1,5}(?:[.,]\d{1,3})?|[.,]\d{1,3}|[Oo](?=\s*m?g\b))\s*"
        r"(kcal|kj|mg|mcg|µg|ug|g|ml|%)?",
        t,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    unit_raw = (m.group(3) or "").lower()
    unit = _UNIT_ALIASES.get(unit_raw) if unit_raw else None
    num = m.group(2)
    if num in ("O", "o"):
        num = "0"
    if unit is None:  # look for a unit a bit further, e.g. "1873 / kJ"
        tail = t[m.end() : m.end() + 8].lower()
        mu = re.search(r"\b(kcal|kj|mg|g)\b", tail)
        if mu:
            unit = _UNIT_ALIASES[mu.group(1)]
    try:
        value = _to_float(num, unit)
    except ValueError:
        return None
    qual = {"≤": "<", "≥": ">"}.get(m.group(1), m.group(1))
    return ParsedValue(value, unit, qual)


def to_base_unit(value: float, unit: str | None, kind: str) -> tuple[float, str | None]:
    """Convert mass values to grams so "390 mg" == "0.39 g". Energy is kept as is."""
    if kind == "mass":
        if unit == "mg":
            return value / 1000.0, "g"
        if unit == "ug":
            return value / 1_000_000.0, "g"
        return value, "g"
    return value, unit
