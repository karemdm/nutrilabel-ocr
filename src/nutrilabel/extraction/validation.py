"""Nutritional sanity checks applied after extraction.

OCR often drops a decimal separator ("9,5 g" -> "95 g"). Nutrition tables obey
simple rules that expose these slips:

* no mass nutrient exceeds 100 g per 100 g;
* saturated fat <= fat, sugars <= carbohydrates;
* kJ ~= 4.184 x kcal.

When a value breaks a rule and dividing it by 10 (or 100) fixes it, the value is
repaired and a warning is added; otherwise only a warning is added.
"""

from __future__ import annotations

import math

from nutrilabel.extraction.parsing import parse_value
from nutrilabel.types import ExtractionResult, NutrientValue

_MASS = ("fat", "saturated_fat", "carbohydrates", "sugars", "fiber", "proteins", "salt", "sodium")
_PARENT = {"saturated_fat": "fat", "sugars": "carbohydrates"}


def _grams(nv: NutrientValue) -> float:
    if nv.unit == "mg":
        return nv.value / 1000
    if nv.unit == "ug":
        return nv.value / 1_000_000
    return nv.value


def _candidates(nv: NutrientValue) -> list[float]:
    """Plausible re-readings of a value, most likely first."""
    raw = nv.raw.strip().rstrip(";,.")
    out: list[float] = []
    if nv.unit is None and raw[-1:] == "9" and raw[:-1].strip()[-1:].isdigit():
        # "32 g" read as "329": the unit was taken for a digit
        stripped = parse_value(raw[:-1])
        if stripped is not None:
            out.append(stripped.value)
            if not any(c in raw for c in ".,") and stripped.value >= 10:
                out.append(stripped.value / 10)  # "9,5 g" read as "959"
    if not any(c in raw for c in ".,") and nv.value >= 10:
        out += [nv.value / 10, nv.value / 100]  # decimal separator lost
    return out


def _repair(nv: NutrientValue, ok, result: ExtractionResult, why: str) -> None:
    for candidate in _candidates(nv):
        if ok(candidate):
            result.warnings.append(
                f"{nv.key}: {nv.value:g} looked implausible ({why}); read as {candidate:g} "
                "(OCR probably lost a decimal separator or read the unit 'g' as '9')"
            )
            nv.value = round(candidate, 4)
            nv.unit = nv.unit or "g"
            nv.confidence = round(nv.confidence * 0.7, 4)
            return
    result.warnings.append(f"{nv.key}: {nv.value:g} looks implausible ({why})")
    nv.confidence = round(nv.confidence * 0.5, 4)


def check(result: ExtractionResult, repair: bool = True) -> ExtractionResult:
    n = result.nutrients
    for basis in ("100g", "serving"):
        # 1. at most 100 g per 100 g
        if basis == "100g":
            for name in _MASS:
                nv = n.get(f"{name}_100g")
                if nv is not None and nv.unit in (None, "g") and nv.value > 100:
                    if repair:
                        _repair(nv, lambda v: v <= 100, result, "more than 100 g per 100 g")
                    else:
                        result.warnings.append(f"{nv.key}: more than 100 g per 100 g")
        # 2. sub-nutrient <= parent
        for child, parent in _PARENT.items():
            c, p = n.get(f"{child}_{basis}"), n.get(f"{parent}_{basis}")
            if c is None or p is None or c.unit not in (None, "g") or p.unit not in (None, "g"):
                continue
            if _grams(c) > _grams(p) * 1.02 + 0.05:
                if repair:
                    _repair(c, lambda v, p=p: v <= _grams(p) * 1.02 + 0.05, result,
                            f"greater than {parent} {p.value:g}")
                else:
                    result.warnings.append(f"{c.key} greater than {p.key}")
        # 3. energy units agree
        kj, kcal = n.get(f"energy_kj_{basis}"), n.get(f"energy_kcal_{basis}")
        if kj is not None and kcal is not None and kcal.value > 0:
            ratio = kj.value / kcal.value
            if not math.isclose(ratio, 4.184, rel_tol=0.08):
                result.warnings.append(
                    f"energy_{basis}: {kj.value:g} kJ vs {kcal.value:g} kcal do not match"
                )
    return result
