"""Layout-aware, rule-based extraction of nutrient values from OCR words.

Pipeline (see ``docs/architecture.md`` for a diagram):

1. split OCR boxes into atoms and group them into lines (:mod:`nutrilabel.extraction.layout`);
2. find *anchors* - nutrient names such as "Fat", "Matières grasses", "Kohlenhydrate";
3. find *value candidates* - tokens such as ``4,5 g``, ``<0.5g``, ``1873kJ``;
4. match each anchor with the values on the same table row (right of the label and
   before the next label on that row), retrying with an estimated vertical shift for
   labels printed slightly above/below their values;
5. decide which value column is "per 100 g" and which is "per serving" from header
   cues ("100 g", "portion", "serving"...) and panel style (EU vs US "Nutrition Facts").

No training is needed, it works with any OCR engine and every decision is explainable.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from nutrilabel.extraction.layout import Line, group_lines, median_height, split_words
from nutrilabel.extraction.nutrients import (
    HEADER_100G,
    HEADER_SERVING,
    IGNORED,
    NOISE_LINE,
    NUTRIENTS,
    USDA_MARKERS,
    normalize_text,
)
from nutrilabel.extraction.parsing import ParsedValue, parse_unit, parse_value
from nutrilabel.types import BBox, ExtractionResult, NutrientValue, OCRResult, Word


def _vertical_ratio(words: list[Word]) -> float:
    """Share of multi-character words whose box is much taller than wide."""
    long = [w for w in words if len(w.text.strip()) >= 3]
    if not long:
        return 0.0
    return sum(w.bbox.height > 1.5 * w.bbox.width for w in long) / len(long)


def _extent(words: list[Word]) -> tuple[float, float]:
    return max(w.bbox.x1 for w in words), max(w.bbox.y1 for w in words)


def _rotate_words(words: list[Word], rot: int) -> list[Word]:
    """Rotate coordinates so that text running top->bottom (90) or bottom->top (270)
    becomes horizontal, left->right."""
    w_max, h_max = _extent(words)
    out = []
    for w in words:
        b = w.bbox
        if rot == 90:  # text reads downwards: (x, y) -> (y, W - x)
            nb = BBox(b.y0, w_max - b.x1, b.y1, w_max - b.x0)
        else:  # text reads upwards: (x, y) -> (H - y, x)
            nb = BBox(h_max - b.y1, b.x0, h_max - b.y0, b.x1)
        out.append(Word(w.text, nb, w.confidence))
    return out


def _rotate_box_back(b: BBox, rot: int, words: list[Word]) -> BBox:
    w_max, h_max = _extent(words)
    if rot == 90:  # inverse of (x, y) -> (y, W - x)
        return BBox(w_max - b.y1, b.x0, w_max - b.y0, b.x1)
    return BBox(b.y0, h_max - b.x1, b.y1, h_max - b.x0)


def _estimate_slope(lines: list[Line], h: float) -> float:
    """Median baseline slope (dy/dx) between neighbouring words of the same line."""
    slopes = []
    for line in lines:
        for a, b in zip(line.atoms, line.atoms[1:]):
            dx = b.bbox.cx - a.bbox.cx
            if dx > 2 * h and abs(b.bbox.height - a.bbox.height) < 0.4 * h:
                slopes.append((b.bbox.cy - a.bbox.cy) / dx)
    if len(slopes) < 5:
        return 0.0
    s = statistics.median(slopes)
    return s if abs(s) > 0.01 else 0.0


_G_AS_9 = re.compile(r"^([<>]?\d+[.,]\d+)9$")


def _repair_g_as_9(values: list[Value]) -> list[Value]:
    """Fix the classic OCR slip "7,5 g" -> "7,59" (space lost, "g" read as "9").

    Only applied to unit-less values with two or more decimals, on labels where
    nearly every other value carries an explicit gram unit.
    """
    with_unit = [v for v in values if v.parsed.unit]
    grams = [v for v in with_unit if v.parsed.unit in ("g", "mg")]
    unitless_dec = [v for v in values if v.parsed.unit is None and re.search(r"[.,]\d", v.raw)]
    suspects = [v for v in unitless_dec if _G_AS_9.match(v.raw.strip())]
    printed_grams = len(grams) >= 3 and len(grams) >= 0.6 * len(with_unit)
    # or: the slip is systematic (most unit-less decimals end with a "9")
    systematic = len(suspects) >= 3 and len(suspects) >= 0.6 * len(unitless_dec)
    if not (printed_grams or systematic):
        return values
    out = []
    for v in values:
        m = _G_AS_9.match(v.raw.strip())
        if v.parsed.unit is None and m:
            fixed = parse_value(m.group(1))
            if fixed is not None:
                v = Value(ParsedValue(fixed.value, "g", fixed.qualifier), v.bbox, v.raw,
                          v.confidence * 0.9)
        out.append(v)
    return out


def _looks_like_pct(values: list[Value]) -> bool:
    """Unit-less small integers, typical of a "% RI" column whose "%" signs were lost."""
    if len(values) < 2:
        return False
    bare_int = [v for v in values if v.parsed.unit is None and float(v.parsed.value).is_integer()
                and v.parsed.value <= 100]
    return len(bare_int) / len(values) >= 0.6


def _swap(b: BBox) -> BBox:
    return BBox(b.y0, b.x0, b.y1, b.x1)


_BARE_100 = re.compile(r"(?<![\d.,])100(?![\d.,%])")

_HEADER_CONTEXT = re.compile(
    r"\b(per|pour|por|para|pro|voor|je|par|fur|na|valeurs?|values?|valori|average|typical|"
    r"moyennes?|medios?|gemiddelde|durchschnitt\w*|nutrition\w*)\b"
)


@dataclass
class Anchor:
    """A nutrient name found on a line of the label (e.g. "Fat"), with its box."""

    nutrient: str  # "fat", "energy", ...
    bbox: BBox
    text: str
    line_idx: int
    energy_hint: str | None = None  # "kJ" / "kcal" when the label itself names the unit


@dataclass
class Value:
    """A parsed number (with unit) found on the label, and whether a label has claimed it."""

    parsed: ParsedValue
    bbox: BBox
    raw: str
    confidence: float
    used: bool = False


@dataclass
class _Match:
    anchor: Anchor
    columns: dict[int, Value] = field(default_factory=dict)  # column index -> value
    dys: list[float] = field(default_factory=list)
    energy: list[Value] = field(default_factory=list)  # energy rows, left to right


class NutritionExtractor:
    """Turn OCR words into structured nutrient values.

    Parameters
    ----------
    row_tolerance:
        Max vertical distance between a label and its value, as a fraction of the
        median text height.
    max_columns:
        Value columns considered per table (one "per 100 g" and one "per serving").
        Columns that look like "% of reference intake" are skipped.
    validate:
        Apply nutritional sanity checks (:mod:`nutrilabel.extraction.validation`), which repair
        values whose decimal separator was lost by the OCR ("95 g" of saturated fat
        under "12.5 g" of fat becomes "9.5 g") and flag the rest in ``warnings``.
    """

    def __init__(self, row_tolerance: float = 0.65, max_columns: int = 2,
                 validate: bool = True) -> None:
        self.row_tolerance = row_tolerance
        self.validate = validate
        self.max_columns = max_columns
        self._centers: dict[int, float] = {}
        self._slope = 0.0
        self._inline = False
        self._pct_cols: set[int] = set()

    # ------------------------------------------------------------------ public API
    def extract(self, ocr: OCRResult | list[Word]) -> ExtractionResult:
        """Extract nutrient values from an :class:`OCRResult` or a plain list of words."""
        words = ocr.words if isinstance(ocr, OCRResult) else ocr
        # Photos taken sideways: OCR engines that keep the page orientation return tall,
        # narrow word boxes. Re-run in rotated coordinates and keep the richer result
        # (rotations are tried first when most words are vertical).
        ratio = _vertical_ratio(words) if words else 0.0
        order = [0]
        if ratio > 0.5:
            order = [90, 270, 0]
        elif ratio > 0.3:
            order = [0, 90, 270]
        best, best_rot = None, 0
        for rot in order:
            candidate = self._extract_words(_rotate_words(words, rot) if rot else words)
            if best is None or len(candidate.nutrients) > len(best.nutrients):
                best, best_rot = candidate, rot
        if best_rot:
            best.warnings.append(f"text read in a frame rotated by {best_rot} degrees")
            for nv in best.nutrients.values():  # report boxes in the original image frame
                if nv.bbox is not None:
                    nv.bbox = _rotate_box_back(nv.bbox, best_rot, words)
        best.ocr = ocr if isinstance(ocr, OCRResult) else None
        if self.validate:
            from nutrilabel.extraction.validation import check

            check(best)
        return best

    def _extract_words(self, words: list[Word]) -> ExtractionResult:
        result = ExtractionResult()
        atoms = split_words(words)
        if not atoms:
            result.warnings.append("no text found")
            return result
        lines = group_lines(atoms)
        h = median_height(atoms)
        doc_text = normalize_text(" ".join(a.text for a in atoms))

        anchors = self._find_anchors(lines)
        if not anchors:
            result.warnings.append("no nutrient labels found")
            return result

        values = self._find_values(lines, h)
        self._slope = _estimate_slope(lines, h)
        self._match_and_emit(lines, anchors, values, doc_text, h, result)

        # Horizontal ("transposed") tables: nutrient names side by side on one row and
        # their values underneath. Swapping x/y turns them into the usual layout.
        per_line: dict[int, int] = {}
        for a in anchors:
            per_line[a.line_idx] = per_line.get(a.line_idx, 0) + 1
        if max(per_line.values()) >= 3:
            alt = ExtractionResult()
            t_anchors = [Anchor(a.nutrient, _swap(a.bbox), a.text, a.line_idx, a.energy_hint)
                         for a in anchors]
            t_values = [Value(v.parsed, _swap(v.bbox), v.raw, v.confidence)
                        for v in self._find_values(lines, h)]
            t_lines = [Line([Word(w.text, _swap(w.bbox), w.confidence) for w in ln.atoms])
                       for ln in lines]
            t_h = median_height([Word("", _swap(a.bbox)) for a in atoms])
            slope, self._slope = self._slope, 0.0
            self._match_and_emit(t_lines, t_anchors, t_values, doc_text, t_h, alt)
            self._slope = slope
            if len(alt.nutrients) >= len(result.nutrients) + 2:
                for nv in alt.nutrients.values():
                    nv.bbox = _swap(nv.bbox) if nv.bbox else None
                alt.warnings.append("horizontal (transposed) table layout")
                result = alt
        if not result.nutrients:
            result.warnings.append("nutrient labels found but no values matched")
        return result

    def _match_and_emit(self, lines: list[Line], anchors: list[Anchor], values: list[Value],
                        doc_text: str, h: float, result: ExtractionResult) -> None:
        matches = self._match_rows(anchors, values, h)
        bases = self._column_bases(lines, anchors, matches, doc_text, h)
        self._emit(matches, bases, result)

    # ------------------------------------------------------------------ anchors
    def _find_anchors(self, lines: list[Line]) -> list[Anchor]:
        anchors: list[Anchor] = []
        for li, line in enumerate(lines):
            text, offsets = line.text_with_offsets()
            norm = normalize_text(text)
            if len(norm) != len(text):  # NFKD may change length; fall back to char map
                norm = "".join(normalize_text(c)[:1] or " " for c in text)
            if NOISE_LINE.search(norm):
                continue
            consumed = [False] * len(norm)
            found: list[tuple[int, int, str]] = []
            for nut in NUTRIENTS:
                for m in nut.pattern.finditer(norm):
                    s, e = m.span()
                    if e <= s or any(consumed[s:e]):
                        continue
                    for i in range(s, e):
                        consumed[i] = True
                    found.append((s, e, nut.name))
            found.sort()
            line_anchors: list[Anchor] = []
            for s, e, name in found:
                idx = [i for i, (a0, a1) in enumerate(offsets) if a0 < e and a1 > s]
                if not idx:
                    continue
                box = line.atoms[idx[0]].bbox
                for i in idx[1:]:
                    box = box.union(line.atoms[i].bbox)
                hint = None
                if name == "energy":
                    hint = self._energy_hint(norm, e, line, idx[-1])
                prev = line_anchors[-1] if line_anchors else None
                # Merge "Fat/Matières grasses/Tuky" (same nutrient, adjacent) into one anchor,
                # unless a number sits between them ("Fat 3 g ... fat 2 g" in running text).
                between = [
                    a for a in line.atoms
                    if prev is not None and prev.bbox.x1 <= a.bbox.cx <= box.x0
                    and parse_value(a.text) is not None
                ]
                if (prev and prev.nutrient == name and not between
                        and box.x0 - prev.bbox.x1 < 6 * box.height):
                    prev.bbox = prev.bbox.union(box)
                    prev.text += " " + norm[s:e]
                    prev.energy_hint = prev.energy_hint or hint
                    continue
                line_anchors.append(Anchor(name, box, norm[s:e], li, hint))
            anchors.extend(line_anchors)
        return anchors

    @staticmethod
    def _energy_hint(norm: str, end: int, line: Line, last_idx: int) -> str | None:
        if "calor" in norm[max(0, end - 9) : end]:
            return "kcal"
        # Unit words printed as part of the label, e.g. "Energy kJ" / "Energy (kcal)".
        for a in line.atoms[last_idx + 1 : last_idx + 3]:
            if parse_value(a.text) is not None and parse_unit(a.text) is None:
                break  # reached a number: the unit belongs to the value, not the label
            unit = parse_unit(a.text.split("/")[0])
            if unit in ("kJ", "kcal"):
                return unit
        return None

    # ------------------------------------------------------------------ values
    def _find_values(self, lines: list[Line], h: float) -> list[Value]:
        values: list[Value] = []
        for line in lines:
            norm_line = normalize_text(" ".join(a.text for a in line.atoms))
            header_line = bool(_HEADER_CONTEXT.search(norm_line))
            atoms = line.atoms
            i = 0
            while i < len(atoms):
                a = atoms[i]
                pv = parse_value(a.text)
                if pv is None:
                    i += 1
                    continue
                box, raw, conf = a.bbox, a.text, a.confidence
                # Merge a following bare unit: "4,5" + "g", "1873" + "kJ".
                if pv.unit is None and i + 1 < len(atoms):
                    nxt = atoms[i + 1]
                    unit = parse_unit(nxt.text)
                    if unit and nxt.bbox.x0 - a.bbox.x1 < 1.5 * max(h, a.bbox.height):
                        pv = ParsedValue(pv.value, unit, pv.qualifier)
                        box = box.union(nxt.bbox)
                        raw = f"{raw} {nxt.text}"
                        conf = min(conf, nxt.confidence)
                        i += 1
                i += 1
                if pv.unit in ("%", "ml"):
                    continue
                if header_line and pv.value == 100 and pv.unit in (None, "g"):
                    continue  # "per 100 g" column header
                values.append(Value(pv, box, raw, conf))
        return _repair_g_as_9(values)

    # ------------------------------------------------------------------ matching
    @staticmethod
    def _compatible(anchor: Anchor, v: Value) -> bool:
        unit = v.parsed.unit
        if anchor.nutrient == "energy":
            return unit in (None, "kJ", "kcal")
        return unit not in ("kJ", "kcal")

    def _candidates(self, anchor: Anchor, anchors: list[Anchor], values: list[Value],
                    h: float, shift: float) -> list[tuple[Value, float]]:
        """Values on the anchor's table row, with their vertical distance (cost)."""
        tol = self.row_tolerance * max(h, anchor.bbox.height)
        cy = anchor.bbox.cy + shift
        # Right boundary: next anchor on the same row (handles inline text and
        # side-by-side multilingual tables).
        right_limit = float("inf")
        for other in anchors:
            if other is anchor or other.bbox.x0 <= anchor.bbox.x1:
                continue
            if abs(other.bbox.cy - anchor.bbox.cy) <= tol:
                right_limit = min(right_limit, other.bbox.x0)
        band_lo, band_hi = cy - tol, cy + tol
        if anchor.nutrient == "energy":
            # kJ and kcal are often stacked on two lines inside the same cell.
            nxt = [o.bbox.cy for o in anchors if o.bbox.cy > anchor.bbox.cy + tol]
            band_hi = min(cy + 2.2 * tol, min(nxt) - tol * 0.5) if nxt else cy + 2.2 * tol
            band_hi = max(band_hi, cy + tol)
            prv = [o.bbox.cy for o in anchors if o.bbox.cy < anchor.bbox.cy - tol]
            band_lo = max(cy - 1.8 * tol, max(prv) + tol * 0.5) if prv else cy - 1.8 * tol
            band_lo = min(band_lo, cy - tol)
        out = []
        for v in values:
            # Skewed photos: follow the text baseline slope from the label to the value.
            skew = self._slope * (v.bbox.cx - anchor.bbox.cx)
            vy = v.bbox.cy - skew
            if not (band_lo <= vy <= band_hi):
                continue
            if v.bbox.x0 < anchor.bbox.x1 - 0.5 * h or v.bbox.x1 > right_limit + 0.5 * h:
                continue
            if not self._compatible(anchor, v):
                continue
            out.append((v, abs(vy - cy) / max(h, 1.0)))
        return out

    @staticmethod
    def _columns(values: list[Value], h: float) -> list[tuple[float, float]]:
        """Merge overlapping x-intervals of values into table columns (left to right)."""
        spans = sorted((v.bbox.x0, v.bbox.x1) for v in values)
        cols: list[list[float]] = []
        for x0, x1 in spans:
            if cols and x0 <= cols[-1][1] + 0.3 * h:
                cols[-1][1] = max(cols[-1][1], x1)
            else:
                cols.append([x0, x1])
        return [(c[0], c[1]) for c in cols]

    def _assign(self, anchors: list[Anchor], values: list[Value], h: float,
                shift: float, done: set[int]) -> list[_Match]:
        """Greedy one-to-one assignment of mass values to (anchor, column) slots."""
        mass = [a for a in anchors if a.nutrient != "energy" and id(a) not in done]
        pairs = []
        for a in mass:
            for v, cost in self._candidates(a, anchors, values, h, shift):
                if not v.used:
                    pairs.append((cost, a, v))
        if not pairs:
            return []
        cand_values = list({id(v): v for _, _, v in pairs}.values())
        columns = self._columns(cand_values, h)

        def col_of(v: Value) -> int:
            for k, (x0, x1) in enumerate(columns):
                if x0 - 0.3 * h <= v.bbox.cx <= x1 + 0.3 * h:
                    return k
            return min(range(len(columns)), key=lambda k: abs(sum(columns[k]) / 2 - v.bbox.cx))

        # Keep columns supported by several labels; drop stray numbers.
        support: dict[int, set[int]] = {}
        for _, a, v in pairs:
            support.setdefault(col_of(v), set()).add(id(a))
        n_rows = len({id(a) for _, a, _ in pairs})
        keep = [k for k, s in support.items() if len(s) >= min(2, n_rows)]
        if not keep:
            keep = list(support)
        keep.sort()
        col_rank = {k: r for r, k in enumerate(keep)}

        slots: dict[tuple[int, int], Value] = {}
        by_anchor: dict[int, _Match] = {}
        for _cost, a, v in sorted(pairs, key=lambda p: p[0]):
            k = col_of(v)
            if v.used or k not in col_rank or (id(a), k) in slots:
                continue
            v.used = True
            slots[(id(a), k)] = v
            m = by_anchor.setdefault(id(a), _Match(a, {}, []))
            m.columns[col_rank[k]] = v
            m.dys.append(v.bbox.cy - a.bbox.cy)
        if done:  # second pass: keep the column geometry of the first pass
            return list(by_anchor.values())
        self._centers = {col_rank[k]: (columns[k][0] + columns[k][1]) / 2 for k in keep}
        col_values: dict[int, list[Value]] = {}
        for (_, k), v in slots.items():
            col_values.setdefault(col_rank[k], []).append(v)
        self._pct_cols = {k for k, vs in col_values.items() if k > 0 and _looks_like_pct(vs)
                          and not _looks_like_pct(col_values.get(0, []))}
        return list(by_anchor.values())

    def _match_rows(self, anchors: list[Anchor], values: list[Value], h: float) -> list[_Match]:
        self._centers = {}
        self._pct_cols = set()
        matches = self._assign(anchors, values, h, 0.0, set())
        # Second pass: labels misaligned with their values by a constant offset.
        if matches:
            shift = statistics.median(d for m in matches for d in m.dys)
            if abs(shift) > 0.15 * h:
                done = {id(m.anchor) for m in matches}
                matches += self._assign(anchors, values, h, shift, done)
        # Irregular layouts (running text): values do not line up in columns, so a
        # label's first value is not necessarily in the first table column.
        ranked = [(k, r) for m in matches
                  for r, (k, _) in enumerate(sorted(m.columns.items()))]
        self._inline = bool(ranked) and sum(k == r for k, r in ranked) / len(ranked) < 0.7
        # Energy rows: kJ and kcal values, ordered left to right per unit.
        for a in anchors:
            if a.nutrient != "energy":
                continue
            cands = [v for v, _ in self._candidates(a, anchors, values, h, 0.0) if not v.used]
            if not cands:
                continue
            cands.sort(key=lambda v: v.bbox.x0)
            for v in cands:
                v.used = True
            matches.append(_Match(a, {}, [v.bbox.cy - a.bbox.cy for v in cands], cands))
        return matches

    # ------------------------------------------------------------------ columns
    def _column_bases(self, lines: list[Line], anchors: list[Anchor], matches: list[_Match],
                      doc_text: str, h: float) -> list[str]:
        """Return the basis ("100g" / "serving") of value column 0, 1, ..."""
        centers: list[float | None] = [self._centers.get(k) for k in range(self.max_columns)]
        if all(c is None for c in centers):  # only energy rows: use their x positions
            xs = sorted(v.bbox.cx for m in matches for v in m.energy)
            if xs:
                centers[0] = xs[0]

        # Header cues above / around the table.
        table_bottom = max(a.bbox.y1 for a in anchors)
        anchor_lines = {a.line_idx for a in anchors}
        cues: list[tuple[float, str]] = []
        for li, line in enumerate(lines):
            if line.bbox.y0 > table_bottom:
                continue
            text, offsets = line.text_with_offsets()
            norm = normalize_text(text)
            if len(norm) != len(text):
                continue
            in_body = li in anchor_lines
            for rx, basis in ((HEADER_100G, "100g"), (_BARE_100, "100g"),
                              (HEADER_SERVING, "serving")):
                if in_body and rx is not HEADER_100G:
                    continue  # "Energy per serving" inside the table body is not a header
                for mt in rx.finditer(norm):
                    idx = [i for i, (s, e) in enumerate(offsets) if s < mt.end() and e > mt.start()]
                    if not idx:
                        continue
                    xs = [line.atoms[i].bbox.cx for i in idx]
                    cues.append((sum(xs) / len(xs), basis))

        bases: list[str | None] = [None] * self.max_columns
        known = [c for c in centers if c is not None]
        if cues and known:
            spacing = abs(known[1] - known[0]) if len(known) > 1 else 8 * h
            limit = max(0.75 * spacing, 4 * h)
            # Each cue votes for its nearest column; each column keeps its nearest cue.
            best_cue: dict[int, tuple[float, str]] = {}
            for x, basis in cues:
                usable = [k for k, c in enumerate(centers)
                          if c is not None and k not in self._pct_cols]
                if not usable:
                    continue
                k = min(usable, key=lambda k: abs(centers[k] - x))
                d = abs(centers[k] - x)
                if d <= limit and (k not in best_cue or d < best_cue[k][0]):
                    best_cue[k] = (d, basis)
            for k, (_, basis) in best_cue.items():
                bases[k] = basis
            n_value_cols = len([c for k, c in enumerate(centers)
                                if c is not None and k not in self._pct_cols])
            if n_value_cols == 1 and bases[0] is None:
                kinds = {b for _, b in cues}
                if len(kinds) == 1:
                    bases[0] = kinds.pop()
        if len(bases) > 1 and bases[0] is not None and bases[0] == bases[1]:
            bases[1] = None  # two columns cannot share a basis; trust the first one
        for k in self._pct_cols:
            if k < len(bases):
                bases[k] = "pct"  # "% of reference intake" column without "%" signs
        usda = bool(USDA_MARKERS.search(doc_text)) and not HEADER_100G.search(doc_text)
        has_100g_cue = any(b == "100g" for _, b in cues)
        if usda and not has_100g_cue:
            # US "Nutrition Facts": every amount is per serving (2nd column = per container).
            return ["serving"] + ["skip"] * (self.max_columns - 1)
        if self._inline:
            # Running text: one basis for the whole label, taken from the wording.
            first = "100g" if has_100g_cue or not cues else cues[0][1]
            other = "serving" if first == "100g" else "100g"
            return [first, other][: self.max_columns] + ["skip"] * (self.max_columns - 2)
        default = ["serving", "100g"] if usda else ["100g", "serving"]
        remaining = [b for b in default if b not in bases]
        return [b if b is not None else remaining.pop(0) if remaining else "skip"
                for b in bases]

    # ------------------------------------------------------------------ output
    def _emit(self, matches: list[_Match], bases: list[str], result: ExtractionResult) -> None:
        best: dict[str, tuple[float, NutrientValue]] = {}

        def offer(key: str, v: Value, score: float) -> None:
            nv = NutrientValue(
                key=key,
                value=v.parsed.value,
                unit=v.parsed.unit,
                raw=v.raw,
                bbox=v.bbox,
                confidence=round(v.confidence * (1.0 if score < 0.5 else 0.8), 4),
                qualifier=v.parsed.qualifier,
            )
            if key not in best or score < best[key][0]:
                best[key] = (score, nv)

        for m in matches:
            h = max(1.0, m.anchor.bbox.height)
            if m.anchor.nutrient == "energy":
                kj = [v for v in m.energy if v.parsed.unit == "kJ"]
                kcal = [v for v in m.energy if v.parsed.unit == "kcal"]
                bare = [v for v in m.energy if v.parsed.unit is None]
                hint = m.anchor.energy_hint
                if bare and hint == "kcal" and not kcal:
                    kcal = bare
                elif bare and hint == "kJ" and not kj:
                    kj = bare
                elif bare and not kj and not kcal and len(bare) >= 2:
                    # "Energy 1873 / 449" without units: the larger is kJ (ratio ~4.184).
                    a, b = bare[0], bare[1]
                    if b.parsed.value and 3.5 < a.parsed.value / b.parsed.value < 4.8:
                        kj, kcal = [a], [b]
                elif bare and len(bare) == 1 and kcal and not kj:
                    kj = bare  # "Energy 1873 / 449 kcal": the unit-less one is kJ
                elif bare and len(bare) == 1 and kj and not kcal:
                    kcal = bare
                score = min((abs(d) for d in m.dys), default=0.0) / h
                e_bases = [b for b in bases if b in ("100g", "serving")]
                for unit_key, vals in (("energy_kj", kj), ("energy_kcal", kcal)):
                    for k, v in enumerate(vals[: len(e_bases)]):
                        offer(f"{unit_key}_{e_bases[k]}", v, score)
            elif m.anchor.nutrient != IGNORED:
                cols = m.columns
                if self._inline:
                    # Running text / irregular layout: rank of the value after its own label.
                    cols = dict(enumerate(v for _, v in sorted(cols.items())))
                for k, v in sorted(cols.items()):
                    if k >= len(bases) or bases[k] not in ("100g", "serving"):
                        continue
                    offer(f"{m.anchor.nutrient}_{bases[k]}", v, abs(v.bbox.cy - m.anchor.bbox.cy) / h)

        result.nutrients = {k: nv for k, (_, nv) in best.items()}
