"""Geometry helpers: split OCR boxes into word atoms and group them into text lines.

OCR engines disagree on granularity: Tesseract and docTR return words, while
PaddleOCR, RapidOCR and EasyOCR return whole text lines. Everything downstream
works on *atoms* (single words / values), so line-level boxes are split here,
with each atom's horizontal extent interpolated from its character offsets.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from nutrilabel.types import BBox, Word

# Split on whitespace, "/" and "|" (keeping nothing), and between a label and a glued
# number such as "Energia:1840" or "KJ-440".
_SPLIT_RX = re.compile(
    r"\s+|/|\||(?<=[^\W\d_]):(?=\d)|(?<=[^\W\d_])-(?=\d)"
    # values glued together without spaces, as some recognisers output them:
    # "449kcal562kJ" -> "449kcal" + "562kJ", "12g3,6g" -> "12g" + "3,6g"
    r"|(?<=\dkcal)(?=\d)|(?<=\dkj)(?=\d)|(?<=\dmg)(?=\d)|(?<=\dg)(?=\d)|(?<=\d%)(?=\d)"
    r"|(?<=\d kcal)(?=\d)|(?<=\d kj)(?=\d)",
    re.IGNORECASE,
)


def split_words(words: list[Word]) -> list[Word]:
    """Split multi-token boxes into single-token atoms with interpolated boxes."""
    atoms: list[Word] = []
    for w in words:
        text = w.text.replace("\n", " ")
        if not text.strip():
            continue
        n = len(text)
        pos = 0
        for m in list(_SPLIT_RX.finditer(text)) + [None]:
            end = m.start() if m else n
            piece = text[pos:end]
            if piece.strip():
                lead = len(piece) - len(piece.lstrip())
                start_c = pos + lead
                end_c = start_c + len(piece.strip())
                x0 = w.bbox.x0 + w.bbox.width * start_c / n
                x1 = w.bbox.x0 + w.bbox.width * end_c / n
                atoms.append(Word(piece.strip(), BBox(x0, w.bbox.y0, x1, w.bbox.y1), w.confidence))
            if m is None:
                break
            pos = m.end()
    return atoms


@dataclass
class Line:
    atoms: list[Word] = field(default_factory=list)

    @property
    def bbox(self) -> BBox:
        b = self.atoms[0].bbox
        for a in self.atoms[1:]:
            b = b.union(a.bbox)
        return b

    @property
    def cy(self) -> float:
        return statistics.median(a.bbox.cy for a in self.atoms)

    def text_with_offsets(self) -> tuple[str, list[tuple[int, int]]]:
        """Join atoms with spaces; return the text and each atom's (start, end) offsets."""
        parts: list[str] = []
        offsets: list[tuple[int, int]] = []
        pos = 0
        for a in self.atoms:
            if parts:
                parts.append(" ")
                pos += 1
            parts.append(a.text)
            offsets.append((pos, pos + len(a.text)))
            pos += len(a.text)
        return "".join(parts), offsets


def median_height(words: list[Word], default: float = 10.0) -> float:
    hs = [w.bbox.height for w in words if w.bbox.height > 0]
    return statistics.median(hs) if hs else default


def group_lines(atoms: list[Word]) -> list[Line]:
    """Greedy line grouping by vertical overlap. Returns lines sorted top-to-bottom,
    atoms inside each line sorted left-to-right."""
    lines: list[Line] = []
    for a in sorted(atoms, key=lambda w: (w.bbox.cy, w.bbox.x0)):
        best: Line | None = None
        best_overlap = 0.0
        for line in lines[-6:]:  # only recent lines can overlap a box sorted by cy
            lb = line.atoms[-1].bbox
            ref_h = statistics.median(x.bbox.height for x in line.atoms)
            if abs(a.bbox.cy - line.cy) > 0.5 * max(ref_h, a.bbox.height):
                continue
            overlap = min(a.bbox.y1, lb.y1) - max(a.bbox.y0, lb.y0)
            if overlap > best_overlap:
                best, best_overlap = line, overlap
        if best is None:
            lines.append(Line([a]))
        else:
            best.atoms.append(a)
    for line in lines:
        line.atoms.sort(key=lambda w: w.bbox.x0)
    lines.sort(key=lambda ln: ln.cy)
    return lines
