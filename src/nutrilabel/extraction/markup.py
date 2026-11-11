"""Turn text-only OCR output (markdown / HTML tables, plain lines) into a word grid.

Vision-language OCR models (PaddleOCR-VL, GLM-OCR, olmOCR, Qwen-VL...) return the
label as markdown or HTML instead of word boxes. Laying the cells out on a synthetic
grid - one row per table row, one column slot per cell - lets the very same
layout-aware extractor (and benchmark) handle them.
"""

from __future__ import annotations

import html
import re

from nutrilabel.types import BBox, Word

ROW_H = 30.0  # vertical pitch of the synthetic grid
LINE_H = 20.0  # height of a word box
CHAR_W = 10.0  # width of one character
CELL_GAP = 40.0  # horizontal gap between table cells

_TAG = re.compile(r"<[^>]+>")
_HTML_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.I | re.S)
_HTML_CELL = re.compile(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", re.I | re.S)
_MD_SEPARATOR = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")


def _rows(text: str) -> list[list[str]]:
    """Split the text into rows of cells. Non-table lines become one-cell rows."""
    rows: list[list[str]] = []
    pos = 0
    for m in re.finditer(r"<table\b.*?</table>", text, re.I | re.S):
        rows.extend(_text_rows(text[pos:m.start()]))
        for tr in _HTML_ROW.finditer(m.group(0)):
            cells = [html.unescape(_TAG.sub(" ", c)).strip() for c in _HTML_CELL.findall(tr.group(1))]
            if any(cells):
                rows.append(cells)
        pos = m.end()
    rows.extend(_text_rows(text[pos:]))
    return rows


def _text_rows(text: str) -> list[list[str]]:
    rows = []
    for raw in text.splitlines():
        line = html.unescape(_TAG.sub(" ", raw)).strip()
        if not line or _MD_SEPARATOR.match(line.replace(" ", "")):
            continue
        if line.startswith("|") or line.count("|") >= 2:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows.append(cells)
        else:
            rows.append([line.lstrip("#*- ").strip()])
    return rows


def text_to_words(text: str, confidence: float = 1.0) -> list[Word]:
    """Lay out markdown / HTML / plain text on a grid of :class:`Word` boxes.

    Cell ``j`` of every table row starts at the same x, so value columns line up the
    way they do on the printed label.
    """
    rows = _rows(text)
    n_cols = max((len(r) for r in rows if len(r) > 1), default=1)
    widths = [0.0] * n_cols
    for r in rows:
        if len(r) > 1:
            for j, cell in enumerate(r):
                widths[j] = max(widths[j], len(cell) * CHAR_W)
    starts = [0.0]
    for w in widths[:-1]:
        starts.append(starts[-1] + w + CELL_GAP)

    words: list[Word] = []
    for i, cells in enumerate(rows):
        y = i * ROW_H
        for j, cell in enumerate(cells):
            x0 = starts[j] if len(cells) > 1 and j < len(starts) else 0.0
            for m in re.finditer(r"\S+", cell):
                bx0 = x0 + m.start() * CHAR_W
                words.append(Word(m.group(0), BBox(bx0, y, bx0 + len(m.group(0)) * CHAR_W,
                                                   y + LINE_H), confidence))
    return words
