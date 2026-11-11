"""Access to the Open Food Facts nutrient dataset.

Dataset: https://huggingface.co/datasets/openfoodfacts/nutrient-detection-layout
License: CC-BY-SA 3.0 (images and annotations) - Open Food Facts contributors.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_ID = "openfoodfacts/nutrient-detection-layout"
_LABEL_NAMES: list[str] | None = None


@dataclass
class Sample:
    idx: int
    tokens: list[str]
    bboxes: list[list[int]]
    labels: list[str]  # BIO tag names, e.g. "B-FAT_100G"
    meta: dict[str, Any]
    image: Any = None  # PIL image, None when loaded from a token-only JSONL dump


def iter_hf(split: str = "test", limit: int | None = None, streaming: bool = False) -> Iterator[Sample]:
    """Yield samples from the Hugging Face Hub (downloads ~650 MB on first use)."""
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise SystemExit("pip install 'nutrilabel-ocr[benchmark]'  (needs the `datasets` package)") from e
    ds = load_dataset(DATASET_ID, split=split, streaming=streaming)
    names = ds.features["ner_tags"].feature.names
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield Sample(
            idx=i,
            tokens=row["tokens"],
            bboxes=row["bboxes"],
            labels=[names[t] for t in row["ner_tags"]],
            meta=row["meta"],
            image=row["image"],
        )


def iter_jsonl(path: str | Path, limit: int | None = None) -> Iterator[Sample]:
    """Yield samples from a token-only JSONL dump (see ``tests/fixtures``)."""
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            d = json.loads(line)
            yield Sample(
                idx=int(d.get("row_idx", i)),
                tokens=d["tokens"],
                bboxes=d["bboxes"],
                labels=d["labels"],
                meta=d.get("meta", {}),
            )
