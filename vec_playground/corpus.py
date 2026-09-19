"""The classification corpus: the same text lab 1 works on."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
CORPUS_PATH = DATA_DIR / "corpus_ru.csv"

CONTENT_TYPES = ("документация", "интерфейс", "маркетинг", "юридический")


def load_corpus() -> pd.DataFrame:
    """The Russian reference translations and the content type of each."""
    return pd.read_csv(CORPUS_PATH)


def texts_and_labels() -> tuple[pd.Series, pd.Series]:
    corpus = load_corpus()
    return corpus["text"], corpus["type"]
