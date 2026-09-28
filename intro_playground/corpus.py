"""The course corpus: 160 localisation segments, four content types.

Only the columns the intro lecture uses travel with this app: the source text,
the reference translation and the content type. The machine translation is lab
3's business and stays in the translation-metrics playground.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
CORPUS_PATH = DATA_DIR / "corpus.csv"

CONTENT_TYPES = ("интерфейс", "документация", "маркетинг", "юридический")


def load_corpus() -> pd.DataFrame:
    """id, type, en, ru_ref — one row per segment, in the course's order."""
    return pd.read_csv(CORPUS_PATH)
