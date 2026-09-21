"""The corpus, the lab's annotated segments, and what lab 1 measured offline.

All four files are written by ``scripts/prepare_labels_data.py``. The corpus is
the course's 160 segments with BLEU and chrF already attached; the annotations
are the ten segments lab 3 publishes as its worked example; the remedies and
the newsgroup curve are lab 1's section 13, which needs a 14 MB corpus the
browser cannot fetch.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
CORPUS_PATH = DATA_DIR / "corpus.csv"
MQM_PATH = DATA_DIR / "mqm_examples.csv"
REMEDIES_PATH = DATA_DIR / "remedies.csv"
NEWSGROUPS_PATH = DATA_DIR / "newsgroups_curve.csv"

CONTENT_TYPES = ("интерфейс", "документация", "маркетинг", "юридический")


def load_corpus() -> pd.DataFrame:
    """id, type, en, ru_ref, ru_mt, bleu, chrf — one row per segment."""
    return pd.read_csv(CORPUS_PATH)


def load_annotations() -> pd.DataFrame:
    """The lab's ten annotated segments, joined with their text and scores.

    ``category`` and ``severity`` are the lab's own annotation; ``category_second``
    is the second annotator from the lab's agreement cell, who marked only the
    category.
    """
    annotations = pd.read_csv(MQM_PATH)
    corpus = load_corpus()
    return annotations.merge(corpus, on="id", how="left", validate="one_to_one")


def load_remedies() -> pd.DataFrame:
    """Lab 1, section 13: what each remedy did to accuracy."""
    return pd.read_csv(REMEDIES_PATH)


def load_newsgroups_curve() -> pd.DataFrame:
    """Accuracy of lab 1's method on 20 Newsgroups as the training set grows."""
    return pd.read_csv(NEWSGROUPS_PATH)
