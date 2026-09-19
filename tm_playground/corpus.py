"""The translation memory, the incoming queries and the pretrained vectors.

Everything here is read-only and small enough to load on every rerun: 160
segments and 173 vectors of 384 floats.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
CORPUS_PATH = DATA_DIR / "tm_corpus.csv"
QUERIES_PATH = DATA_DIR / "queries.csv"
SEGMENT_VECTORS_PATH = DATA_DIR / "emb_tm.npy"
QUERY_VECTORS_PATH = DATA_DIR / "emb_queries.npy"

CONTENT_TYPES = ("документация", "интерфейс", "маркетинг", "юридический")

# The model the shipped vectors came from, for the interface to name honestly.
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def load_corpus() -> pd.DataFrame:
    """The translation memory: id, content type, English source, Russian target."""
    return pd.read_csv(CORPUS_PATH)


def load_queries() -> pd.DataFrame:
    """The incoming segments from the lab, each with the edit it demonstrates."""
    queries = pd.read_csv(QUERIES_PATH)
    queries["note"] = queries["note"].fillna("")
    return queries


def load_segment_vectors() -> np.ndarray:
    """One pretrained vector per memory segment, row-aligned with the corpus."""
    return np.load(SEGMENT_VECTORS_PATH)


def load_query_vectors() -> np.ndarray:
    """One pretrained vector per lab query, row-aligned with ``load_queries``."""
    return np.load(QUERY_VECTORS_PATH)
