"""The training text for the language model."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
CORPUS_PATH = DATA_DIR / "corpus_ru.csv"

CONTENT_TYPES = ("интерфейс", "документация", "маркетинг", "юридический")
HELDOUT_SHARE = 0.2


def load_texts(content_types: tuple[str, ...] | list[str] | None = None) -> list[str]:
    """The Russian reference translations, optionally filtered by content type."""
    corpus = pd.read_csv(CORPUS_PATH)
    if content_types:
        corpus = corpus[corpus["type"].isin(list(content_types))]
    return [str(text) for text in corpus["text"].tolist()]


def split(texts: list[str], seed: int = 0, heldout_share: float = HELDOUT_SHARE) -> tuple[list[str], list[str]]:
    """Split into a training part and a held-out part.

    The held-out part exists for exactly one question: is the model
    generalising or memorising. Without it perplexity always looks splendid.
    """
    shuffled = list(texts)
    random.Random(seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * heldout_share))
    return shuffled[cut:], shuffled[:cut]


def corpus_statistics(texts: list[str]) -> dict[str, int]:
    joined = "\n".join(texts)
    return {
        "segments": len(texts),
        "characters": len(joined),
        "alphabet": len(set(joined)),
    }
