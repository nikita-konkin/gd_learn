"""Обучающий текст для языковой модели."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
CORPUS_PATH = DATA_DIR / "corpus_ru.csv"

CONTENT_TYPES = ("интерфейс", "документация", "маркетинг", "юридический")
HELDOUT_SHARE = 0.2


def load_texts(content_types: tuple[str, ...] | list[str] | None = None) -> list[str]:
    """Русские эталонные переводы, при желании отфильтрованные по типу."""
    corpus = pd.read_csv(CORPUS_PATH)
    if content_types:
        corpus = corpus[corpus["type"].isin(list(content_types))]
    return [str(text) for text in corpus["text"].tolist()]


def split(texts: list[str], seed: int = 0, heldout_share: float = HELDOUT_SHARE) -> tuple[list[str], list[str]]:
    """Разделить на обучающую и отложенную части.

    Отложенная часть нужна ровно для одного вопроса: модель обобщает или
    запоминает. Без неё перплексия всегда выглядит прекрасно.
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
