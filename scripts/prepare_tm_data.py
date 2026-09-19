"""Collect the data files the translation-memory playground ships.

Three of them are derived from files already in this repository; the
embeddings are not. Pretrained vectors cannot be recomputed here — that needs
``torch`` and ``transformers`` and half a gigabyte of weights — so they are
copied from the course folder, which is where they were computed once.

    python scripts/prepare_tm_data.py
    python scripts/prepare_tm_data.py --embeddings /path/to/course/data

Row order is the whole ballgame: row *i* of ``emb_tm.npy`` is segment *i* of
the corpus, and nothing in the file records that. The script checks the shapes
line up and refuses to write a mismatched pair rather than letting the rows
drift apart silently.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORPUS_SOURCE = ROOT / "mt_playground" / "data" / "loc_corpus.csv"
EMBEDDING_SOURCE = Path(
    "/Users/w/MySIlver/дисциплины/Б.1.1.21 Машинное обучение и анализ данных/Основы МО МЛГ/Л.р/data"
)
TARGET = ROOT / "tm_playground" / "data"

# What each query in the lab was built to probe. The texts themselves come from
# the course; these notes name the edit, which is what makes a hit readable.
QUERY_NOTES = {
    "Сохранить все изменения": "добавлено слово",
    "Удалить папку «{name}»?": "заменён термин",
    "Пароль должен содержать не менее 12 символов": "изменено число",
    "Попробуйте бесплатно 14 дней. Карта не нужна.": "изменено число",
    "Резервные копии создаются каждое воскресенье в 03:00 по времени сервера.": "изменено время",
    "Изменения вступают в силу после перезапуска службы.": "заменена концовка",
    "Не удалось подключиться к базе данных": "заменена концовка",
    "Настоящие Правила регулируют использование Платформы.": "заменены термины",
    "Настройки поиска": "переставлены слова",
    "Сохранение изменений": "номинализация",
    "Пожалуйста, очистите кэш и повторите попытку": "короткий сегмент внутри длинного",
    "По вашему запросу результатов нет": "синонимия без общих слов",
    "Максимальный размер вложения — 50 МБ.": "похожего сегмента нет",
}


def build(
    corpus_source: Path = CORPUS_SOURCE,
    embedding_source: Path = EMBEDDING_SOURCE,
    target: Path = TARGET,
) -> dict[str, int]:
    """Write the four data files and return how big each one turned out."""
    target.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_csv(corpus_source)[["id", "type", "en", "ru_ref"]]
    corpus.to_csv(target / "tm_corpus.csv", index=False)

    query_texts = json.loads((embedding_source / "emb_запросы.json").read_text(encoding="utf-8"))
    queries = pd.DataFrame({"text": query_texts})
    queries["note"] = queries["text"].map(QUERY_NOTES).fillna("")
    queries.to_csv(target / "queries.csv", index=False)

    segment_vectors = np.load(embedding_source / "emb_ru_ref.npy")
    query_vectors = np.load(embedding_source / "emb_запросы.npy")

    if len(segment_vectors) != len(corpus):
        raise SystemExit(
            f"{len(segment_vectors)} segment vectors against {len(corpus)} segments — "
            "the rows no longer line up, recompute the embeddings"
        )
    if len(query_vectors) != len(queries):
        raise SystemExit(
            f"{len(query_vectors)} query vectors against {len(queries)} queries — "
            "the rows no longer line up, recompute the embeddings"
        )
    if segment_vectors.shape[1] != query_vectors.shape[1]:
        raise SystemExit("segment and query vectors have different widths")

    shutil.copyfile(embedding_source / "emb_ru_ref.npy", target / "emb_tm.npy")
    shutil.copyfile(embedding_source / "emb_запросы.npy", target / "emb_queries.npy")

    return {
        "segments": len(corpus),
        "queries": len(queries),
        "dimensions": int(segment_vectors.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_SOURCE)
    parser.add_argument("--embeddings", type=Path, default=EMBEDDING_SOURCE)
    args = parser.parse_args()

    counts = build(args.corpus, args.embeddings)
    print(
        f"Wrote {counts['segments']} segments and {counts['queries']} queries "
        f"with {counts['dimensions']}-dimensional vectors into "
        f"{TARGET.relative_to(ROOT)}"
    )
    for path in sorted(TARGET.iterdir()):
        print(f"  {path.name:<18} {path.stat().st_size:>8} bytes")


if __name__ == "__main__":
    main()
