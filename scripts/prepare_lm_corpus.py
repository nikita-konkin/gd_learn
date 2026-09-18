"""Собрать обучающий текст для языковой модели из корпуса локализации.

Читает корпус, который уже лежит в репозитории, и выписывает только русские
эталонные переводы: языковой модели нужен чистый текст, а не разметка по
столбцам. Выход — `lm_playground/data/corpus_ru.csv` со столбцами `type` и
`text`.

Запускается вручную после правки корпуса:

    python scripts/prepare_lm_corpus.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "mt_playground" / "data" / "loc_corpus.csv"
TARGET = ROOT / "lm_playground" / "data" / "corpus_ru.csv"


def build(source: Path = SOURCE, target: Path = TARGET) -> pd.DataFrame:
    corpus = pd.read_csv(source)
    texts = corpus[["type", "ru_ref"]].rename(columns={"ru_ref": "text"})
    texts["text"] = texts["text"].astype(str).str.strip()
    texts = texts[texts["text"] != ""]

    target.parent.mkdir(parents=True, exist_ok=True)
    texts.to_csv(target, index=False)
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--target", type=Path, default=TARGET)
    args = parser.parse_args()

    texts = build(args.source, args.target)
    characters = int(texts["text"].str.len().sum())
    print(f"Wrote {len(texts)} segments ({characters} characters) to {args.target}")
    print(texts["type"].value_counts().to_string())


if __name__ == "__main__":
    main()
