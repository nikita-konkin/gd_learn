"""Собрать текстовые корпуса playground'ов из корпуса локализации.

Читает корпус, который уже лежит в репозитории, и выписывает русские эталонные
переводы со столбцами `type` и `text`: языковой модели нужен чистый текст, а
классификатору — текст и метка класса. Формат один и тот же, поэтому файл
пишется сразу в оба пакета — каждое приложение на сайте получает отдельный
каталог, так что общий файл всё равно уехал бы туда дважды.

Запускается вручную после правки корпуса:

    python scripts/prepare_text_corpora.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "mt_playground" / "data" / "loc_corpus.csv"
TARGETS = (
    ROOT / "lm_playground" / "data" / "corpus_ru.csv",
    ROOT / "vec_playground" / "data" / "corpus_ru.csv",
)


def build(source: Path = SOURCE, targets: tuple[Path, ...] = TARGETS) -> pd.DataFrame:
    corpus = pd.read_csv(source)
    texts = corpus[["type", "ru_ref"]].rename(columns={"ru_ref": "text"})
    texts["text"] = texts["text"].astype(str).str.strip()
    texts = texts[texts["text"] != ""]

    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        texts.to_csv(target, index=False)
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()

    texts = build(args.source)
    characters = int(texts["text"].str.len().sum())
    print(f"Wrote {len(texts)} segments ({characters} characters) to:")
    for target in TARGETS:
        print(f"  {target.relative_to(ROOT)}")
    print(texts["type"].value_counts().to_string())


if __name__ == "__main__":
    main()
