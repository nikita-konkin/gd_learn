"""Загрузка учебного корпуса локализации и расчёт метрик по всем сегментам."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from mt_playground.checks import NEG_RU, formal_checks
from mt_playground.metrics import bleu, chrf, ter

DATA_DIR = Path(__file__).resolve().parent / "data"
CORPUS_PATH = DATA_DIR / "loc_corpus.csv"
SEMANTIC_PATH = DATA_DIR / "semantic_ru_mt.csv"

CONTENT_TYPES = ("интерфейс", "документация", "маркетинг", "юридический")


def load_corpus(negation_rule: re.Pattern = NEG_RU) -> pd.DataFrame:
    """Корпус с метриками baseline-перевода `ru_mt` относительно эталона.

    Колонка `semantic` посчитана заранее многоязычной моделью
    `paraphrase-multilingual-MiniLM-L12-v2`: энкодер в браузер не помещается,
    поэтому в комплекте лежат готовые значения, а не векторы.
    """
    corpus = pd.read_csv(CORPUS_PATH)
    semantic = pd.read_csv(SEMANTIC_PATH)
    corpus = corpus.merge(semantic, on="id", how="left")

    corpus["BLEU"] = [bleu(h, r) for h, r in zip(corpus["ru_mt"], corpus["ru_ref"], strict=True)]
    corpus["chrF"] = [chrf(h, r) for h, r in zip(corpus["ru_mt"], corpus["ru_ref"], strict=True)]
    corpus["TER"] = [ter(h, r) for h, r in zip(corpus["ru_mt"], corpus["ru_ref"], strict=True)]
    corpus["проверки"] = [
        formal_checks(e, r, negation_rule) for e, r in zip(corpus["en"], corpus["ru_mt"], strict=True)
    ]
    corpus["есть_замечания"] = corpus["проверки"].apply(bool)
    return corpus


def segment_label(row) -> str:
    """Подпись сегмента для выпадающего списка."""
    source = row["en"]
    if len(source) > 44:
        source = source[:41] + "…"
    return f"{row['id']} · {row['type']} · {source}"


def coverage_table(corpus: pd.DataFrame, bleu_threshold: float, semantic_threshold: float) -> pd.DataFrame:
    """Сколько сегментов отправит на проверку каждое средство.

    Воспроизводит «цену тревоги» из раздела 8: у порога по BLEU она в десять
    раз выше, чем у формальных проверок.
    """
    total = len(corpus)
    rows = [
        {"средство": f"BLEU < {bleu_threshold:.2f}", "на проверку": int((corpus["BLEU"] < bleu_threshold).sum())},
        {"средство": f"chrF < {bleu_threshold:.2f}", "на проверку": int((corpus["chrF"] < bleu_threshold).sum())},
        {
            "средство": f"семантика < {semantic_threshold:.2f}",
            "на проверку": int((corpus["semantic"] < semantic_threshold).sum()),
        },
        {"средство": "формальные проверки", "на проверку": int(corpus["есть_замечания"].sum())},
    ]
    table = pd.DataFrame(rows)
    table["доля корпуса"] = (table["на проверку"] / total * 100).round(1)
    return table


def blind_spots(corpus: pd.DataFrame, bleu_threshold: float) -> pd.DataFrame:
    """Сегменты, где формальная проверка сработала, а BLEU не забил тревогу.

    Это и есть слепая зона метрики: перевод выглядит хорошо по n-граммам, но
    собираться в продукте он не будет.
    """
    missed = corpus[corpus["есть_замечания"] & (corpus["BLEU"] >= bleu_threshold)]
    return missed[["id", "type", "en", "ru_ref", "ru_mt", "BLEU", "chrF", "semantic", "проверки"]]
