"""Loading the teaching corpus and scoring every segment in it."""

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
    """The corpus, scored: baseline translation ``ru_mt`` against the reference.

    The ``semantic`` column was computed ahead of time by the multilingual
    model ``paraphrase-multilingual-MiniLM-L12-v2``. The encoder does not fit
    in the browser, so the kit ships the finished numbers rather than vectors.
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
    """The label a segment gets in the dropdown."""
    source = row["en"]
    if len(source) > 44:
        source = source[:41] + "…"
    return f"{row['id']} · {row['type']} · {source}"


def coverage_table(corpus: pd.DataFrame, bleu_threshold: float, semantic_threshold: float) -> pd.DataFrame:
    """How many segments each instrument would send back for review.

    This is the "cost of an alarm" from section 8: a BLEU threshold raises it
    ten times as often as the formal checks do.
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
    """Segments a formal check flagged and BLEU did not.

    This is the measure's blind spot: the translation looks fine by n-grams
    and will not build in the product.
    """
    missed = corpus[corpus["есть_замечания"] & (corpus["BLEU"] >= bleu_threshold)]
    return missed[["id", "type", "en", "ru_ref", "ru_mt", "BLEU", "chrF", "semantic", "проверки"]]
