"""Which words go in pairs: frequency against pointwise mutual information.

The intro lecture's slide «Поиск правил (ассоциации)» ranks adjacent word pairs
of the reference translations two ways (figure script ``ассоциации`` in the
course repo). The tokenisation, the counting and the PMI formula here are that
script's, and so is the order of ties: pairs with equal PMI stay in the order
they first occur in the corpus.

What the slide cannot show is how little there is to rank. On 160 short
segments only a couple of dozen pairs occur twice or more, so both rankings
reshuffle nearly the same pairs — and below that threshold PMI rewards pairs
seen once, which is the textbook failure of the measure.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import pandas as pd

TOKEN = re.compile(r"[а-яё]+")
LECTURE_MIN_COUNT = 2
LECTURE_TOP = 8


def tokenize(text: str) -> list[str]:
    """Lower-cased Cyrillic words; digits, Latin and punctuation are dropped."""
    return TOKEN.findall(str(text).lower())


def pair_table(texts) -> pd.DataFrame:
    """Every adjacent pair with its count and PMI in bits, in order of first occurrence."""
    words: Counter[str] = Counter()
    pairs: Counter[tuple[str, str]] = Counter()
    for text in texts:
        tokens = tokenize(text)
        words.update(tokens)
        pairs.update(zip(tokens, tokens[1:], strict=False))
    word_total, pair_total = sum(words.values()), sum(pairs.values())
    rows = [
        {
            "pair": f"{first} {second}",
            "count": count,
            "pmi": math.log2((count / pair_total) / ((words[first] / word_total) * (words[second] / word_total))),
        }
        for (first, second), count in pairs.items()
    ]
    return pd.DataFrame(rows, columns=["pair", "count", "pmi"])


def top_by_count(table: pd.DataFrame, min_count: int, top: int) -> pd.DataFrame:
    kept = table[table["count"] >= min_count]
    return kept.sort_values("count", ascending=False, kind="stable").head(top).reset_index(drop=True)


def top_by_pmi(table: pd.DataFrame, min_count: int, top: int) -> pd.DataFrame:
    kept = table[table["count"] >= min_count]
    return kept.sort_values("pmi", ascending=False, kind="stable").head(top).reset_index(drop=True)
