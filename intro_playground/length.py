"""How much room a translation takes: regression of Russian length on English.

The intro lecture's slide «Сколько места займёт перевод» fits one line to the
whole corpus (figure script ``регрессия`` in the course repo) and reads a single
expansion figure off it. The point of this module is the part a single number
hides: the expansion differs by content type more than the correction itself,
and on interface strings — the ones with a hard width limit — it is largest.

Two averages are easy to confuse. The slide's «+10 %» is the mean of the
per-segment ratios; the ratio of total characters is smaller, because long
segments expand less. A price per character follows the second; a button that
must fit follows neither, but the spread.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# The rule of thumb the lecture quotes and tests: "Russian is about fifteen per
# cent longer than English".
RULE_OF_THUMB = 0.15
FIT_QUANTILE = 0.9


def lengths(corpus: pd.DataFrame) -> pd.DataFrame:
    """Characters in the source and in the reference, and their ratio, per segment."""
    en = corpus["en"].astype(str).str.len().astype(float)
    ru = corpus["ru_ref"].astype(str).str.len().astype(float)
    return pd.DataFrame({"type": corpus["type"], "en": en, "ru": ru, "ratio": ru / en}, index=corpus.index)


@dataclass(frozen=True)
class Line:
    slope: float
    intercept: float
    r2: float

    def predict(self, en_length: float) -> float:
        return self.slope * en_length + self.intercept


def fit_line(table: pd.DataFrame) -> Line:
    """Least squares, ``ru = slope · en + intercept``, as ``np.polyfit`` in the figure script."""
    slope, intercept = np.polyfit(table["en"], table["ru"], 1)
    r = float(np.corrcoef(table["en"], table["ru"])[0, 1])
    return Line(slope=float(slope), intercept=float(intercept), r2=r * r)


def mean_expansion(table: pd.DataFrame) -> float:
    """Mean of per-segment ratios, minus one: the slide's «+10 %»."""
    return float(table["ratio"].mean() - 1)


def total_expansion(table: pd.DataFrame) -> float:
    """Total Russian characters over total English characters, minus one."""
    return float(table["ru"].sum() / table["en"].sum() - 1)


def expansion_by_type(table: pd.DataFrame, types) -> pd.DataFrame:
    """Per content type: segments, mean, median, total and the 90th-percentile expansion."""
    rows = []
    for content_type in types:
        part = table[table["type"] == content_type]
        rows.append(
            {
                "type": content_type,
                "segments": len(part),
                "mean": mean_expansion(part),
                "median": float(part["ratio"].median() - 1),
                "total": total_expansion(part),
                "high": float(part["ratio"].quantile(FIT_QUANTILE) - 1),
            }
        )
    return pd.DataFrame(rows)


def overflow_share(table: pd.DataFrame, margin: float, types) -> pd.Series:
    """Share of each type's segments longer than ``en · (1 + margin)``: space reserved, space overrun."""
    over = table["ru"] > table["en"] * (1 + margin)
    return pd.Series({content_type: float(over[table["type"] == content_type].mean()) for content_type in types})
