"""Agreement between two annotators, and how much of it is luck.

Cohen's kappa is computed by hand here rather than taken from sklearn, because
its two ingredients are the point: how often the annotators matched, and how
often they would have matched by chance given how often each uses each label.
The tests check the result against ``sklearn.metrics.cohen_kappa_score``.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Lab 3, section 6: five kinds of error and "no error".
ERROR_CATEGORIES = ("точность", "беглость", "терминология", "стиль", "разметка")
NO_ERROR = "нет"
CATEGORIES = (*ERROR_CATEGORIES, NO_ERROR)


@dataclass(frozen=True)
class Agreement:
    observed: float  # share of segments where the two labels match
    expected: float  # share that would match if both labelled at random
    kappa: float  # NaN when chance agreement is total and kappa is undefined
    count: int


def agreement(first: Sequence[str], second: Sequence[str]) -> Agreement:
    """Observed agreement, chance agreement and Cohen's kappa.

    Chance agreement assumes each annotator keeps their own label frequencies
    but picks independently of the other: for every label, the product of the
    two frequencies, summed. Kappa is how much of the gap between chance and
    perfect agreement the annotators actually covered.
    """
    if len(first) != len(second):
        raise ValueError("both annotators must label the same segments")
    count = len(first)
    if count == 0:
        raise ValueError("nothing to compare")

    observed = sum(a == b for a, b in zip(first, second, strict=True)) / count
    first_counts, second_counts = Counter(first), Counter(second)
    expected = sum(first_counts[label] * second_counts[label] for label in first_counts) / count**2

    kappa = math.nan if math.isclose(expected, 1.0) else (observed - expected) / (1.0 - expected)
    return Agreement(observed=observed, expected=expected, kappa=kappa, count=count)


def agreement_matrix(first: Sequence[str], second: Sequence[str], labels: Sequence[str] = CATEGORIES) -> pd.DataFrame:
    """Counts: rows are the first annotator's labels, columns the second's."""
    used = [label for label in labels if label in set(first) | set(second)]
    matrix = pd.DataFrame(0, index=used, columns=used)
    for a, b in zip(first, second, strict=True):
        matrix.loc[a, b] += 1
    return matrix


def label_shares(share_no_error: float, error_categories: int = len(ERROR_CATEGORIES)) -> np.ndarray:
    """A label distribution: ``share_no_error`` for "нет", the rest split evenly."""
    if not 0.0 <= share_no_error <= 1.0:
        raise ValueError("a share lies between 0 and 1")
    return np.array([share_no_error] + [(1.0 - share_no_error) / error_categories] * error_categories)


def chance_agreement(share_no_error: float, error_categories: int = len(ERROR_CATEGORIES)) -> float:
    """How often two independent annotators with the same habits match."""
    shares = label_shares(share_no_error, error_categories)
    return float((shares**2).sum())


def expected_raw_agreement(true_agreement: float, share_no_error: float) -> float:
    """Raw agreement of annotators who truly agree on a share of segments.

    The model: on a ``true_agreement`` share of segments the second annotator
    sees what the first saw; on the rest each labels independently with the
    same habits. Raw agreement is then the true share plus luck on the rest —
    and kappa strips the luck back out, so in expectation kappa equals
    ``true_agreement`` whatever the label distribution.
    """
    if not 0.0 <= true_agreement <= 1.0:
        raise ValueError("a share lies between 0 and 1")
    chance = chance_agreement(share_no_error)
    return true_agreement + (1.0 - true_agreement) * chance


def chance_sweep(true_agreement: float, shares: Sequence[float]) -> pd.DataFrame:
    """Raw agreement, chance agreement and kappa across label distributions."""
    rows = []
    for share in shares:
        chance = chance_agreement(share)
        raw = expected_raw_agreement(true_agreement, share)
        kappa = math.nan if math.isclose(chance, 1.0) else (raw - chance) / (1.0 - chance)
        rows.append({"share_no_error": float(share), "raw": raw, "chance": chance, "kappa": kappa})
    return pd.DataFrame(rows)


# Lab 3, the table after the agreement cell (Landis and Koch's reading).
KAPPA_READINGS = (
    (0.20, "согласия практически нет"),
    (0.40, "слабое"),
    (0.60, "умеренное"),
    (0.80, "существенное"),
)


def kappa_reading(kappa: float) -> str:
    if math.isnan(kappa):
        return "не определена: оба разметчика всегда ставят одну и ту же метку"
    for upper, reading in KAPPA_READINGS:
        if kappa <= upper:
            return reading
    return "почти полное"
