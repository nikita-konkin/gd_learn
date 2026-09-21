"""MQM penalties, and what happens when BLEU is asked to do their job.

Lab 3, section 6 weighs errors by the cost of their consequences — one critical
error outweighs twenty-five minor ones — and section 7 shows BLEU disagreeing
with that weighing in both directions. This module turns section 7's single
cut at the median into any cut at all, to show that no BLEU threshold agrees
with the annotation on the lab's own ten segments.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

CRITICAL, MAJOR, MINOR, NONE = "критическая", "серьёзная", "незначительная", "нет"
SEVERITIES = (CRITICAL, MAJOR, MINOR, NONE)

# The MQM weights the lab uses.
SEVERITY_WEIGHTS = {CRITICAL: 25, MAJOR: 5, MINOR: 1, NONE: 0}

SEVERITY_WHEN = {
    CRITICAL: "продукт сломан или смысл противоположный",
    MAJOR: "пользователь будет введён в заблуждение",
    MINOR: "заметно специалисту, задачу не срывает",
    NONE: "ошибок не обнаружено",
}

# Outcomes of accepting a segment on its BLEU alone.
MISSED_CRITICAL = "принят с критической ошибкой"
REJECTED_CORRECT = "отклонён без ошибок"
ACCEPTED = "принят"
REJECTED = "отклонён"


def penalties(severities: Sequence[str], weights: Mapping[str, int] = SEVERITY_WEIGHTS) -> np.ndarray:
    return np.array([weights[severity] for severity in severities], dtype=float)


def lab_flags(annotations: pd.DataFrame, median: float) -> pd.Series:
    """Section 7's rule, as the lab states it, at the corpus median.

    A critical error BLEU rates above the median went unnoticed; an error-free
    translation BLEU rates below it was punished. Everything else is unflagged.
    """
    flags = pd.Series("", index=annotations.index, dtype=object)
    flags[(annotations["severity"] == CRITICAL) & (annotations["bleu"] > median)] = "метрика не заметила"
    flags[(annotations["severity"] == NONE) & (annotations["bleu"] < median)] = "метрика наказала"
    return flags


def acceptance(annotations: pd.DataFrame, threshold: float) -> pd.Series:
    """What accepting on ``BLEU >= threshold`` does to each annotated segment."""
    accepted = annotations["bleu"] >= threshold
    outcome = pd.Series(np.where(accepted, ACCEPTED, REJECTED), index=annotations.index, dtype=object)
    outcome[accepted & (annotations["severity"] == CRITICAL)] = MISSED_CRITICAL
    outcome[~accepted & (annotations["severity"] == NONE)] = REJECTED_CORRECT
    return outcome


def acceptance_errors(annotations: pd.DataFrame, threshold: float) -> dict[str, int]:
    outcome = acceptance(annotations, threshold)
    return {
        MISSED_CRITICAL: int((outcome == MISSED_CRITICAL).sum()),
        REJECTED_CORRECT: int((outcome == REJECTED_CORRECT).sum()),
        "accepted": int(outcome.isin([ACCEPTED, MISSED_CRITICAL]).sum()),
    }


def acceptance_sweep(annotations: pd.DataFrame, thresholds: Sequence[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        errors = acceptance_errors(annotations, threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "missed_critical": errors[MISSED_CRITICAL],
                "rejected_correct": errors[REJECTED_CORRECT],
                "accepted": errors["accepted"],
            }
        )
    return pd.DataFrame(rows)


def clean_thresholds(sweep: pd.DataFrame) -> pd.DataFrame:
    """Thresholds that pass no critical error and turn away no correct translation."""
    return sweep[(sweep["missed_critical"] == 0) & (sweep["rejected_correct"] == 0)]


def rank_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    """Spearman's coefficient: Pearson's on average ranks, ties shared.

    Done by hand because pandas hands Spearman to scipy, and nothing else in
    this app needs scipy's import to succeed.
    """
    ranks_first = pd.Series(first, dtype=float).rank()
    ranks_second = pd.Series(second, dtype=float).rank()
    return float(np.corrcoef(ranks_first, ranks_second)[0, 1])


def penalty_by_type(annotations: pd.DataFrame, weights: Mapping[str, int] = SEVERITY_WEIGHTS) -> pd.DataFrame:
    """Errors and total penalty per content type — the summary of lab 3, task 4."""
    frame = annotations.assign(penalty=penalties(annotations["severity"], weights))
    frame = frame.assign(is_error=frame["severity"] != NONE)
    summary = frame.groupby("type").agg(errors=("is_error", "sum"), penalty=("penalty", "sum"))
    return summary.sort_values("penalty", ascending=False).reset_index()
