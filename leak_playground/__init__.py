"""Площадка «Утечка данных» — тема «Оценка качества», модуль 2 курса Б.1.2.2.

Select the ``k`` best features on the whole sample, cross-validate, and watch a
logistic regression score far above chance on data that is pure noise. The
honest pipeline, selecting inside each training block, lands near 0.5 — which
is the truth.
"""

from leak_playground.app import main
from leak_playground.experiment import (
    TRUTH,
    Scores,
    Selection,
    all_p_values,
    chance_overlap,
    expected_false_positives,
    honest_score,
    leaky_score,
    scores,
    selection,
)
from leak_playground.noise import INFORMATIVE, NoiseSettings, make_data
from leak_playground.plotting import pvalues_figure, recurrence_figure, scores_figure

__all__ = [
    "INFORMATIVE",
    "TRUTH",
    "NoiseSettings",
    "Scores",
    "Selection",
    "all_p_values",
    "chance_overlap",
    "expected_false_positives",
    "honest_score",
    "leaky_score",
    "main",
    "make_data",
    "pvalues_figure",
    "recurrence_figure",
    "scores",
    "scores_figure",
    "selection",
]
