"""Площадка «Утечка данных» — модуль 2 лабораторных работ Б.1.2.2.

Lecture 6's experiment with handles on it: select the ``k`` best features on the
whole sample, cross-validate, and watch a logistic regression reach 0.757 on
data that is pure noise. The honest pipeline, selecting inside each training
block, reports 0.510 — which is the truth.
"""

from leak_playground.app import main
from leak_playground.experiment import (
    LECTURE_HONEST,
    LECTURE_LEAKY,
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
    "LECTURE_HONEST",
    "LECTURE_LEAKY",
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
