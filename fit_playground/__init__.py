"""Площадка «Сложность модели» — модули 2–3 лабораторных работ Б.1.2.2.

Lecture 4's polynomials, lecture 5's neighbours and tree, lecture 5's Ridge and
Lasso: one question — how flexible a model should be — with the error measured
on data the model has not seen.
"""

from fit_playground.app import main
from fit_playground.models import (
    ALPHAS,
    LECTURE_DEGREES,
    Classifier,
    Penalty,
    PolynomialFit,
    diabetes,
    feature_names,
    fit_polynomial,
    fresh_data,
    mean_prediction_error,
    moons,
    neighbours,
    penalty,
    polynomial_data,
    tree,
)

__all__ = [
    "ALPHAS",
    "LECTURE_DEGREES",
    "Classifier",
    "Penalty",
    "PolynomialFit",
    "diabetes",
    "feature_names",
    "fit_polynomial",
    "fresh_data",
    "main",
    "mean_prediction_error",
    "moons",
    "neighbours",
    "penalty",
    "polynomial_data",
    "tree",
]
