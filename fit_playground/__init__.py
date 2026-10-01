"""Площадка «Сложность модели» — тема «Задачи и модели», модули 2–3 курса Б.1.2.2.

Polynomials, neighbours and a tree, Ridge and Lasso: one question — how
flexible a model should be — with the error measured on data the model has not
seen.
"""

from fit_playground.app import main
from fit_playground.models import (
    ALPHAS,
    SHOWCASE_DEGREES,
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
    "SHOWCASE_DEGREES",
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
