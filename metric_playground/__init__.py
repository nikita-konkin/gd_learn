"""Площадка «Метрика и дисбаланс» — модуль 2 лабораторных работ Б.1.2.2.

The accuracy trap on rare events, lecture 4's threshold and the price of its two
mistakes, and lecture 6's three remedies for imbalance under one threshold. All
metrics are computed in NumPy from scores prepared offline.
"""

from metric_playground.app import main
from metric_playground.metrics import (
    REMEDIES,
    Confusion,
    Operating,
    Population,
    average_precision,
    cheapest_threshold,
    confusion,
    load_lecture4,
    load_lecture6,
    majority,
    nearest_recall,
    populations,
    precision_recall_curve,
)

__all__ = [
    "REMEDIES",
    "Confusion",
    "Operating",
    "Population",
    "average_precision",
    "cheapest_threshold",
    "confusion",
    "load_lecture4",
    "load_lecture6",
    "main",
    "majority",
    "nearest_recall",
    "populations",
    "precision_recall_curve",
]
