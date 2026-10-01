"""Площадка «Метрика и дисбаланс» — тема «Оценка качества», модуль 2 курса Б.1.2.2.

The accuracy trap on rare events, a threshold and the price of its two
mistakes, and three remedies for imbalance under one threshold. All metrics are
computed in NumPy from scores prepared offline.
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
    load_failures,
    load_rare_class,
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
    "load_failures",
    "load_rare_class",
    "main",
    "majority",
    "nearest_recall",
    "populations",
    "precision_recall_curve",
]
