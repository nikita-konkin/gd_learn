"""Площадка «Конвейер подготовки данных» — тема «Данные», модуль 1 курса Б.1.2.2.

A cleaning chain with its order made a parameter, the same chain on a thousand
generated readings, and two speed-and-memory measurements made live in the
browser.
"""

from data_playground.app import main
from data_playground.cleaning import (
    AFTER,
    BEFORE,
    MEAN,
    MEDIAN,
    OUTLIER_LIMIT,
    Cleaned,
    Estimate,
    Readings,
    ReadingsSettings,
    clean_defect_table,
    clean_signal,
    defect_table,
    estimate,
    inspect,
    make_readings,
)
from data_playground.speed import Memory, Timing, memory, results_agree, sum_of_squares

__all__ = [
    "AFTER",
    "BEFORE",
    "MEAN",
    "MEDIAN",
    "OUTLIER_LIMIT",
    "Cleaned",
    "Estimate",
    "Memory",
    "Readings",
    "ReadingsSettings",
    "Timing",
    "clean_defect_table",
    "clean_signal",
    "estimate",
    "inspect",
    "defect_table",
    "main",
    "make_readings",
    "memory",
    "results_agree",
    "sum_of_squares",
]
