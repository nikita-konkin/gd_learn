"""Площадка «Конвейер подготовки данных» — модуль 1 лабораторных работ Б.1.2.2.

Lecture 2's cleaning chain with its order made a parameter, the same chain on a
thousand generated readings, and the lecture's two speed-and-memory
measurements repeated live in the browser.
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
    clean_lecture_table,
    clean_signal,
    estimate,
    inspect,
    lecture_table,
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
    "clean_lecture_table",
    "clean_signal",
    "estimate",
    "inspect",
    "lecture_table",
    "main",
    "make_readings",
    "memory",
    "results_agree",
    "sum_of_squares",
]
