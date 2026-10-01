"""Площадка «Граф задач и отказы» — модуль 4 лабораторных работ Б.1.2.2.

Lecture 12's scheduler with handles on it: which task fails, how many times, how
many attempts the scheduler allows, and whether the edge that guards publication
is in the graph at all.
"""

from dag_playground.app import main
from dag_playground.scheduler import (
    ALWAYS,
    CYCLE_EDGE,
    GUARD_EDGE,
    LECTURE_FAILURE,
    LECTURE_RECOVERY,
    PIPELINE_EDGES,
    CycleError,
    Entry,
    Outcome,
    Scenario,
    execution_levels,
    failing_handler,
    run,
    simulate,
    task_names,
)

__all__ = [
    "ALWAYS",
    "CYCLE_EDGE",
    "GUARD_EDGE",
    "LECTURE_FAILURE",
    "LECTURE_RECOVERY",
    "PIPELINE_EDGES",
    "CycleError",
    "Entry",
    "Outcome",
    "Scenario",
    "execution_levels",
    "failing_handler",
    "main",
    "run",
    "simulate",
    "task_names",
]
