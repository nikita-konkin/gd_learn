"""Площадка «Граф задач и отказы» — тема «Конвейеры и эксплуатация», модуль 4 курса Б.1.2.2.

A small scheduler with handles on it: which task fails, how many times, how
many attempts the scheduler allows, and whether the edge that guards publication
is in the graph at all.
"""

from dag_playground.app import main
from dag_playground.scheduler import (
    ALWAYS,
    CYCLE_EDGE,
    FAILING_RUN,
    GUARD_EDGE,
    PIPELINE_EDGES,
    RECOVERING_RUN,
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
    "FAILING_RUN",
    "RECOVERING_RUN",
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
