"""A forty-line scheduler, plus the means to break it on purpose.

``execution_levels`` and ``run`` are the whole algorithm: lay the graph out in
levels, run each level in turn, retry a failing task, and skip any task whose
parents did not all succeed. The rest of the module chooses *which* task fails,
*how often*, and *which edges* the graph has.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# A small training pipeline. A pair means "the first must finish before the second".
PIPELINE_EDGES: tuple[tuple[str, str], ...] = (
    ("выгрузка", "очистка"),
    ("очистка", "признаки"),
    ("очистка", "отчёт_о_качестве"),
    ("признаки", "обучение"),
    ("обучение", "оценка"),
    ("оценка", "публикация"),
    ("отчёт_о_качестве", "публикация"),
)

# The edge that stops an unverified model from being published. Removing it is
# the playground's way of showing that the graph, not the developer's care, is
# what holds publication back after a failure.
GUARD_EDGE = ("оценка", "публикация")
# An edge that closes a loop: publication would have to precede the export.
CYCLE_EDGE = ("публикация", "выгрузка")

SUCCESS = "успех"
SKIPPED = "пропущена"
FAILED = "провал"

TRANSIENT_ERROR = "источник данных недоступен"
PERMANENT_ERRORS = {"обучение": "модель не сходится"}
PERMANENT_DEFAULT = "неустранимый сбой"

# How many times a task fails before it recovers. ``None`` means never.
ALWAYS = None


class CycleError(ValueError):
    """The graph has a loop, so no order of execution exists."""


def execution_levels(edges) -> list[list[str]]:
    """Tasks grouped into levels; within a level the tasks are independent.

    Raises ``CycleError`` — a ``ValueError``, so plain code can catch it — when
    some task can never become ready.
    """
    requirements: dict[str, set[str]] = {}
    all_tasks: set[str] = set()
    for source, target in edges:
        requirements.setdefault(target, set()).add(source)
        all_tasks.update([source, target])

    finished: set[str] = set()
    order: list[list[str]] = []
    while len(finished) < len(all_tasks):
        ready = sorted(task for task in all_tasks if task not in finished and requirements.get(task, set()) <= finished)
        if not ready:
            raise CycleError("В графе цикл: выполнение невозможно")
        order.append(ready)
        finished.update(ready)
    return order


@dataclass(frozen=True)
class Entry:
    """One line of the run's journal."""

    task: str
    outcome: str
    attempts: int

    @property
    def status(self) -> str:
        """The outcome without its error message: успех, пропущена or провал."""
        return FAILED if self.outcome.startswith(FAILED) else self.outcome


def run(edges, handlers: dict[str, Callable[[], None]], attempts: int = 3) -> tuple[list[Entry], dict[str, str]]:
    """Execute the graph level by level.

    A task is skipped when any parent did not succeed; otherwise it is tried up
    to ``attempts`` times. A pause between attempts is left out:
    a browser has nothing to wait for.
    """
    log: list[Entry] = []
    status: dict[str, str] = {}

    for level in execution_levels(edges):
        for task in level:
            parents = [source for source, target in edges if target == task]
            if any(status.get(parent) != SUCCESS for parent in parents):
                status[task] = SKIPPED
                log.append(Entry(task, SKIPPED, 0))
                continue

            for attempt in range(1, attempts + 1):
                try:
                    handlers.get(task, lambda: None)()
                    status[task] = SUCCESS
                    log.append(Entry(task, SUCCESS, attempt))
                    break
                except Exception as error:  # any failure is a failed attempt
                    if attempt == attempts:
                        status[task] = FAILED
                        log.append(Entry(task, f"{FAILED}: {error}", attempt))
    return log, status


def failing_handler(task: str, failures: int | None) -> Callable[[], None]:
    """A task that fails ``failures`` times and then succeeds; ``ALWAYS`` never succeeds.

    Training gets messages of its own: a model that does not converge is the
    failure people meet first.
    """
    calls = {"count": 0}

    def handler() -> None:
        calls["count"] += 1
        if failures is ALWAYS:
            raise RuntimeError(PERMANENT_ERRORS.get(task, PERMANENT_DEFAULT))
        if calls["count"] <= failures:
            raise RuntimeError(TRANSIENT_ERROR)

    return handler


@dataclass(frozen=True)
class Scenario:
    """Everything the page lets the student change. Hashable, so it can be cached."""

    failing_task: str | None = "обучение"
    failures: int | None = 2
    attempts: int = 3
    guard: bool = True
    cycle: bool = False

    def edges(self) -> tuple[tuple[str, str], ...]:
        edges = [edge for edge in PIPELINE_EDGES if self.guard or edge != GUARD_EDGE]
        if self.cycle:
            edges.append(CYCLE_EDGE)
        return tuple(edges)

    def handlers(self) -> dict[str, Callable[[], None]]:
        if self.failing_task is None or self.failures == 0:
            return {}
        return {self.failing_task: failing_handler(self.failing_task, self.failures)}


# Two telling runs: one the retries save, one they cannot.
RECOVERING_RUN = Scenario(failing_task="обучение", failures=2, attempts=3)
FAILING_RUN = Scenario(failing_task="обучение", failures=ALWAYS, attempts=2)


@dataclass(frozen=True)
class Outcome:
    """The result of one scenario, or the reason it could not run."""

    levels: tuple[tuple[str, ...], ...]
    log: tuple[Entry, ...]
    status: dict[str, str]
    cycle_error: str | None = None

    @property
    def succeeded(self) -> int:
        return sum(1 for state in self.status.values() if state == SUCCESS)

    @property
    def skipped(self) -> int:
        return sum(1 for state in self.status.values() if state == SKIPPED)

    @property
    def total_attempts(self) -> int:
        return sum(entry.attempts for entry in self.log)

    def published_unverified(self) -> bool:
        """Publication ran although the model it publishes was never evaluated."""
        return self.status.get("публикация") == SUCCESS and self.status.get("оценка") != SUCCESS


def simulate(scenario: Scenario) -> Outcome:
    """Lay out the scenario's graph and run it, catching a cycle as an outcome."""
    edges = scenario.edges()
    try:
        levels = execution_levels(edges)
    except CycleError as error:
        return Outcome(levels=(), log=(), status={}, cycle_error=str(error))

    log, status = run(edges, scenario.handlers(), attempts=scenario.attempts)
    return Outcome(
        levels=tuple(tuple(level) for level in levels),
        log=tuple(log),
        status=status,
    )


def task_names() -> list[str]:
    """Every task of the pipeline, in execution order."""
    return [task for level in execution_levels(PIPELINE_EDGES) for task in level]
