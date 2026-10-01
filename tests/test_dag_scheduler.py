"""The scheduler must order, retry and skip as the page says it does."""

import pytest

from dag_playground.scheduler import (
    ALWAYS,
    FAILING_RUN,
    PIPELINE_EDGES,
    RECOVERING_RUN,
    CycleError,
    Scenario,
    execution_levels,
    failing_handler,
    run,
    simulate,
    task_names,
)


def test_the_pipeline_has_seven_tasks_and_seven_edges():
    tasks = {task for edge in PIPELINE_EDGES for task in edge}

    assert len(tasks) == 7
    assert len(PIPELINE_EDGES) == 7


def test_the_levels_put_independent_tasks_together():
    levels = execution_levels(PIPELINE_EDGES)

    assert levels == [
        ["выгрузка"],
        ["очистка"],
        ["отчёт_о_качестве", "признаки"],
        ["обучение"],
        ["оценка"],
        ["публикация"],
    ]


def test_a_cycle_is_rejected_with_a_message():
    with pytest.raises(CycleError, match="В графе цикл: выполнение невозможно"):
        execution_levels([("a", "b"), ("b", "c"), ("c", "a")])


def test_a_cycle_error_is_still_a_value_error():
    assert issubclass(CycleError, ValueError)


def test_the_first_run_recovers_on_the_third_attempt():
    outcome = simulate(RECOVERING_RUN)

    assert outcome.succeeded == 7
    attempts = {entry.task: entry.attempts for entry in outcome.log}
    assert attempts["обучение"] == 3
    assert all(count == 1 for task, count in attempts.items() if task != "обучение")
    assert outcome.total_attempts == 9


def test_the_second_run_succeeds_four_of_seven_and_skips_the_rest():
    outcome = simulate(FAILING_RUN)

    assert (outcome.succeeded, len(outcome.status)) == (4, 7)
    outcomes = {entry.task: entry.outcome for entry in outcome.log}
    assert outcomes["обучение"] == "провал: модель не сходится"
    assert outcomes["оценка"] == "пропущена"
    assert outcomes["публикация"] == "пропущена"
    assert not outcome.published_unverified()


def test_without_the_guard_edge_an_unverified_model_is_published():
    outcome = simulate(Scenario(failures=ALWAYS, attempts=2, guard=False))

    assert outcome.status["обучение"] == "провал"
    assert outcome.status["оценка"] == "пропущена"
    assert outcome.status["публикация"] == "успех"
    assert outcome.published_unverified()
    # Publication now sits on the same level as the failing training.
    assert ("обучение", "публикация") in outcome.levels


def test_the_cycle_edge_makes_the_scenario_unrunnable():
    outcome = simulate(Scenario(cycle=True))

    assert outcome.cycle_error == "В графе цикл: выполнение невозможно"
    assert outcome.log == ()


def test_no_failure_runs_every_task_once():
    outcome = simulate(Scenario(failing_task=None))

    assert outcome.succeeded == 7
    assert outcome.total_attempts == 7


@pytest.mark.parametrize("failures", [0, 1, 2, 3, 4])
def test_a_task_recovers_exactly_when_it_fails_fewer_times_than_the_attempts(failures):
    outcome = simulate(Scenario(failing_task="признаки", failures=failures, attempts=3))

    recovered = outcome.status["признаки"] == "успех"
    assert recovered is (failures < 3)


def test_a_failing_handler_counts_its_own_calls():
    handler = failing_handler("очистка", 1)

    with pytest.raises(RuntimeError, match="источник данных недоступен"):
        handler()
    handler()  # second call succeeds


def test_a_permanent_failure_outside_training_gets_a_generic_message():
    with pytest.raises(RuntimeError, match="неустранимый сбой"):
        failing_handler("очистка", ALWAYS)()


def test_run_works_with_handlers_given_for_no_task():
    log, status = run(PIPELINE_EDGES, {}, attempts=3)

    assert len(log) == 7
    assert set(status.values()) == {"успех"}


def test_task_names_follow_execution_order():
    assert task_names()[0] == "выгрузка"
    assert task_names()[-1] == "публикация"
