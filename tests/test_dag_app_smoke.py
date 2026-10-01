"""The task-graph interface renders, and says what lecture 12 says."""

import pytest

from dag_playground import app as dag_app


def _run(monkeypatch, overrides=None):
    from tests.fake_streamlit import FakeStreamlit

    fake_st = FakeStreamlit(overrides=overrides)
    monkeypatch.setattr(dag_app, "st", fake_st)
    dag_app.main()
    return fake_st


def _metric(fake_st, label):
    return next(value for name, value in fake_st.metrics if name == label)


@pytest.fixture
def default_run(monkeypatch):
    return _run(monkeypatch)


def test_main_renders_without_errors(default_run):
    assert default_run.page_config["page_title"] == "Граф задач и отказы"
    assert not default_run.errors
    assert len(default_run.figures) == 1


def test_the_default_is_the_lectures_first_run(default_run):
    assert any("Первый прогон лекции 12" in message for message in default_run.infos)
    assert _metric(default_run, "Выполнено успешно") == "7 из 7"
    assert _metric(default_run, "Попыток всего") == "9"
    assert any("с 3-й попытки" in caption for caption in default_run.captions)


def test_the_default_shows_the_parallel_level(default_run):
    assert any("отчёт о качестве, признаки — одновременно" in text for text in default_run.markdowns)


def test_the_lectures_second_run_prints_four_of_seven(monkeypatch):
    fake_st = _run(monkeypatch, overrides={dag_app.FAILURES_LABEL: "всегда", dag_app.ATTEMPTS_LABEL: 2})

    assert _metric(fake_st, "Выполнено успешно") == "4 из 7"
    assert _metric(fake_st, "Пропущено") == "2"
    assert any("Второй прогон лекции 12" in message for message in fake_st.infos)
    assert any("Это сделал граф" in message for message in fake_st.successes)


def test_dropping_the_guard_edge_publishes_an_unverified_model(monkeypatch):
    fake_st = _run(
        monkeypatch,
        overrides={dag_app.FAILURES_LABEL: "всегда", dag_app.ATTEMPTS_LABEL: 2, dag_app.GUARD_LABEL: True},
    )

    assert _metric(fake_st, "Выполнено успешно") == "5 из 7"
    assert any("опубликована без оценки" in message for message in fake_st.warnings)


def test_a_cycle_stops_the_page_before_anything_runs(monkeypatch):
    fake_st = _run(monkeypatch, overrides={dag_app.CYCLE_LABEL: True})

    assert any("В графе цикл" in message for message in fake_st.errors)
    assert fake_st.metrics == []
    assert fake_st.figures == []


def test_nobody_failing_needs_one_attempt_per_task(monkeypatch):
    fake_st = _run(monkeypatch, overrides={dag_app.FAILING_LABEL: "никакая"})

    assert _metric(fake_st, "Попыток всего") == "7"
    assert not fake_st.infos


def test_the_page_says_what_it_does_not_do(default_run):
    assert "Чего эта площадка не делает" in default_run.expanders
