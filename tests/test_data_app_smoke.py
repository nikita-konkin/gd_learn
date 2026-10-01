"""The data-preparation interface renders, and shows lecture 2's results."""

import pytest

from data_playground import app as data_app
from data_playground.cleaning import BEFORE, MEAN


def _run(monkeypatch, overrides=None):
    from tests.fake_streamlit import FakeStreamlit

    data_app._timings.cache_clear()
    fake_st = FakeStreamlit(overrides={data_app.SIZE_LABEL: 10_000, **(overrides or {})})
    monkeypatch.setattr(data_app, "st", fake_st)
    data_app.main()
    return fake_st


def _metric(fake_st, label, occurrence=0):
    return [value for name, value in fake_st.metrics if name == label][occurrence]


@pytest.fixture
def default_run(monkeypatch):
    return _run(monkeypatch)


def test_main_renders_without_errors(default_run):
    assert default_run.page_config["page_title"] == "Конвейер подготовки данных"
    assert not default_run.errors
    # cleaned histogram, timings
    assert len(default_run.figures) == 2


def test_the_lecture_tab_reproduces_the_lecture(default_run):
    assert _metric(default_run, "Пропусков") == "2"
    assert _metric(default_run, "Полных дубликатов") == "1"
    assert _metric(default_run, "Заполнено значением") == "-48.65"
    assert _metric(default_run, "Невозможных значений после очистки") == "0"
    assert any("Пропусков осталось: 0" in message for message in default_run.infos)


def test_the_early_mean_is_reported_as_an_error(monkeypatch):
    fake_st = _run(monkeypatch, overrides={data_app.ORDER_LABEL: BEFORE, data_app.STATISTIC_LABEL: MEAN})

    assert _metric(fake_st, "Заполнено значением") == "-332.43"
    assert any("2 невозможных значения" in message for message in fake_st.errors)
    assert _metric(fake_st, "Ошибка очистки") == "-8.06"
    assert not fake_st.infos


def test_the_early_median_gets_a_quieter_warning(monkeypatch):
    fake_st = _run(monkeypatch, overrides={data_app.ORDER_LABEL: BEFORE})

    assert _metric(fake_st, "Заполнено значением") == "-52.30"
    assert any("Медиана устойчивее" in message for message in fake_st.warnings)


def test_the_larger_table_reports_its_error(default_run):
    assert _metric(default_run, "Истинное среднее") == "-50.29"
    assert _metric(default_run, "Ошибка очистки") == "+0.00"


def test_the_speed_tab_reports_agreement(default_run):
    assert _metric(default_run, "Результаты совпадают") == "да"


def test_the_memory_caption_explains_the_platform(default_run):
    assert any("указатель занимает" in caption and "3.4 МБ" in caption for caption in default_run.captions)


def test_every_slider_opens_on_its_default(default_run):
    defaults = {label: value for label, value, _, _ in default_run.sliders}

    assert defaults == {
        data_app.ROWS_LABEL: 1000,
        data_app.MISSING_LABEL: 10,
        data_app.OUTLIERS_LABEL: 5,
        data_app.SEED_LABEL: 0,
        data_app.SIZE_LABEL: 1_000_000,
    }


def test_the_page_says_what_it_does_not_do(default_run):
    assert "Чего эта площадка не делает" in default_run.expanders
