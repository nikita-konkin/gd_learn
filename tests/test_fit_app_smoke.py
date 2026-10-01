"""The model-complexity interface renders, and each tab opens where the trouble is plainest."""

import pytest

from fit_playground import app as fit_app


def _run(monkeypatch, overrides=None):
    from tests.fake_streamlit import FakeStreamlit

    fake_st = FakeStreamlit(overrides=overrides)
    monkeypatch.setattr(fit_app, "st", fake_st)
    fit_app.main()
    return fake_st


def _metric(fake_st, label):
    return next(value for name, value in fake_st.metrics if name == label)


@pytest.fixture
def default_run(monkeypatch):
    return _run(monkeypatch)


def test_main_renders_without_errors(default_run):
    assert default_run.page_config["page_title"] == "Сложность модели"
    assert not default_run.errors
    # polynomial, error curve, two boundaries, weight paths
    assert len(default_run.figures) == 5


def test_the_polynomial_tab_opens_on_degree_17(default_run):
    assert _metric(default_run, "Ошибка на обучении") == "0.023"
    assert _metric(default_run, "Ошибка на новых точках") == "13.2"
    assert any("хуже, чем если бы модель всегда отвечала средним" in message for message in default_run.warnings)


def test_a_moderate_degree_raises_no_warning_about_new_points(monkeypatch):
    fake_st = _run(monkeypatch, overrides={fit_app.DEGREE_LABEL: 4})

    assert _metric(fake_st, "Ошибка на обучении") == "0.059"
    assert not any("хуже, чем если бы" in message for message in fake_st.warnings)


def test_the_neighbours_tab_shows_both_accuracies(default_run):
    assert _metric(default_run, "k = 1: на обучении") == "1.000"
    assert _metric(default_run, "k = 1: на проверке") == "0.950"
    assert _metric(default_run, "глубина 3: на проверке") == "0.890"
    assert any("цена переобучения здесь мала" in caption for caption in default_run.captions)


def test_the_penalty_tab_opens_on_the_collapse(default_run):
    assert _metric(default_run, "Lasso обнулил признаков") == "10 из 10"
    assert any("отказ от модели" in message and "alpha ≈ 53" in message for message in default_run.warnings)


def test_a_middle_penalty_selects_instead_of_collapsing(monkeypatch):
    fake_st = _run(monkeypatch, overrides={fit_app.ALPHA_LABEL: "10.8"})

    assert _metric(fake_st, "Lasso обнулил признаков") == "6 из 10"
    assert not any("отказ от модели" in message for message in fake_st.warnings)
    assert any("Lasso обнулил 6 признаков из 10 и оставил 4" in message for message in fake_st.infos)


def test_an_unlimited_tree_is_labelled(monkeypatch):
    fake_st = _run(monkeypatch, overrides={fit_app.DEPTH_LABEL: "без ограничения"})

    assert _metric(fake_st, "глубина ∞: на обучении") == "1.000"


def test_every_slider_opens_on_its_default(default_run):
    defaults = {label: value for label, value, _, _ in default_run.sliders}

    assert defaults == {
        fit_app.DEGREE_LABEL: 17,
        fit_app.NEIGHBOURS_LABEL: 1,
        fit_app.DEPTH_LABEL: "3",
        fit_app.ALPHA_LABEL: "100",
    }


def test_the_page_says_what_it_does_not_do(default_run):
    assert "Чего эта площадка не делает" in default_run.expanders
