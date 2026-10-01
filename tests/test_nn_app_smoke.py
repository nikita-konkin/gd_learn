"""The network interface renders, and its numbers say what the page claims."""

import pytest

from nn_playground import app as nn_app
from nn_playground.network import RELU_BUG, SIGN_BUG


def _run(monkeypatch, overrides=None):
    from tests.fake_streamlit import FakeStreamlit

    fake_st = FakeStreamlit(overrides=overrides)
    monkeypatch.setattr(nn_app, "st", fake_st)
    nn_app.main()
    return fake_st


def _metric(fake_st, label):
    return next(value for name, value in fake_st.metrics if name == label)


@pytest.fixture
def default_run(monkeypatch):
    return _run(monkeypatch)


def test_main_renders_without_errors(default_run):
    assert default_run.page_config["page_title"] == "Нейросеть на NumPy"
    assert not default_run.errors
    # learning curve, decision boundary
    assert len(default_run.figures) == 2


def test_the_default_run_learns_and_its_gradient_checks_out(default_run):
    assert float(_metric(default_run, "Потери до обучения")) > float(_metric(default_run, "Потери после"))
    assert float(_metric(default_run, "Точность на отложенной выборке")) > 0.9
    assert _metric(default_run, "Аналитический W1[0, 0]") == _metric(default_run, "Численный W1[0, 0]")
    assert any("все 21 вес проверочной" in caption for caption in default_run.captions)


def test_correct_formulas_pass_the_check(default_run):
    assert _metric(default_run, "Расходятся компонент") == "0 из 21"
    assert any("Все 21 компонент совпадают" in message for message in default_run.successes)


def test_the_relu_bug_trains_plausibly_and_fails_the_check(monkeypatch):
    fake_st = _run(monkeypatch, overrides={nn_app.BUG_LABEL: RELU_BUG})

    assert float(_metric(fake_st, "Точность на отложенной выборке")) > 0.8
    assert _metric(fake_st, "Расходятся компонент") == "15 из 21"
    assert any("неверны" in message for message in fake_st.errors)
    assert any("6 весов" in caption for caption in fake_st.captions)
    assert any("без ошибки дают точность" in caption for caption in fake_st.captions)
    assert not fake_st.infos


def test_a_flipped_sign_is_reported(monkeypatch):
    fake_st = _run(monkeypatch, overrides={nn_app.BUG_LABEL: SIGN_BUG})

    assert _metric(fake_st, "Точность на отложенной выборке") == "0.500"
    assert _metric(fake_st, "Расходятся компонент") == "21 из 21"


def test_another_step_changes_the_result(monkeypatch, default_run):
    fake_st = _run(monkeypatch, overrides={nn_app.RATE_LABEL: 2.0})

    assert _metric(fake_st, "Точность на отложенной выборке") != _metric(default_run, "Точность на отложенной выборке")


def test_every_slider_opens_on_its_default(default_run):
    defaults = {label: value for label, value, _, _ in default_run.sliders}

    assert defaults == {
        nn_app.RATE_LABEL: 0.5,
        nn_app.EPOCHS_LABEL: 400,
        nn_app.HIDDEN_LABEL: 16,
        nn_app.SEED_LABEL: 0,
    }


def test_the_page_says_what_it_does_not_do(default_run):
    assert "Чего эта площадка не делает" in default_run.expanders
