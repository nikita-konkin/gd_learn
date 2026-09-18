import pytest

from lm_playground import app as lm_app
from lm_playground.corpus import CONTENT_TYPES
from tests.fake_streamlit import FakeStreamlit


def _run(monkeypatch, overrides=None):
    fake_st = FakeStreamlit(overrides=overrides)
    monkeypatch.setattr(lm_app, "st", fake_st)
    lm_app.main()
    return fake_st


def test_main_renders_without_errors(monkeypatch):
    fake_st = _run(monkeypatch)

    assert fake_st.page_config["page_title"] == "Языковая модель и температура"
    # кривая переобучения + распределение следующего символа
    assert len(fake_st.figures) == 2


def test_default_order_generates_and_measures(monkeypatch):
    fake_st = _run(monkeypatch)
    generation = fake_st.session_state["generation"]

    assert len(generation.text) == 240
    assert len(fake_st.session_state["copied_mask"]) == 240


def test_long_context_raises_the_memorisation_warning(monkeypatch):
    """При n=8 текст почти целиком списан, и интерфейс обязан это сказать."""
    fake_st = _run(monkeypatch, overrides={"Порядок n — сколько символов контекста": 8})

    assert any("списано" in message or "Осторожно" in message for message in fake_st.warnings)


def test_short_context_raises_the_opposite_notice(monkeypatch):
    fake_st = _run(monkeypatch, overrides={"Порядок n — сколько символов контекста": 1})

    assert any("бессмыслен" in message for message in fake_st.infos)


def test_highlight_wraps_only_the_copied_spans():
    html = lm_app._highlight("абвгд", [False, True, True, False, False])

    assert "<mark>бв</mark>" in html
    assert "<mark>а" not in html


def test_highlight_escapes_markup():
    html = lm_app._highlight("<b>", [False, False, False])

    assert "&lt;b&gt;" in html
    assert "<b>" not in html


def test_highlight_handles_an_all_copied_text():
    html = lm_app._highlight("абв", [True, True, True])

    assert "<mark>абв</mark>" in html


def test_generation_seed_changes_the_text_but_not_the_measurements(monkeypatch):
    """Регрессия: зерно генерации переставляло и разбиение на выборки.

    Ползунок «случайное зерно» должен менять только жребий при сэмплировании.
    Пока он заодно переразбивал корпус, перплексия прыгала на каждый щелчок и
    кривая переобучения тонула в шуме.
    """
    first = _run(monkeypatch, overrides={"Случайное зерно": 3})
    second = _run(monkeypatch, overrides={"Случайное зерно": 777})

    def perplexities(fake):
        return [value for label, value in fake.metrics if label.startswith("Перплексия")]

    assert perplexities(first) == perplexities(second)
    assert first.session_state["generation"].text != second.session_state["generation"].text


def test_sweep_does_not_accept_a_sampling_seed():
    """Кривая не должна зависеть от зерна генерации даже по недосмотру."""
    with pytest.raises(TypeError):
        lm_app._sweep(CONTENT_TYPES, 7)
