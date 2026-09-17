import pytest

from mt_playground import app as mt_app
from tests.fake_streamlit import FakeStreamlit


def _render(pressed=None, overrides=None):
    fake_st = FakeStreamlit(pressed=pressed, overrides=overrides)
    return fake_st


def test_main_renders_without_errors(monkeypatch):
    fake_st = _render()
    monkeypatch.setattr(mt_app, "st", fake_st)

    mt_app.main()

    assert fake_st.page_config["page_title"] == "Метрики машинного перевода"
    # сравнение метрик + четыре вкладки корпуса
    assert len(fake_st.figures) >= 2


def test_first_render_seeds_the_editor_with_the_model_output(monkeypatch):
    fake_st = _render()
    monkeypatch.setattr(mt_app, "st", fake_st)

    mt_app.main()

    assert fake_st.session_state["segment_id"] == "s001"
    assert fake_st.session_state["hypothesis_text"] == "Сохранить изменения"


def test_set_hypothesis_bumps_the_editor_revision(monkeypatch):
    """Ключ поля ввода обязан меняться вместе с текстом.

    Иначе Streamlit оставляет в поле старое значение — или, если виджет не
    отрисовался из-за `st.rerun()`, вообще стирает его.
    """
    fake_st = _render()
    monkeypatch.setattr(mt_app, "st", fake_st)
    mt_app.main()

    before = fake_st.session_state["editor_revision"]
    mt_app._set_hypothesis("другой перевод", "правка")

    assert fake_st.session_state["hypothesis_text"] == "другой перевод"
    assert fake_st.session_state["editor_revision"] == before + 1
    assert fake_st.session_state["last_action"] == "правка"


def test_editor_never_relies_on_bare_widget_state(monkeypatch):
    """Регрессия: текст пропадал, потому что источником правды был ключ виджета."""
    fake_st = _render()
    monkeypatch.setattr(mt_app, "st", fake_st)
    mt_app.main()

    # Имитируем то, что делает Streamlit, когда виджет не отрисовался:
    # удаляет его ключ из состояния.
    for key in [k for k in fake_st.session_state if k.startswith("hypothesis_editor_")]:
        del fake_st.session_state[key]

    mt_app.main()

    assert fake_st.session_state["hypothesis_text"] == "Сохранить изменения"


def test_scores_are_computed_against_the_reference(monkeypatch):
    fake_st = _render()
    monkeypatch.setattr(mt_app, "st", fake_st)

    mt_app.main()

    scores = mt_app._score(fake_st.session_state["hypothesis_text"], "Сохранить изменения")
    assert scores["BLEU"] == pytest.approx(1.0)
    assert scores["TER"] == pytest.approx(0.0)


# --- вывод про слепую зону -------------------------------------------------


def _verdict(monkeypatch, segment_id, hypothesis, bleu_threshold=0.45):
    from mt_playground.checks import NEG_RU
    from mt_playground.corpus import load_corpus

    corpus = load_corpus()
    segment = corpus[corpus["id"] == segment_id].iloc[0]

    fake_st = FakeStreamlit()
    fake_st.session_state["hypothesis_text"] = hypothesis
    monkeypatch.setattr(mt_app, "st", fake_st)

    current = mt_app._score(hypothesis, segment["ru_ref"])
    baseline = mt_app._score(segment["ru_mt"], segment["ru_ref"])
    mt_app._render_verdict(segment, current, baseline, NEG_RU, bleu_threshold)
    return fake_st


def test_broken_placeholder_raises_the_blind_spot_warning(monkeypatch):
    """Сломанный плейсхолдер при высоком BLEU — главный вывод работы."""
    fake_st = _verdict(monkeypatch, "s002", 'Удалить файл «(name}»?')

    assert any("Вот оно" in message for message in fake_st.warnings)
    assert any("плейсхолдеры" in message for message in fake_st.errors)


def test_warning_fires_even_when_the_model_output_was_already_broken(monkeypatch):
    """Регрессия: сравнение с выходом модели глушило вывод на s002.

    Выход модели на этом сегменте сам сломан, поэтому «стало хуже» не
    наступало никогда и подсказка не появлялась.
    """
    fake_st = _verdict(monkeypatch, "s002", 'Удалить файл "(имя}"?')

    assert any("Вот оно" in message for message in fake_st.warnings)


def test_synonym_swap_raises_the_opposite_notice(monkeypatch):
    """Смысл сохранён, BLEU ниже порога, проверки молчат."""
    fake_st = _verdict(monkeypatch, "s001", "Записать изменения", bleu_threshold=0.8)

    assert any("И наоборот" in message for message in fake_st.infos)
    assert fake_st.successes, "проверки обязаны молчать"


def test_clean_reference_translation_raises_nothing(monkeypatch):
    fake_st = _verdict(monkeypatch, "s001", "Сохранить изменения")

    assert fake_st.warnings == []
    assert fake_st.infos == []
