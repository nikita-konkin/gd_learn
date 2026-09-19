"""The translation-memory interface renders, and says the right thing."""

from tests.fake_streamlit import FakeStreamlit
from tm_playground import app as tm_app

QUERY_LABEL = "Сегмент на перевод"
SOURCE_LABEL = "Что ищем в памяти"


def _run(monkeypatch, overrides=None):
    fake_st = FakeStreamlit(overrides=overrides)
    monkeypatch.setattr(tm_app, "st", fake_st)
    tm_app.main()
    return fake_st


def test_main_renders_without_errors(monkeypatch):
    fake_st = _run(monkeypatch)

    assert fake_st.page_config["page_title"] == "Память переводов"
    assert not fake_st.errors
    # divergence scatter, threshold curve, ARI bars, corpus map
    assert len(fake_st.figures) == 4


def test_the_default_query_is_one_the_measures_agree_on(monkeypatch):
    fake_st = _run(monkeypatch)

    assert any("Меры согласны" in message for message in fake_st.successes)
    assert not fake_st.warnings or all(
        "Меры разошлись" not in message for message in fake_st.warnings
    )


def test_a_reordered_query_reports_the_measures_splitting(monkeypatch):
    fake_st = _run(monkeypatch, overrides={QUERY_LABEL: "Настройки поиска"})

    divergence = [message for message in fake_st.warnings if "Меры разошлись" in message]
    assert divergence, "the central case of the lab has to be called out"
    assert "Поиск по настройкам" in divergence[0]


def test_the_headline_count_matches_the_lab(monkeypatch):
    fake_st = _run(monkeypatch)
    agreed = next(value for label, value in fake_st.metrics if "одно и то же" in label)

    assert agreed == "9 из 13"


def test_typing_your_own_text_says_why_embeddings_go_missing(monkeypatch):
    fake_st = _run(
        monkeypatch,
        overrides={SOURCE_LABEL: tm_app.OWN_QUERY, QUERY_LABEL: "Сохранить настройки поиска"},
    )

    unavailable = [message for message in fake_st.infos if "готовые эмбеддинги" in message]
    assert unavailable, "a measure that cannot answer must say so rather than vanish"
    assert "отдельный этап конвейера" in unavailable[0]


def test_a_lab_query_keeps_its_embeddings(monkeypatch):
    fake_st = _run(monkeypatch, overrides={QUERY_LABEL: "По вашему запросу результатов нет"})

    assert not any("готовые эмбеддинги" in message for message in fake_st.infos)


def test_the_clustering_tab_keeps_its_negative_result(monkeypatch):
    fake_st = _run(monkeypatch)

    assert any("Отрицательный результат" in message for message in fake_st.warnings)


def test_an_empty_query_asks_for_one_instead_of_failing(monkeypatch):
    fake_st = _run(monkeypatch, overrides={SOURCE_LABEL: tm_app.OWN_QUERY, QUERY_LABEL: "   "})

    assert any("Введите сегмент" in message for message in fake_st.infos)
    assert not fake_st.errors


class _NoTextArea(FakeStreamlit):
    """A Streamlit double that refuses the widget which needs an explicit apply."""

    def text_area(self, *args, **kwargs):
        raise AssertionError(
            "st.text_area only applies its contents on Ctrl+Enter, so clicking away "
            "leaves the page answering the previous query with no sign of it"
        )


def test_the_free_text_box_commits_without_a_keyboard_shortcut(monkeypatch):
    """Caught in the browser: typing and clicking away left a stale verdict."""
    query = "Очистите кэш браузера и попробуйте снова"
    fake_st = _NoTextArea(overrides={SOURCE_LABEL: tm_app.OWN_QUERY, QUERY_LABEL: query})
    monkeypatch.setattr(tm_app, "st", fake_st)

    tm_app.main()

    divergence = [message for message in fake_st.warnings if "Меры разошлись" in message]
    assert divergence, "the trailing clause is what makes the two measures split"
    assert "Очистить кэш" in divergence[0]
    assert "Укажите резервный номер телефона" in divergence[0]
