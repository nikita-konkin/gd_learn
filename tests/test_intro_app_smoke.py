"""The intro-lecture interface renders, and says what the slides say."""

from intro_playground import app as intro_app
from tests.fake_streamlit import FakeStreamlit

FEATURES_LABEL = "Признаки"
DEPTH_LABEL = "Глубина дерева"
MARGIN_LABEL = "Запас под перевод, %"
MIN_COUNT_LABEL = "Минимум вхождений пары"


def _run(monkeypatch, overrides=None):
    fake_st = FakeStreamlit(overrides=overrides)
    monkeypatch.setattr(intro_app, "st", fake_st)
    intro_app.main()
    return fake_st


def _metric(fake_st, label):
    return next(value for name, value in fake_st.metrics if name == label)


def test_main_renders_without_errors(monkeypatch):
    fake_st = _run(monkeypatch)

    assert fake_st.page_config["page_title"] == "Три задачи вводной лекции"
    assert not fake_st.errors
    # depth curve, length scatter, overflow bars, two pair rankings
    assert len(fake_st.figures) == 5


def test_the_defaults_are_the_slides_numbers(monkeypatch):
    fake_st = _run(monkeypatch)

    assert _metric(fake_st, "Точность дерева") == "0.631"
    assert _metric(fake_st, "n-граммы, Л.р. № 1") == "0.744"
    assert _metric(fake_st, "R² прямой") == "0.79"
    assert _metric(fake_st, "Длиннее на сегмент") == "+10%"
    assert _metric(fake_st, "Длиннее в сумме") == "+7%"
    assert _metric(fake_st, "Строк интерфейса не влезает") == "50%"
    assert _metric(fake_st, "Не реже 2 раз") == 23


def test_the_caption_names_the_idle_features(monkeypatch):
    fake_st = _run(monkeypatch)

    assert any("Не понадобились" in caption and "начинается с инфинитива" in caption for caption in fake_st.captions)
    assert any("не влезает 20 строк интерфейса из 40" in caption for caption in fake_st.captions)
    assert not any("«есть «" in caption for caption in fake_st.captions), "no guillemets inside guillemets"
    # «Save changes»: the forecast says 18 characters, the course's own translation takes 19.
    assert any("«Сохранить изменения» — 19 симв." in caption for caption in fake_st.captions)


def test_a_deeper_tree(monkeypatch):
    assert _metric(_run(monkeypatch, {DEPTH_LABEL: 4}), "Точность дерева") == "0.644"


def test_dropping_the_full_stop_is_explained(monkeypatch):
    rest = ["words", "infinitive", "placeholder", "polite_you", "legal_markers"]
    fake_st = _run(monkeypatch, {FEATURES_LABEL: rest})

    assert _metric(fake_st, "Точность дерева") == "0.506"
    assert any("Без финальной точки точность 0.506 вместо 0.631" in message for message in fake_st.infos)


def test_no_features_is_refused_politely(monkeypatch):
    fake_st = _run(monkeypatch, {FEATURES_LABEL: []})

    assert any("хотя бы один признак" in message for message in fake_st.warnings)
    assert not fake_st.errors


def test_even_half_again_leaves_one_button_in_ten_short(monkeypatch):
    """Interface strings grow by up to +47 % at the 90th percentile, so +50 % still misses some."""
    assert _metric(_run(monkeypatch, {MARGIN_LABEL: 50}), "Строк интерфейса не влезает") == "10%"


def test_pmi_without_a_threshold_is_warned_about(monkeypatch):
    fake_st = _run(monkeypatch, {MIN_COUNT_LABEL: 1})

    assert any("PMI любит редкое" in message for message in fake_st.warnings)
    assert _metric(fake_st, "Не реже 1 раза") == 741


def test_the_threshold_metric_reads_as_russian(monkeypatch):
    """«Не реже 1 раза», but «не реже 2 раз» and «5 раз»."""
    labels = {count: [name for name, _ in _run(monkeypatch, {MIN_COUNT_LABEL: count}).metrics] for count in (2, 5)}

    assert "Не реже 2 раз" in labels[2]
    assert "Не реже 5 раз" in labels[5]
