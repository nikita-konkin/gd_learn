from tests.fake_streamlit import FakeStreamlit
from vec_playground import app as vec_app


def _run(monkeypatch, overrides=None):
    fake_st = FakeStreamlit(overrides=overrides)
    monkeypatch.setattr(vec_app, "st", fake_st)
    vec_app.main()
    return fake_st


def test_main_renders_without_errors(monkeypatch):
    fake_st = _run(monkeypatch)

    assert fake_st.page_config["page_title"] == "Векторизация текста"
    # n-gram curve, configuration comparison, confusion matrix
    assert len(fake_st.figures) == 3


def test_default_settings_reproduce_the_lab_number(monkeypatch):
    fake_st = _run(monkeypatch)
    accuracy = next(value for label, value in fake_st.metrics if "Кросс-валидация" in label)

    assert accuracy.startswith("0.53")


def test_reference_settings_report_nothing_flipped(monkeypatch):
    fake_st = _run(monkeypatch)

    assert any("эталонная настройка" in message for message in fake_st.infos)


def test_character_ngrams_beat_the_reference(monkeypatch):
    fake_st = _run(
        monkeypatch,
        overrides={
            "Из чего строим признаки": "символы внутри слов",
            "Длина n-граммы": (2, 4),
        },
    )
    accuracy = next(value for label, value in fake_st.metrics if "Кросс-валидация" in label)

    assert accuracy.startswith("0.74")
    assert not any("эталонная настройка" in message for message in fake_st.infos)
