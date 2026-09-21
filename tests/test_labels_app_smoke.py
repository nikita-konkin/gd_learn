"""The data-and-annotation interface renders, and says the right thing."""

from labels_playground import app as labels_app
from labels_playground.corpus import load_annotations
from tests.fake_streamlit import FakeStreamlit

CLASSIFIER_LABEL = "Классификатор"
THRESHOLD_LABEL = "Порог приёмки по BLEU"
CATCH_UP_LABEL = "Символьные догоняют слова на объёме"
SHARE_LABEL = "Доля сегментов без ошибок"
TRUE_SHARE_LABEL = "На какой доле сегментов разметчики согласны по существу"


def _run(monkeypatch, overrides=None):
    fake_st = FakeStreamlit(overrides=overrides)
    monkeypatch.setattr(labels_app, "st", fake_st)
    labels_app.main()
    return fake_st


def _metric(fake_st, label, occurrence=0):
    return [value for name, value in fake_st.metrics if name == label][occurrence]


def _annotator_label(segment_id: str) -> str:
    row = load_annotations().set_index("id").loc[segment_id]
    return f"{segment_id} · {labels_app._short(row['ru_mt'])}"


def test_main_renders_without_errors(monkeypatch):
    fake_st = _run(monkeypatch)

    assert fake_st.page_config["page_title"] == "Данные и разметка"
    assert not fake_st.errors
    # learning curves, newsgroups, MQM scatter, acceptance sweep, agreement matrix, chance curves
    assert len(fake_st.figures) == 6


def test_the_first_tab_carries_lab_1s_numbers(monkeypatch):
    fake_st = _run(monkeypatch)

    assert _metric(fake_st, "Слова, TF-IDF, 128 сегментов") == "0.531"
    assert _metric(fake_st, "Символьные n-граммы 2–4, 128 сегментов") == "0.744"
    assert _metric(fake_st, CATCH_UP_LABEL) == "87 сегментов"
    assert any("ещё растёт" in message for message in fake_st.infos)
    assert any("У 44 из 160 предложений" in caption for caption in fake_st.captions)


def test_changing_the_classifier_changes_the_catch_up(monkeypatch):
    fake_st = _run(monkeypatch, overrides={CLASSIFIER_LABEL: "bayes"})

    assert _metric(fake_st, CATCH_UP_LABEL) == "46 сегментов"
    # 128 - 46 = 82: the saving is spelled with the few-form, not a baked-in plural.
    assert any("экономит 82 сегмента разметки" in caption for caption in fake_st.captions)


def test_at_the_median_bleu_makes_one_mistake_of_each_kind(monkeypatch):
    fake_st = _run(monkeypatch)

    assert _metric(fake_st, "Принято с критической ошибкой") == 1
    assert _metric(fake_st, "Отклонено без ошибок") == 1
    assert any("Порога, при котором приёмка по BLEU не ошибается" in message for message in fake_st.warnings)


def test_a_zero_threshold_waves_every_critical_error_through(monkeypatch):
    fake_st = _run(monkeypatch, overrides={THRESHOLD_LABEL: 0.0})

    assert _metric(fake_st, "Принято с критической ошибкой") == 5
    assert _metric(fake_st, "Отклонено без ошибок") == 0


def test_the_labs_two_annotators_give_the_labs_kappa(monkeypatch):
    fake_st = _run(monkeypatch)

    assert _metric(fake_st, "Совпали ответы") == "80%"
    assert _metric(fake_st, "Совпали бы случайно") == "22%"
    assert _metric(fake_st, "Каппа Коэна", 0) == "0.744"


def test_settling_one_dispute_raises_kappa(monkeypatch):
    fake_st = _run(monkeypatch, overrides={_annotator_label("s003"): "точность"})

    assert _metric(fake_st, "Совпали ответы") == "90%"
    assert _metric(fake_st, "Каппа Коэна", 0) == "0.870"


def test_strangers_agree_often_and_score_zero(monkeypatch):
    fake_st = _run(monkeypatch)

    assert _metric(fake_st, "Совпадение ответов") == "65%"
    assert _metric(fake_st, "Каппа Коэна", 1) == "0.00"


def test_raters_who_always_agree_score_one(monkeypatch):
    fake_st = _run(monkeypatch, overrides={TRUE_SHARE_LABEL: 1.0, SHARE_LABEL: 0.5})

    assert _metric(fake_st, "Совпадение ответов") == "100%"
    assert _metric(fake_st, "Каппа Коэна", 1) == "1.00"
