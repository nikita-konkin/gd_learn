"""The interface renders, and the numbers it shows are lecture 6's numbers."""

import pytest

from leak_playground import app as leak_app

OBSERVATIONS_LABEL = leak_app.OBSERVATIONS_LABEL
FEATURES_LABEL = leak_app.FEATURES_LABEL
SIGNAL_LABEL = leak_app.SIGNAL_LABEL
SELECT_LABEL = leak_app.SELECT_LABEL
FOLDS_LABEL = leak_app.FOLDS_LABEL


def _run(monkeypatch, overrides=None):
    from tests.fake_streamlit import FakeStreamlit

    fake_st = FakeStreamlit(overrides=overrides)
    monkeypatch.setattr(leak_app, "st", fake_st)
    leak_app.main()
    return fake_st


def _metric(fake_st, label, occurrence=0):
    return [value for name, value in fake_st.metrics if name == label][occurrence]


@pytest.fixture
def default_run(monkeypatch):
    return _run(monkeypatch)


def test_main_renders_without_errors(default_run):
    assert default_run.page_config["page_title"] == "Утечка данных"
    assert not default_run.errors
    # per-block accuracy, recurrence, p-value histogram
    assert len(default_run.figures) == 3


def test_the_headline_carries_the_lectures_three_numbers(default_run):
    assert _metric(default_run, "С утечкой") == "0.757"
    assert _metric(default_run, "Честно") == "0.510"
    assert _metric(default_run, "Разрыв") == "+0.247"


def test_the_default_settings_announce_themselves_as_the_lectures(default_run):
    assert any("Настройка лекции 6" in message for message in default_run.infos)
    assert any("0.757" in message and "0.510" in message for message in default_run.infos)


def test_the_page_spells_out_what_it_ran_on(default_run):
    assert any("300 наблюдений, 2000 признаков" in caption for caption in default_run.captions)


def test_the_multiple_comparisons_numbers_are_on_the_page(default_run):
    assert _metric(default_run, "Признаков с p < 0.05") == "108 из 2000"
    assert _metric(default_run, "Выдержали поправку Бонферрони") == "0"


def test_the_page_reports_how_few_picks_survive_a_change_of_block(default_run):
    assert _metric(default_run, "Переотобраны во всех 5 блоках") == "1 из 20"
    assert _metric(default_run, "Настоящих признаков в данных") == "0"


def test_the_recurrence_caption_agrees_with_its_number(default_run):
    assert any("а 1 из 20 всё равно дотянулся до верха" in caption for caption in default_run.captions)


def test_the_page_says_the_branches_differ_in_one_thing(default_run):
    assert any("только тем, где сделан отбор" in caption for caption in default_run.captions)


def test_the_page_says_what_it_does_not_do(default_run):
    assert "Чего эта площадка не делает" in default_run.expanders


def test_a_changed_setting_stops_claiming_to_be_the_lecture(monkeypatch):
    fake_st = _run(monkeypatch, overrides={SELECT_LABEL: 40})

    assert not any("Настройка лекции 6" in message for message in fake_st.infos)
    assert _metric(fake_st, "С утечкой") != "0.757"


def test_fewer_blocks_relabel_the_recurrence_metric(monkeypatch):
    fake_st = _run(monkeypatch, overrides={FOLDS_LABEL: 3})

    assert _metric(fake_st, "Переотобраны во всех 3 блоках").endswith("из 20")
    assert any("блоков 3" in caption for caption in fake_st.captions)


def test_real_signal_changes_what_the_page_says_about_the_gap(monkeypatch):
    fake_st = _run(monkeypatch, overrides={SIGNAL_LABEL: 0.6})

    assert _metric(fake_st, "Честно") == "0.630"
    assert _metric(fake_st, "Из них настоящих") == "5 из 5"
    assert any("Сигнал подмешан" in caption for caption in fake_st.captions)
    assert not any("придуман отбором" in caption for caption in fake_st.captions)


def test_narrower_data_leaves_less_room_for_luck(monkeypatch):
    fake_st = _run(monkeypatch, overrides={FEATURES_LABEL: 200})

    # Twenty picks out of two hundred columns instead of two thousand: much less
    # scope to find a lucky one, so the invented gap shrinks.
    assert float(_metric(fake_st, "Разрыв")) < 0.247


def test_every_slider_opens_on_the_lectures_setting(default_run):
    # A default that is not the lecture's would open the page on numbers that
    # match nothing in the course materials.
    defaults = {label: value for label, value, _, _ in default_run.sliders}

    assert defaults == {
        OBSERVATIONS_LABEL: 300,
        FEATURES_LABEL: 2000,
        SIGNAL_LABEL: 0.0,
        leak_app.SEED_LABEL: 1,
        SELECT_LABEL: 20,
        FOLDS_LABEL: 5,
    }


def test_every_slider_can_reach_both_sides_of_its_default(default_run):
    for label, value, low, high in default_run.sliders:
        assert low <= value <= high, label
        assert low < high, label
