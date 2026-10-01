"""The metric interface renders, and shows lectures 4 and 6 and the RGR baseline."""

import pytest

from metric_playground import app as metric_app


def _run(monkeypatch, overrides=None):
    from tests.fake_streamlit import FakeStreamlit

    fake_st = FakeStreamlit(overrides=overrides)
    monkeypatch.setattr(metric_app, "st", fake_st)
    metric_app.main()
    return fake_st


def _metric(fake_st, label, occurrence=0):
    return [value for name, value in fake_st.metrics if name == label][occurrence]


@pytest.fixture
def default_run(monkeypatch):
    return _run(monkeypatch)


def test_main_renders_without_errors(default_run):
    assert default_run.page_config["page_title"] == "Метрика и дисбаланс"
    assert not default_run.errors
    # confusion matrix, lecture 4 curve, lecture 6 curves
    assert len(default_run.figures) == 3


def test_the_trap_opens_on_lecture_4(default_run):
    assert _metric(default_run, "Доля правильных") == "97,9 %"
    assert _metric(default_run, "Найдено отказов") == "0"
    assert any("0.979  <- выглядит отлично" in message for message in default_run.infos)


def test_secom_shows_the_rgr_baseline(monkeypatch):
    fake_st = _run(monkeypatch, overrides={metric_app.POPULATION_LABEL: "SECOM: брак полупроводников (РГР)"})

    assert _metric(fake_st, "Доля правильных") == "93,4 %"
    assert _metric(fake_st, "PR-AUC случайной модели") == "0.066"
    assert any("задания на РГР" in message for message in fake_st.infos)


def test_the_threshold_tab_prints_the_lectures_three_points(default_run):
    table = next(text for text in default_run.markdowns if "нужна полнота" in text)

    assert "| 0.60 | 0.59 | 0.63 | 0.952 |" in table
    assert "| 0.80 | 0.81 | 0.48 | 0.871 |" in table
    assert "| 0.95 | 0.94 | 0.08 | 0.303 |" in table


def test_at_the_default_threshold_the_weighted_model_finds_almost_everything(default_run):
    # The first «Полнота» belongs to the trap tab, where it is zero by construction.
    assert _metric(default_run, "Полнота", 0) == "0.000"
    assert _metric(default_run, "Полнота", 1) == "0.938"
    assert _metric(default_run, "Точность", 0) == "0.136"


def test_the_price_of_a_miss_sets_the_cheapest_threshold(default_run):
    assert _metric(default_run, "Самый дешёвый порог") == "0.834"
    assert _metric(default_run, "Его стоимость") == "61"


def test_equal_prices_move_the_threshold_up(monkeypatch):
    fake_st = _run(monkeypatch, overrides={metric_app.MISS_COST_LABEL: 1})

    assert _metric(fake_st, "Самый дешёвый порог") == "0.946"


def test_the_remedies_table_repeats_lecture_6(default_run):
    table = next(text for text in default_run.markdowns if "| способ |" in text)

    assert "| Без учёта дисбаланса | 0.615 | 0.367 | 0.880 |" in table
    assert "| Веса классов | 0.622 | 0.333 | 0.952 |" in table
    assert "| Уменьшение большинства 3:1 | 0.624 | 0.650 | 0.476 |" in table
    assert any("Веса классов здесь не подняли полноту" in message for message in default_run.warnings)
    assert any("Лекция объясняет почему" in message for message in default_run.warnings)


def test_the_page_does_not_import_scikit_learn():
    # The page is meant to load without it. The tests themselves import it, so the
    # check reads the playground's sources instead of sys.modules.
    import ast
    from pathlib import Path

    for module in ("app.py", "metrics.py", "plotting.py"):
        tree = ast.parse(Path("metric_playground", module).read_text(encoding="utf-8"))
        imported = {
            alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
        } | {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        assert "sklearn" not in imported, module


def test_every_slider_opens_on_its_default(default_run):
    defaults = {label: value for label, value, _, _ in default_run.sliders}

    assert defaults == {
        metric_app.THRESHOLD_LABEL: 0.5,
        metric_app.MISS_COST_LABEL: 10,
        metric_app.REMEDY_THRESHOLD_LABEL: 0.5,
    }


def test_the_page_says_what_it_does_not_do(default_run):
    assert "Чего эта площадка не делает" in default_run.expanders
