"""The two surface measures, checked against the values printed in the lab."""

import numpy as np
import pytest

from tm_playground.measures import levenshtein, match_percent
from tm_playground.search import rank_descending


def test_substitution_insertion_and_deletion_each_cost_one():
    assert levenshtein("кот", "кит") == 1
    assert levenshtein("кот", "скот") == 1
    assert levenshtein("скот", "кот") == 1


def test_distance_is_symmetric_and_zero_on_identity():
    assert levenshtein("Сохранить", "Сохранять") == levenshtein("Сохранять", "Сохранить")
    assert levenshtein("Сохранить", "Сохранить") == 0


def test_match_percent_reproduces_the_lab():
    """The three lines the notebook prints right after defining it."""
    assert match_percent("Сохранить изменения", "Сохранить все изменения") == pytest.approx(82.6, abs=0.1)
    assert match_percent("Сохранить изменения", "Отменить изменения") == pytest.approx(73.7, abs=0.1)
    assert match_percent("Сохранить изменения", "Соединение восстановлено") == pytest.approx(25.0, abs=0.1)


def test_identical_strings_score_a_hundred():
    assert match_percent("Очистить кэш", "Очистить кэш") == 100.0
    assert match_percent("", "") == 100.0


def test_a_short_segment_inside_a_long_query_is_penalised_by_length_alone():
    """Dividing by the longer string is why this measure misses that case.

    The memory segment is contained in the query almost word for word, and the
    percentage still lands nowhere near the industry cut-off.
    """
    query = "Пожалуйста, очистите кэш и повторите попытку"

    assert match_percent(query, "Очистить кэш") < 40


def test_word_order_costs_as_much_as_rewriting():
    """Swapping two words is cheap for a person and expensive for this measure."""
    assert match_percent("Настройки поиска", "Поиск по настройкам") < 50


def test_ranking_is_descending_with_ties_going_to_the_earlier_row():
    scores = np.array([0.5, 0.9, 0.9, 0.1])

    assert rank_descending(scores, 3) == [1, 2, 0]


def test_ranking_never_returns_more_than_asked():
    assert len(rank_descending(np.arange(160, dtype=float), 3)) == 3
