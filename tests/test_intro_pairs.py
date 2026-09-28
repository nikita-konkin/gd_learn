"""Word pairs ranked by frequency and by PMI, checked against the slide «Поиск правил»."""

import pytest

from intro_playground.corpus import load_corpus
from intro_playground.pairs import LECTURE_MIN_COUNT, LECTURE_TOP, pair_table, tokenize, top_by_count, top_by_pmi


@pytest.fixture(scope="module")
def table():
    return pair_table(load_corpus()["ru_ref"])


def test_tokens_are_cyrillic_words_only():
    assert tokenize("Удалить файл «{name}»? 25 МБ, TLS 1.3") == ["удалить", "файл", "мб"]


def test_so_few_pairs_repeat_that_there_is_little_to_rank(table):
    assert len(table) == 741
    assert int((table["count"] >= LECTURE_MIN_COUNT).sum()) == 23


def test_frequency_puts_function_phrases_on_top(table):
    top = top_by_count(table, LECTURE_MIN_COUNT, LECTURE_TOP)

    assert (top.loc[0, "pair"], top.loc[0, "count"]) == ("в течение", 5)
    assert "по умолчанию" in set(top["pair"])


def test_pmi_raises_the_pairs_the_slide_names(table):
    top = top_by_pmi(table, LECTURE_MIN_COUNT, LECTURE_TOP)

    assert {"программное обеспечение", "настоящие условия"} <= set(top["pair"])
    assert top["pmi"].is_monotonic_decreasing


def test_the_two_rankings_share_three_of_eight(table):
    by_count = set(top_by_count(table, LECTURE_MIN_COUNT, LECTURE_TOP)["pair"])
    by_pmi = set(top_by_pmi(table, LECTURE_MIN_COUNT, LECTURE_TOP)["pair"])

    assert by_count & by_pmi == {"должен содержать", "доступ к", "убедитесь что"}


def test_without_a_threshold_pmi_ranks_pairs_seen_once(table):
    """The textbook failure of PMI, and the reason the slide keeps pairs seen at least twice."""
    assert set(top_by_pmi(table, 1, LECTURE_TOP)["count"]) == {1}
