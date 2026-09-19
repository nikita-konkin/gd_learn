"""Searching the memory, checked against the numbers the lab produces.

The notebook was executed to get these: they are what a student sees, not what
the prose around them claims.
"""

import pytest

from tm_playground.corpus import (
    load_corpus,
    load_queries,
    load_query_vectors,
    load_segment_vectors,
)
from tm_playground.measures import CHAR_NGRAMS, EMBEDDINGS, LEVENSHTEIN, LSA
from tm_playground.search import (
    SearchIndex,
    comparison_table,
    divergences,
    suggestions,
    threshold_sweep,
)

SYNONYM_QUERY = "По вашему запросу результатов нет"
SYNONYM_ANSWER = "Ничего не найдено"


@pytest.fixture(scope="module")
def index():
    return SearchIndex(load_corpus(), load_queries(), load_segment_vectors(), load_query_vectors())


@pytest.fixture(scope="module")
def table(index):
    return comparison_table(index)


# --- the memory itself -----------------------------------------------------


def test_memory_and_vectors_line_up(index):
    assert len(index.segments) == 160
    assert index.segment_vectors.shape == (160, 384)
    assert index.query_vectors.shape == (len(index.queries), 384)


def test_every_representation_has_one_row_per_segment(index):
    for matrix in index.representations().values():
        assert matrix.shape[0] == 160


# --- one query at a time ---------------------------------------------------


def test_both_measures_agree_on_the_labs_first_example(index):
    query = "Сохранить все изменения"

    by_edits = index.best(query, LEVENSHTEIN)
    by_angle = index.best(query, CHAR_NGRAMS)

    assert by_edits.segment == by_angle.segment == "Сохранить изменения"
    assert by_edits.score == pytest.approx(82.6, abs=0.1)
    assert by_angle.score == pytest.approx(0.909, abs=0.001)


def test_reordered_words_split_the_measures(index):
    """Order-blindness wins this one outright."""
    query = "Настройки поиска"

    assert index.best(query, CHAR_NGRAMS).segment == "Поиск по настройкам"
    assert index.best(query, LEVENSHTEIN).segment != "Поиск по настройкам"


def test_a_short_segment_inside_a_long_query_splits_the_measures(index):
    query = "Пожалуйста, очистите кэш и повторите попытку"

    assert index.best(query, CHAR_NGRAMS).segment == "Очистить кэш"
    assert index.best(query, LEVENSHTEIN).segment != "Очистить кэш"


def test_nominalisation_gives_two_defensible_answers(index):
    """Both are reasonable and they differ grammatically — a person picks."""
    query = "Сохранение изменений"

    assert index.best(query, LEVENSHTEIN).segment == "Сохранить изменения"
    assert index.best(query, CHAR_NGRAMS).segment == "Изменения сохранены"


def test_neither_surface_measure_finds_a_synonym(index):
    """Same meaning, no shared words: both measures work on form, not sense."""
    answer = index.segments.index(SYNONYM_ANSWER)

    assert index.rank_of(SYNONYM_QUERY, LEVENSHTEIN, answer) > 3
    assert index.rank_of(SYNONYM_QUERY, CHAR_NGRAMS, answer) > 3
    assert index.rank_of(SYNONYM_QUERY, LSA, answer) > 3


def test_pretrained_vectors_find_the_synonym_but_not_in_first_place(index):
    """The lab's honest result: the cure works, and it works imperfectly."""
    answer = index.segments.index(SYNONYM_ANSWER)

    assert index.rank_of(SYNONYM_QUERY, EMBEDDINGS, answer) == 2
    assert index.best(SYNONYM_QUERY, EMBEDDINGS).segment != SYNONYM_ANSWER


def test_search_returns_as_many_hits_as_asked_in_ranked_order(index):
    hits = index.search("Сохранить все изменения", CHAR_NGRAMS, top_n=3)

    assert [hit.rank for hit in hits] == [1, 2, 3]
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


# --- what the browser cannot do -------------------------------------------


def test_embeddings_are_unavailable_for_text_the_student_types(index):
    assert index.supports("совершенно новый сегмент", EMBEDDINGS) is False
    assert index.scores("совершенно новый сегмент", EMBEDDINGS) is None
    assert index.search("совершенно новый сегмент", EMBEDDINGS) == []


def test_every_other_measure_handles_arbitrary_text(index):
    for measure in (LEVENSHTEIN, CHAR_NGRAMS, LSA):
        assert index.supports("совершенно новый сегмент", measure) is True
        assert len(index.scores("совершенно новый сегмент", measure)) == 160


def test_a_lab_query_is_recognised_despite_stray_whitespace(index):
    assert index.supports(f"  {SYNONYM_QUERY} ", EMBEDDINGS) is True


# --- all the queries at once ----------------------------------------------


def test_the_measures_agree_on_nine_queries_of_thirteen(table):
    assert len(table) == 13
    assert int(table["agree"].sum()) == 9


def test_the_four_divergences_are_the_ones_the_lab_discusses(table):
    diverging = set(divergences(table)["query"])

    assert diverging == {
        "Настройки поиска",
        "Сохранение изменений",
        "Пожалуйста, очистите кэш и повторите попытку",
        SYNONYM_QUERY,
    }


def test_every_query_carries_the_edit_it_demonstrates(table):
    assert (table["note"].str.len() > 0).all()


# --- the threshold --------------------------------------------------------


def test_raising_the_threshold_never_adds_suggestions(index):
    sweep = threshold_sweep(index)
    offered = list(sweep["offered"])

    assert offered == sorted(offered, reverse=True)
    assert (sweep["agreeing"] <= sweep["offered"]).all()


def test_the_industry_threshold_hides_almost_half_the_queries(index):
    """75% is a decision about the cost of a mistake, and it costs coverage."""
    sweep = threshold_sweep(index)
    at_75 = sweep.loc[sweep["threshold"] == 75.0].iloc[0]

    assert int(at_75["offered"]) == 7
    assert int(at_75["total"]) == 13


def test_a_stricter_threshold_keeps_only_the_near_identical(index):
    strict = suggestions(index, threshold=90.0)

    assert int(strict["offered"].sum()) == 2
    assert (strict.loc[strict["offered"], "percent"] >= 90).all()


def test_suggestions_are_listed_best_first(index):
    table = suggestions(index, threshold=75.0)

    assert list(table["percent"]) == sorted(table["percent"], reverse=True)
