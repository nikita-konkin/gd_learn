"""Clustering without labels, and the negative result that follows.

Executing the lab notebook prints ARI 0.079 for character n-grams and 0.172 for
the pretrained vectors, and this package reproduces both on the same machine.
Neither number is asserted here, because neither is portable: k-means reaches a
different local optimum under a different BLAS, and the run-to-run spread of the
character-n-gram figure (0.015 to 0.079 across eight seeds) is wider than the
gap between the two representations. Pinning one of them to three decimals
would be asserting a property of the linear-algebra library.

So these tests assert what the lab actually claims, which does hold everywhere:
the clusters barely agree with the labelling, better vectors move the number
without rescuing it, and reseeding moves it enough that a single value is not a
result.
"""

import pytest

from tm_playground.clustering import (
    agreement,
    agreement_across_seeds,
    agreement_table,
    cluster,
    corpus_map,
    crosstab,
)
from tm_playground.corpus import (
    CONTENT_TYPES,
    load_corpus,
    load_queries,
    load_query_vectors,
    load_segment_vectors,
)
from tm_playground.measures import CHAR_NGRAMS, EMBEDDINGS
from tm_playground.search import SearchIndex


@pytest.fixture(scope="module")
def index():
    return SearchIndex(load_corpus(), load_queries(), load_segment_vectors(), load_query_vectors())


@pytest.fixture(scope="module")
def labels(index):
    return index.corpus["type"]


@pytest.fixture(scope="module")
def table(index, labels):
    return agreement_table(index.representations(), labels)


def test_character_ngrams_barely_recover_the_labelling(index, labels):
    """The lab's second negative result: the clusters are not the content types."""
    score = agreement(labels, cluster(index.representations()[CHAR_NGRAMS]))

    assert score < 0.15, "anything higher would mean the negative result had gone away"


def test_pretrained_vectors_move_the_number_without_fixing_the_task(index, labels):
    """Better vectors shift the agreement upwards and leave it far from 1.0."""
    characters = agreement(labels, cluster(index.representations()[CHAR_NGRAMS]))
    pretrained = agreement(labels, cluster(index.representations()[EMBEDDINGS]))

    assert pretrained > characters
    assert pretrained < 0.45, "a shifted result is still nothing like a recovered labelling"


def test_better_vectors_win_on_a_typical_starting_point_not_just_a_lucky_one(index, labels):
    """The ordering survives reseeding, which is the part worth relying on."""
    characters = sorted(agreement_across_seeds(index.representations()[CHAR_NGRAMS], labels))
    pretrained = sorted(agreement_across_seeds(index.representations()[EMBEDDINGS], labels))
    middle = len(characters) // 2

    assert pretrained[middle] > characters[middle]
    assert min(pretrained) > min(characters)


def test_a_single_ari_is_not_a_result(index, labels):
    """Reseeding alone moves the character-n-gram number substantially.

    This is why the playground draws the spread rather than one bar: quoting a
    single ARI to three decimals on 160 segments says more about where k-means
    started than about the representation.
    """
    across = agreement_across_seeds(index.representations()[CHAR_NGRAMS], labels)

    assert max(across) - min(across) > 0.02


def test_the_table_covers_every_representation(table):
    assert set(table["representation"]) == {"char_ngrams", "lsa", "embeddings"}
    assert (table["ari_low"] <= table["ari"]).all()
    assert (table["ari"] <= table["ari_high"]).all()


def test_clusters_partition_the_whole_corpus(index, labels):
    counts = crosstab(labels, cluster(index.representations()[CHAR_NGRAMS]))

    assert counts.values.sum() == 160
    assert set(counts.columns) == set(CONTENT_TYPES)


def test_no_cluster_is_a_content_type_in_disguise(index, labels):
    """Every cluster mixes types — which is the whole point of the section."""
    counts = crosstab(labels, cluster(index.representations()[CHAR_NGRAMS]))

    assert ((counts > 0).sum(axis=1) > 1).all()


def test_the_map_gives_two_coordinates_per_segment(index):
    coordinates = corpus_map(index.representations()[CHAR_NGRAMS])

    assert coordinates.shape == (160, 2)
