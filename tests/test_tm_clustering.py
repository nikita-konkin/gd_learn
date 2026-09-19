"""Clustering without labels, and the negative result that follows.

The ARI values here come from executing the lab notebook, which prints 0.079
for character n-grams and 0.172 for the pretrained vectors. The tolerances are
loose enough to survive a scikit-learn point release and tight enough that a
real change in behaviour fails the run.
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
    """The lab's second negative result, reproduced."""
    score = agreement(labels, cluster(index.representations()[CHAR_NGRAMS]))

    assert score == pytest.approx(0.079, abs=0.02)


def test_pretrained_vectors_move_the_number_without_fixing_the_task(index, labels):
    score = agreement(labels, cluster(index.representations()[EMBEDDINGS]))

    assert score == pytest.approx(0.172, abs=0.03)
    assert score < 0.4, "a shifted result is still nothing like a recovered labelling"


def test_better_vectors_beat_worse_ones_on_every_starting_point(index, labels):
    """The ordering is solid even though either single number is not."""
    characters = agreement_across_seeds(index.representations()[CHAR_NGRAMS], labels)
    pretrained = agreement_across_seeds(index.representations()[EMBEDDINGS], labels)

    assert max(characters) < min(pretrained)


def test_a_single_ari_is_not_a_result(index, labels):
    """Reseeding moves the character-n-gram number by more than its own size.

    This is why the playground draws the spread: quoting one ARI to three
    decimals on 160 segments says more about the seed than the representation.
    """
    across = agreement_across_seeds(index.representations()[CHAR_NGRAMS], labels)

    assert max(across) - min(across) > max(across) / 2


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
