"""Learning curves on the course corpus, checked against lab 1.

Lab 1, section 12 draws the curve for character n-grams and logistic
regression; section 9 compares three classifiers at full size. Both are
reproduced here. The saved output of section 12 in the course repository still
shows ``FitFailedWarning`` — five fits on a single-class subset — but that
output predates the ``shuffle=True`` the cell now carries; executed today the
cell fits cleanly, and so must this.
"""

import numpy as np
import pytest

from labels_playground.corpus import load_corpus
from labels_playground.learning import (
    BAYES,
    CHARACTERS,
    CLASSIFIERS,
    CURVE_CLASSIFIERS,
    FEATURE_SETS,
    LOGISTIC,
    NEIGHBOURS,
    WORDS,
    baseline_accuracy,
    build_vectorizer,
    catch_up_size,
    empty_vectors,
    learning_curve_for,
)


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(scope="module")
def curves(corpus):
    return {
        (feature_set, classifier): learning_curve_for(corpus["ru_ref"], corpus["type"], feature_set, classifier)
        for feature_set in FEATURE_SETS
        for classifier in CLASSIFIERS
    }


def test_no_fit_fails_on_any_subset(curves):
    """A single-class subset would come back as NaN and silently drop a point."""
    for curve in curves.values():
        assert not np.isnan(curve.test).any()
        assert not np.isnan(curve.train).any()


def test_the_training_sizes_are_those_of_five_fold_cross_validation(curves):
    assert list(curves[(CHARACTERS, LOGISTIC)].sizes) == [25, 46, 66, 87, 107, 128]


def test_full_size_matches_the_labs_headline_numbers(curves):
    """0.531 for words, 0.744 for character n-grams — the numbers of lab 1."""
    assert curves[(WORDS, LOGISTIC)].final == pytest.approx(0.531, abs=0.0005)
    assert curves[(CHARACTERS, LOGISTIC)].final == pytest.approx(0.744, abs=0.0005)


def test_full_size_matches_section_9_for_every_classifier(curves):
    assert curves[(CHARACTERS, BAYES)].final == pytest.approx(0.613, abs=0.0005)
    assert curves[(CHARACTERS, NEIGHBOURS)].final == pytest.approx(0.550, abs=0.0005)


def test_the_curve_is_still_climbing_at_the_right_edge(curves):
    """Section 12's reading: the model has not run out of data to learn from."""
    test = curves[(CHARACTERS, LOGISTIC)].test

    assert test[-1] > test[-2] > test[-3]


def test_the_training_curve_stays_far_above_the_held_out_one(curves):
    curve = curves[(CHARACTERS, LOGISTIC)]

    assert (curve.train - curve.test).min() > 0.2


def test_better_features_are_worth_forty_segments(curves):
    """Characters on 87 segments already beat words on all 128."""
    assert catch_up_size(curves[(CHARACTERS, LOGISTIC)], curves[(WORDS, LOGISTIC)]) == 87


def test_the_weaker_the_model_the_more_the_features_are_worth(curves):
    gaps = {
        classifier: catch_up_size(curves[(CHARACTERS, classifier)], curves[(WORDS, classifier)])
        for classifier in CURVE_CLASSIFIERS
    }
    assert gaps == {LOGISTIC: 87, BAYES: 46}


def test_catch_up_is_none_when_the_better_curve_never_arrives(curves):
    assert catch_up_size(curves[(WORDS, LOGISTIC)], curves[(CHARACTERS, LOGISTIC)]) is None


def test_model_choice_moves_accuracy_less_than_feature_choice(curves):
    """Block 5's claim, and how narrowly it holds: 0.194 against 0.213."""
    finals = [curves[(CHARACTERS, classifier)].final for classifier in CLASSIFIERS]
    model_spread = max(finals) - min(finals)
    feature_spread = curves[(CHARACTERS, LOGISTIC)].final - curves[(WORDS, LOGISTIC)].final

    assert model_spread == pytest.approx(0.194, abs=0.001)
    assert feature_spread == pytest.approx(0.213, abs=0.001)
    assert model_spread < feature_spread


def test_baseline_is_one_quarter(corpus):
    assert baseline_accuracy(corpus["type"]) == pytest.approx(0.25)


def test_words_leave_a_quarter_of_the_held_out_sentences_empty(corpus):
    """Why words lose: 44 sentences share no word with their training part."""
    assert empty_vectors(corpus["ru_ref"], corpus["type"], WORDS) == 44
    assert empty_vectors(corpus["ru_ref"], corpus["type"], CHARACTERS) == 0


def _neighbours_accuracy(corpus, feature_set, reverse):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier

    texts, labels = corpus["ru_ref"].to_numpy(), corpus["type"].to_numpy()
    scores = []
    for train, test in StratifiedKFold(5).split(texts, labels):
        order = train[::-1] if reverse else train
        vectorizer = build_vectorizer(feature_set).fit(texts[order])
        model = KNeighborsClassifier(n_neighbors=5).fit(vectorizer.transform(texts[order]), labels[order])
        scores.append((model.predict(vectorizer.transform(texts[test])) == labels[test]).mean())
    return float(np.mean(scores))


def test_nearest_neighbours_on_words_is_decided_by_tie_breaking(corpus):
    """Why the curves do not offer kNN: on words, the order of the rows changes the score."""
    assert NEIGHBOURS not in CURVE_CLASSIFIERS
    assert _neighbours_accuracy(corpus, WORDS, reverse=False) != _neighbours_accuracy(corpus, WORDS, reverse=True)
    assert _neighbours_accuracy(corpus, CHARACTERS, reverse=False) == pytest.approx(
        _neighbours_accuracy(corpus, CHARACTERS, reverse=True)
    )
