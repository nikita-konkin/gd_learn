"""The playground must reproduce lecture 6's numbers. That is the whole point.

If any of these fail, either the playground drifted or the lecture did. The
lecture is the source: check it before changing a number here.
"""

import numpy as np
import pytest

from leak_playground.experiment import (
    LECTURE_HONEST,
    LECTURE_LEAKY,
    TRUTH,
    all_p_values,
    chance_overlap,
    expected_false_positives,
    honest_score,
    leaky_score,
    scores,
    selection,
)
from leak_playground.noise import INFORMATIVE, NoiseSettings, make_data
from playground_common.wording import as_printed

LECTURE_SELECT = 20
LECTURE_FOLDS = 5


@pytest.fixture(scope="module")
def lecture_data():
    return make_data(NoiseSettings())


@pytest.fixture(scope="module")
def lecture_scores(lecture_data):
    matrix, labels = lecture_data
    return scores(matrix, labels, LECTURE_SELECT, LECTURE_FOLDS)


@pytest.fixture(scope="module")
def lecture_selection(lecture_data):
    matrix, labels = lecture_data
    return selection(matrix, labels, LECTURE_SELECT, LECTURE_FOLDS)


def test_the_data_is_the_lectures_data(lecture_data):
    matrix, labels = lecture_data
    generator = np.random.default_rng(1)

    assert np.array_equal(matrix, generator.normal(size=(300, 2000)))
    assert np.array_equal(labels, generator.integers(0, 2, 300))


def test_no_signal_means_the_two_classes_are_balanced(lecture_data):
    _, labels = lecture_data

    assert set(np.unique(labels).tolist()) == {0, 1}
    assert abs(labels.mean() - 0.5) < 0.1


def test_the_leaky_pipeline_prints_the_lectures_number(lecture_scores):
    assert as_printed(lecture_scores.leaky) == f"{LECTURE_LEAKY:.3f}"


def test_the_honest_pipeline_prints_the_lectures_number(lecture_scores):
    assert as_printed(lecture_scores.honest) == f"{LECTURE_HONEST:.3f}"


def test_the_honest_pipeline_lands_near_the_truth(lecture_scores):
    # There is nothing to predict, so an honest estimate has to be a coin toss.
    assert abs(lecture_scores.honest - TRUTH) < 0.05


def test_the_leak_is_worth_a_quarter_of_the_scale(lecture_scores):
    assert as_printed(lecture_scores.gap) == "0.247"


def test_the_leak_beats_the_truth_in_every_block(lecture_scores):
    assert all(value > TRUTH for value in lecture_scores.leaky_folds)
    assert len(lecture_scores.leaky_folds) == LECTURE_FOLDS


def test_the_branches_differ_only_in_where_they_select():
    # Asked to keep every column, the selection has nothing to leak, and the two
    # branches must then agree block for block. A scaler or a model present in
    # one branch only would break this — lecture 6 once had exactly that.
    matrix, labels = make_data(NoiseSettings(observations=60, features=30))

    assert leaky_score(matrix, labels, 30, LECTURE_FOLDS) == honest_score(matrix, labels, 30, LECTURE_FOLDS)


def test_the_selection_keeps_exactly_what_it_was_asked_for(lecture_selection):
    assert len(lecture_selection.indices) == LECTURE_SELECT
    assert len(set(lecture_selection.indices)) == LECTURE_SELECT
    assert lecture_selection.tested == 2000


def test_no_feature_survives_the_multiple_comparisons_correction(lecture_selection):
    # 2000 pure-noise columns, so about 100 clear a 0.05 threshold by luck.
    assert lecture_selection.significant_at_05 == 108
    assert abs(lecture_selection.significant_at_05 - expected_false_positives(2000)) < 30
    assert lecture_selection.significant_after_bonferroni == 0


def test_most_picks_do_not_survive_a_change_of_block(lecture_selection):
    # A column carrying real signal is re-picked by every block, so this is the
    # closest thing to evidence the selection has. It is not proof: exactly one
    # of the twenty reaches all five blocks here, on data with no signal at all.
    # Half of them are re-picked by two blocks or fewer.
    recurrence = lecture_selection.recurrence

    assert sum(1 for value in recurrence if value == LECTURE_FOLDS) == 1
    assert sum(1 for value in recurrence if value <= 2) == 11
    assert lecture_selection.pairwise_overlap == (5, 10, 4, 5, 6, 4, 5, 5, 5, 7)
    assert lecture_selection.mean_pairwise_overlap == pytest.approx(5.6)


def test_the_blocks_agree_more_than_chance_because_they_share_rows(lecture_selection):
    # Not evidence of signal: four fifths of any block's rows are the same rows.
    chance = chance_overlap(LECTURE_SELECT, 2000)

    assert chance == pytest.approx(0.2)
    assert lecture_selection.mean_pairwise_overlap > chance


def test_the_p_values_are_flat_as_pure_noise_requires(lecture_data):
    matrix, labels = lecture_data
    p_values = np.asarray(all_p_values(matrix, labels))

    assert p_values.size == 2000
    counts, _ = np.histogram(p_values, bins=10, range=(0.0, 1.0))
    # A uniform distribution puts 200 in each of ten bins; allow sampling noise.
    assert counts.min() > 140
    assert counts.max() < 260


def test_real_signal_lifts_the_honest_score_and_the_leak_still_adds_on_top():
    settings = NoiseSettings(signal=0.6)
    matrix, labels = make_data(settings)
    result = scores(matrix, labels, LECTURE_SELECT, LECTURE_FOLDS)

    # Five informative columns among two thousand: a modest but real lift, and
    # the leak still sits above it, which is the point the interface makes.
    assert as_printed(result.honest) == "0.630"
    assert as_printed(result.leaky) == "0.787"
    assert result.leaky > result.honest


def test_real_signal_is_what_survives_every_block():
    settings = NoiseSettings(signal=0.6)
    matrix, labels = make_data(settings)
    selected = selection(matrix, labels, LECTURE_SELECT, LECTURE_FOLDS)

    # All five informative columns are picked, and all five are re-picked by
    # every block — against one such column on pure noise.
    assert sum(1 for index in selected.indices if index < INFORMATIVE) == INFORMATIVE
    assert sum(1 for value in selected.recurrence if value == LECTURE_FOLDS) == 5


def test_signal_is_placed_only_in_the_informative_columns():
    plain, labels = make_data(NoiseSettings())
    shifted, shifted_labels = make_data(NoiseSettings(signal=0.5))

    assert np.array_equal(labels, shifted_labels)
    assert np.array_equal(plain[:, INFORMATIVE:], shifted[:, INFORMATIVE:])
    assert not np.array_equal(plain[:, :INFORMATIVE], shifted[:, :INFORMATIVE])


def test_honest_and_leaky_report_one_score_per_block(lecture_data):
    matrix, labels = lecture_data

    assert len(leaky_score(matrix, labels, 10, 3)) == 3
    assert len(honest_score(matrix, labels, 10, 3)) == 3
