"""One control, one cause.

The repository this template came from names its own failure: a seed slider that
governed both the train/held-out split and the generation of the data. Moving it
changed everything at once, so no conclusion followed from the experiment.

These tests enforce the separation structurally rather than by eye. The data
knobs live in ``NoiseSettings`` and reach only ``make_data``; the selection size
reaches only the selector; the number of blocks reaches only the splitter.
"""

import dataclasses
import inspect

import numpy as np

from leak_playground.experiment import _splitter, all_p_values, scores
from leak_playground.noise import NoiseSettings, make_data


def test_the_data_settings_hold_no_evaluation_knob():
    fields = {field.name for field in dataclasses.fields(NoiseSettings)}

    assert fields == {"observations", "features", "seed", "signal"}
    # Nothing about how the result is measured may travel with the data.
    assert not fields & {"select", "k", "folds", "cv", "split", "seed_split"}


def test_make_data_sees_nothing_but_the_data_settings():
    parameters = list(inspect.signature(make_data).parameters)

    assert parameters == ["settings"]


def test_make_data_is_a_pure_function_of_its_settings():
    settings = NoiseSettings(observations=120, features=300, seed=7, signal=0.3)
    first_matrix, first_labels = make_data(settings)
    second_matrix, second_labels = make_data(settings)

    assert np.array_equal(first_matrix, second_matrix)
    assert np.array_equal(first_labels, second_labels)


def test_the_splitter_depends_on_the_block_count_alone():
    parameters = list(inspect.signature(_splitter).parameters)

    assert parameters == ["folds"]


def test_the_split_does_not_look_at_the_feature_matrix():
    labels = np.array([0, 1] * 30)
    wide = np.random.default_rng(0).normal(size=(60, 50))
    narrow = np.random.default_rng(99).normal(size=(60, 3))

    from_wide = [tuple(train) for train, _ in _splitter(5).split(wide, labels)]
    from_narrow = [tuple(train) for train, _ in _splitter(5).split(narrow, labels)]

    assert from_wide == from_narrow


def test_changing_the_selection_size_leaves_the_data_untouched():
    settings = NoiseSettings()
    matrix, labels = make_data(settings)
    before = all_p_values(matrix, labels)

    scores(matrix, labels, 20, 5)
    scores(matrix, labels, 60, 5)

    # The per-column statistics describe the data, so nothing the selector does
    # may move them.
    assert all_p_values(matrix, labels) == before


def test_changing_the_block_count_leaves_the_data_untouched():
    settings = NoiseSettings()
    matrix, labels = make_data(settings)
    reference = matrix.copy()

    scores(matrix, labels, 20, 3)
    scores(matrix, labels, 20, 7)

    assert np.array_equal(matrix, reference)


def test_the_selection_size_moves_the_scores_and_the_block_count_moves_them_differently():
    matrix, labels = make_data(NoiseSettings())
    baseline = scores(matrix, labels, 20, 5)
    wider_selection = scores(matrix, labels, 60, 5)
    more_blocks = scores(matrix, labels, 20, 10)

    # Both controls act, and they act on different things: a different number of
    # selected features is a different model, a different number of blocks is a
    # different measurement of the same model.
    assert wider_selection.leaky != baseline.leaky
    assert more_blocks.leaky != baseline.leaky
    assert len(more_blocks.leaky_folds) == 10
    assert len(wider_selection.leaky_folds) == 5
