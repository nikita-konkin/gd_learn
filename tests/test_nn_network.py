"""The network must print lecture 7's numbers, and each bug must do what the page says."""

import numpy as np
import pytest

from nn_playground.network import (
    BUGS,
    NO_BUG,
    RELU_BUG,
    SIGN_BUG,
    SIZE_BUG,
    gradient_check,
    load_moons,
    train,
)
from playground_common.wording import as_printed


@pytest.fixture(scope="module")
def moons():
    return load_moons()


@pytest.fixture(scope="module")
def lecture_run(moons):
    train_x, test_x, train_y, test_y = moons
    return train(train_x, train_y, test_x, test_y)


def test_the_csv_is_the_lectures_split(moons):
    from scripts.prepare_nn_data import lecture_split

    expected = lecture_split()
    for shipped, original in zip(moons, expected, strict=True):
        assert np.array_equal(shipped, original)


def test_the_split_has_the_lectures_sizes(moons):
    train_x, test_x, train_y, test_y = moons

    assert train_x.shape == (700, 2)
    assert test_x.shape == (300, 2)
    assert set(np.unique(train_y)) == {0, 1}


def test_the_loss_before_training_is_the_lectures(lecture_run):
    assert f"{lecture_run.initial_loss:.4f}" == "0.7102"


def test_the_accuracy_after_training_is_the_lectures(lecture_run):
    assert as_printed(lecture_run.test_accuracy) == "0.947"
    assert len(lecture_run.history) == 400


def test_the_lectures_single_gradient_comparison(moons):
    train_x, _, train_y, _ = moons
    first = gradient_check(train_x, train_y)[0]

    assert first.name == "W1[0, 0]"
    assert f"{first.analytic: .6f}" == "-0.002347"
    assert f"{first.numeric: .6f}" == "-0.002347"


def test_correct_formulas_agree_on_every_weight(moons):
    train_x, _, train_y, _ = moons
    components = gradient_check(train_x, train_y)

    # 2*5 + 5 + 5*1 + 1 weights in the lecture's 2-5-1 check network
    assert len(components) == 21
    assert all(component.agrees for component in components)
    assert max(component.relative_error for component in components) < 1e-8


def test_a_forgotten_relu_derivative_still_trains_plausibly(moons):
    train_x, test_x, train_y, test_y = moons
    broken = train(train_x, train_y, test_x, test_y, bug=RELU_BUG)

    # The lecture's warning, measured: it still learns, just worse.
    assert as_printed(broken.test_accuracy) == "0.887"
    assert broken.final_loss < broken.initial_loss


def test_a_forgotten_relu_derivative_is_invisible_from_the_second_layer(moons):
    train_x, _, train_y, _ = moons
    components = gradient_check(train_x, train_y, RELU_BUG)
    wrong = {component.name for component in components if not component.agrees}

    assert len(wrong) == 15
    # The second layer never passes through the ReLU derivative.
    assert all(not name.startswith(("W2", "b2")) for name in wrong)


@pytest.mark.parametrize("bug", [SIZE_BUG, SIGN_BUG])
def test_the_loud_bugs_break_training_outright(moons, bug):
    train_x, test_x, train_y, test_y = moons
    broken = train(train_x, train_y, test_x, test_y, bug=bug)
    components = gradient_check(train_x, train_y, bug)

    assert broken.final_loss > 10
    assert not any(component.agrees for component in components)


def test_a_flipped_sign_is_no_better_than_a_coin(moons):
    train_x, test_x, train_y, test_y = moons

    assert as_printed(train(train_x, train_y, test_x, test_y, bug=SIGN_BUG).test_accuracy) == "0.500"


def test_every_bug_is_caught_by_the_full_check(moons):
    train_x, _, train_y, _ = moons

    for bug in BUGS:
        caught = not all(component.agrees for component in gradient_check(train_x, train_y, bug))
        assert caught is (bug != NO_BUG), bug


def test_a_larger_step_learns_faster_here(moons):
    # Not a claim of the lecture, which fixes 0.5; recorded so a change is noticed.
    train_x, test_x, train_y, test_y = moons

    assert train(train_x, train_y, test_x, test_y, learning_rate=0.01).test_accuracy < 0.9
    assert train(train_x, train_y, test_x, test_y, learning_rate=2.0).test_accuracy > 0.947
