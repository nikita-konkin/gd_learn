"""The model-complexity playground must print lectures 4 and 5's numbers."""

import numpy as np
import pandas as pd
import pytest

from fit_playground.models import (
    ALPHAS,
    diabetes,
    fit_polynomial,
    mean_prediction_error,
    neighbours,
    penalty,
    polynomial_data,
    tree,
)
from playground_common.wording import as_printed


def test_the_polynomial_points_are_the_lectures():
    features, target = polynomial_data()
    generator = np.random.default_rng(3)
    expected = np.sort(generator.uniform(0, 1, 25))[:, None]

    assert np.array_equal(features, expected)
    assert target.shape == (25,)


@pytest.mark.parametrize(("degree", "printed"), [(1, "0.286"), (4, "0.059"), (17, "0.023")])
def test_the_training_errors_are_the_lectures(degree, printed):
    # Lecture 4 titles each panel «Степень N, ошибка на обучении X».
    assert f"{fit_polynomial(degree).train_error:.3f}" == printed


def test_the_flexible_polynomial_is_worse_than_the_mean_on_new_points():
    # The lecture's claim, measured: degree 17 against always answering the mean.
    fit = fit_polynomial(17)
    baseline = mean_prediction_error()

    # 13.155 here, 13.156 in the browser: degree 17 is badly conditioned, and the
    # page shows one decimal for that reason.
    assert fit.fresh_error == pytest.approx(13.155, abs=0.01)
    assert f"{baseline:.3f}" == "0.575"
    assert fit.fresh_error > 20 * baseline


def test_a_moderate_degree_generalises():
    assert fit_polynomial(4).fresh_error < 0.1


@pytest.mark.parametrize(
    ("count", "train", "cv"),
    [(1, "1.000", "0.950"), (15, "0.953", "0.957"), (99, "0.903", "0.847")],
)
def test_the_lectures_three_neighbour_counts(count, train, cv):
    model = neighbours(count)

    assert as_printed(model.train_accuracy) == train
    assert as_printed(model.cv_accuracy) == cv


def test_one_neighbour_overfits_cheaply_and_ninety_nine_underfit_dearly():
    one, fifteen, many = neighbours(1), neighbours(15), neighbours(99)

    assert fifteen.cv_accuracy - one.cv_accuracy < 0.01
    assert fifteen.cv_accuracy - many.cv_accuracy > 0.1


def test_the_lectures_tree_of_depth_three():
    model = tree(3)

    assert as_printed(model.train_accuracy) == "0.900"
    assert as_printed(model.cv_accuracy) == "0.890"


def test_an_unlimited_tree_memorises_its_data():
    assert tree(None).train_accuracy == 1.0
    assert tree(None).gap > 0.05


def test_the_csv_is_the_lectures_table():
    from scripts.prepare_fit_data import lecture_table

    shipped = pd.read_csv("fit_playground/data/diabetes.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(shipped, lecture_table(), check_dtype=False)


def test_the_split_is_the_lectures():
    train_x, test_x, _, _ = diabetes()

    assert (len(train_x), len(test_x)) == (309, 133)


def test_the_alpha_grid_is_the_lectures():
    assert np.allclose(ALPHAS, np.logspace(-2, 2, 30))


def test_at_the_largest_penalty_lasso_zeroes_everything():
    # Lecture 5 prints «При максимальном штрафе Lasso обнулил 10 признаков из 10».
    result = penalty(ALPHAS[-1])

    assert result.lasso_zeroed == 10
    assert f"{result.lasso_r2:.3f}" == "-0.006"


def test_the_collapse_starts_well_inside_the_lectures_grid():
    first = next(alpha for alpha in ALPHAS if penalty(alpha).lasso_zeroed == 10)

    assert f"{first:.3g}" == "53"


def test_a_middle_penalty_is_where_lasso_actually_selects():
    alpha = min(ALPHAS, key=lambda value: abs(value - 10))
    result = penalty(alpha)

    assert result.lasso_zeroed == 6
    assert result.lasso_r2 > 0.45


def test_ridge_only_halves_the_weights_over_the_same_grid():
    small = np.linalg.norm(penalty(ALPHAS[0]).ridge_weights)
    large = np.linalg.norm(penalty(ALPHAS[-1]).ridge_weights)

    assert 0.4 < large / small < 0.6
    assert f"{penalty(ALPHAS[-1]).ridge_r2:.3f}" == "0.478"
