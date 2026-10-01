"""Flexibility must cost what the page says it costs.

The claims are properties: training error falls with flexibility while error on
new data does not, a penalty strong enough zeroes every weight. The few exact
values left are facts about fixed data, recorded so that a change is noticed.
"""

import numpy as np
import pandas as pd

from fit_playground.models import (
    ALPHAS,
    DEFAULT_DEPTH,
    SHOWCASE_DEGREES,
    diabetes,
    fit_polynomial,
    mean_prediction_error,
    neighbours,
    penalty,
    polynomial_data,
    tree,
)


def test_the_polynomial_points_are_seeded():
    features, target = polynomial_data()
    generator = np.random.default_rng(3)
    expected = np.sort(generator.uniform(0, 1, 25))[:, None]

    assert np.array_equal(features, expected)
    assert target.shape == (25,)


def test_the_training_error_falls_as_the_degree_rises():
    errors = [fit_polynomial(degree).train_error for degree in SHOWCASE_DEGREES]

    assert errors == sorted(errors, reverse=True)
    # Below the noise variance, 0.25 squared: the flexible model has learned the noise.
    assert errors[-1] < 0.25**2


def test_the_flexible_polynomial_is_worse_than_the_mean_on_new_points():
    # Degree 17 against always answering the mean. The exact error differs in the
    # third decimal between platforms (degree 17 is badly conditioned); the page
    # shows one decimal for that reason.
    assert fit_polynomial(17).fresh_error > 10 * mean_prediction_error()


def test_a_moderate_degree_generalises():
    assert fit_polynomial(4).fresh_error < 0.1


def test_one_neighbour_is_perfect_on_its_own_points_only():
    one = neighbours(1)

    assert one.train_accuracy == 1.0
    assert one.cv_accuracy < 1.0


def test_one_neighbour_overfits_cheaply_and_ninety_nine_underfit_dearly():
    one, fifteen, many = neighbours(1), neighbours(15), neighbours(99)

    assert fifteen.cv_accuracy - one.cv_accuracy < 0.01
    assert fifteen.cv_accuracy - many.cv_accuracy > 0.1


def test_a_shallow_tree_neither_memorises_nor_overfits():
    model = tree(DEFAULT_DEPTH)

    assert model.train_accuracy < 1.0
    assert abs(model.gap) < 0.05


def test_an_unlimited_tree_memorises_its_data():
    assert tree(None).train_accuracy == 1.0
    assert tree(None).gap > 0.05


def test_the_csv_is_what_the_preparation_script_writes():
    from scripts.prepare_fit_data import diabetes_table

    shipped = pd.read_csv("fit_playground/data/diabetes.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(shipped, diabetes_table(), check_dtype=False)


def test_the_split_holds_out_thirty_per_cent():
    train_x, test_x, _, _ = diabetes()

    assert (len(train_x), len(test_x)) == (309, 133)


def test_the_alpha_grid_spans_four_orders_of_magnitude():
    assert np.allclose(ALPHAS, np.logspace(-2, 2, 30))


def test_at_the_largest_penalty_lasso_zeroes_everything():
    result = penalty(ALPHAS[-1])

    assert result.lasso_zeroed == 10
    # One answer for everyone: no better than the mean of the test part.
    assert abs(result.lasso_r2) < 0.02


def test_the_collapse_starts_well_inside_the_grid():
    first = next(alpha for alpha in ALPHAS if penalty(alpha).lasso_zeroed == 10)

    assert 10 < first < ALPHAS[-1]


def test_a_middle_penalty_is_where_lasso_actually_selects():
    alpha = min(ALPHAS, key=lambda value: abs(value - 10))
    result = penalty(alpha)

    assert 0 < result.lasso_zeroed < 10
    assert result.lasso_r2 > 0.45


def test_ridge_only_halves_the_weights_over_the_same_grid():
    small = np.linalg.norm(penalty(ALPHAS[0]).ridge_weights)
    large = np.linalg.norm(penalty(ALPHAS[-1]).ridge_weights)

    assert 0.4 < large / small < 0.6
    assert penalty(ALPHAS[-1]).ridge_r2 > 0.45
