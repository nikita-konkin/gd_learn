"""One control, one cause, for the model-complexity playground.

Each tab's computation takes exactly its own control and nothing else, so no
handle can reach into a neighbouring tab.
"""

import inspect

from fit_playground.models import fit_polynomial, neighbours, penalty, tree


def test_each_computation_takes_exactly_one_setting():
    for function, parameter in (
        (fit_polynomial, "degree"),
        (neighbours, "count"),
        (tree, "depth"),
        (penalty, "alpha"),
    ):
        assert list(inspect.signature(function).parameters) == [parameter], function.__name__


def test_the_data_does_not_depend_on_the_setting():
    from fit_playground.models import fresh_data, moons, polynomial_data

    first, again = polynomial_data(), polynomial_data()
    assert (first[0] == again[0]).all()
    assert (fresh_data()[0] == fresh_data()[0]).all()
    assert (moons()[0] == moons()[0]).all()


def test_the_new_points_have_their_own_generator():
    from fit_playground.models import FRESH_SEED, POLY_SEED

    assert FRESH_SEED != POLY_SEED
