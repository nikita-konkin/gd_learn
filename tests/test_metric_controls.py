"""One control, one cause, for the metric playground.

The threshold decides what the model answers; the price of a miss decides only
how those answers are valued. Moving the price must not move the counts.
"""

from metric_playground.metrics import cheapest_threshold, confusion, load_lecture4


def test_the_price_of_a_miss_does_not_change_the_counts():
    labels, scores = load_lecture4()
    counts = confusion(labels, scores, 0.5)

    for miss in (1, 10, 100):
        cheapest_threshold(labels, scores, miss)
        assert confusion(labels, scores, 0.5) == counts


def test_the_price_changes_only_the_valuation():
    labels, scores = load_lecture4()
    counts = confusion(labels, scores, 0.5)

    assert counts.cost(1) != counts.cost(10)
    assert counts.cost(10) == 10 * counts.false_negative + counts.false_positive


def test_the_threshold_moves_the_counts():
    labels, scores = load_lecture4()

    assert confusion(labels, scores, 0.3) != confusion(labels, scores, 0.9)
