"""The NumPy metrics must equal scikit-learn's, and the shipped scores must show what the page says."""

import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as sk

from metric_playground.metrics import (
    REMEDIES,
    average_precision,
    cheapest_threshold,
    confusion,
    load_failures,
    load_rare_class,
    majority,
    nearest_recall,
    populations,
    precision_recall_curve,
)


@pytest.fixture(scope="module")
def failures():
    return load_failures()


@pytest.fixture(scope="module")
def rare_class():
    return load_rare_class()


def _all_scores(failures, rare_class):
    failure_labels, failure_values = failures
    rare_labels, rare_scores = rare_class
    yield failure_labels, failure_values
    for key in REMEDIES:
        yield rare_labels, rare_scores[key]


def test_the_shipped_scores_are_what_the_preparation_script_computes():
    from scripts.prepare_metric_data import failure_scores, rare_class_scores

    for name, fresh in (("failures_scores.csv", failure_scores()), ("rare_scores.csv", rare_class_scores())):
        shipped = pd.read_csv(f"metric_playground/data/{name}", float_precision="round_trip")
        pd.testing.assert_frame_equal(shipped, fresh, check_dtype=False)


def test_the_curve_equals_scikit_learns(failures, rare_class):
    for labels, scores in _all_scores(failures, rare_class):
        ours = precision_recall_curve(labels, scores)
        theirs = sk.precision_recall_curve(labels, scores)
        for mine, reference in zip(ours, theirs, strict=True):
            assert np.allclose(mine, reference)


def test_average_precision_equals_scikit_learns(failures, rare_class):
    for labels, scores in _all_scores(failures, rare_class):
        assert average_precision(labels, scores) == pytest.approx(sk.average_precision_score(labels, scores))


@pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_the_confusion_counts_equal_scikit_learns(failures, threshold):
    labels, scores = failures
    predicted = (scores >= threshold).astype(int)
    counts = confusion(labels, scores, threshold)
    (tn, fp), (fn, tp) = sk.confusion_matrix(labels, predicted)

    assert (counts.true_negative, counts.false_positive, counts.false_negative, counts.true_positive) == (
        tn,
        fp,
        fn,
        tp,
    )
    assert counts.precision == pytest.approx(sk.precision_score(labels, predicted, zero_division=0))
    assert counts.recall == pytest.approx(sk.recall_score(labels, predicted))
    assert counts.f1 == pytest.approx(sk.f1_score(labels, predicted))


def test_always_answering_no_failure_looks_excellent_and_finds_nothing(failures):
    labels, _ = failures
    counts = majority(labels)

    assert counts.accuracy == pytest.approx(1 - labels.mean())
    assert counts.accuracy > 0.95
    assert (counts.precision, counts.recall, counts.f1) == (0.0, 0.0, 0.0)


def test_more_recall_is_bought_with_precision_and_a_lower_threshold(failures):
    labels, scores = failures
    points = [nearest_recall(labels, scores, wanted) for wanted in (0.6, 0.8, 0.95)]

    for point in points:
        assert abs(point.recall - point.wanted) < 0.02
    assert [point.precision for point in points] == sorted((point.precision for point in points), reverse=True)
    assert [point.threshold for point in points] == sorted((point.threshold for point in points), reverse=True)


def test_undersampling_moves_the_operating_point_towards_recall(rare_class):
    labels, scores = rare_class
    at_half = {key: confusion(labels, scores[key], 0.5) for key in REMEDIES}

    assert max(at_half, key=lambda key: at_half[key].recall) == "undersampled"
    assert min(at_half, key=lambda key: at_half[key].precision) == "undersampled"


def test_class_weights_lower_recall_here(rare_class):
    # The finding the page reports: the remedy works against its purpose here.
    labels, scores = rare_class

    assert confusion(labels, scores["weighted"], 0.5).recall < confusion(labels, scores["plain"], 0.5).recall


def test_the_remedies_barely_change_the_ranking(rare_class):
    labels, scores = rare_class
    areas = [average_precision(labels, scores[key]) for key in REMEDIES]

    assert max(areas) - min(areas) < 0.01


def test_secom_gives_the_rgr_baselines():
    secom = next(population for population in populations() if population.name.startswith("SECOM"))

    # Задание на РГР: «93,4 % — столько даёт ответ «годен» всегда», PR-AUC 0,066.
    assert f"{secom.majority_accuracy * 100:.1f}" == "93.4"
    assert f"{secom.pr_auc_baseline:.3f}" == "0.066"


def test_a_dearer_miss_never_raises_the_cheapest_threshold(failures):
    labels, scores = failures
    thresholds = [cheapest_threshold(labels, scores, miss)[0] for miss in (1, 3, 10, 30, 100)]

    assert thresholds == sorted(thresholds, reverse=True)


def test_the_cheapest_threshold_is_really_the_cheapest(failures):
    labels, scores = failures
    best, cost = cheapest_threshold(labels, scores, 10)

    for threshold in np.linspace(0, 1, 101):
        assert confusion(labels, scores, threshold).cost(10) >= cost
    assert confusion(labels, scores, best).cost(10) == cost
