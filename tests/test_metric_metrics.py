"""The NumPy metrics must equal scikit-learn's, and the shipped scores must give the lectures' numbers."""

import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as sk

from metric_playground.metrics import (
    REMEDIES,
    average_precision,
    cheapest_threshold,
    confusion,
    load_lecture4,
    load_lecture6,
    majority,
    nearest_recall,
    populations,
    precision_recall_curve,
)
from playground_common.wording import as_printed


@pytest.fixture(scope="module")
def lecture4():
    return load_lecture4()


@pytest.fixture(scope="module")
def lecture6():
    return load_lecture6()


def _all_scores(lecture4, lecture6):
    labels4, scores4 = lecture4
    labels6, scores6 = lecture6
    yield labels4, scores4
    for key in REMEDIES:
        yield labels6, scores6[key]


def test_the_shipped_scores_are_what_the_lectures_compute():
    from scripts.prepare_metric_data import lecture4_scores, lecture6_scores

    for name, fresh in (("lecture4_scores.csv", lecture4_scores()), ("lecture6_scores.csv", lecture6_scores())):
        shipped = pd.read_csv(f"metric_playground/data/{name}", float_precision="round_trip")
        pd.testing.assert_frame_equal(shipped, fresh, check_dtype=False)


def test_the_curve_equals_scikit_learns(lecture4, lecture6):
    for labels, scores in _all_scores(lecture4, lecture6):
        ours = precision_recall_curve(labels, scores)
        theirs = sk.precision_recall_curve(labels, scores)
        for mine, reference in zip(ours, theirs, strict=True):
            assert np.allclose(mine, reference)


def test_average_precision_equals_scikit_learns(lecture4, lecture6):
    for labels, scores in _all_scores(lecture4, lecture6):
        assert average_precision(labels, scores) == pytest.approx(sk.average_precision_score(labels, scores))


@pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_the_confusion_counts_equal_scikit_learns(lecture4, threshold):
    labels, scores = lecture4
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


def test_the_majority_answer_prints_lecture_4s_numbers(lecture4):
    labels, _ = lecture4
    counts = majority(labels)

    # «Доля отказов в выборке: 2.1%», «Доля правильных ответов: 0.979 <- выглядит отлично»
    assert f"{labels.mean():.1%}" == "2.1%"
    assert as_printed(counts.accuracy) == "0.979"
    assert (counts.precision, counts.recall, counts.f1) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("wanted", "printed"),
    [
        (0.6, "Полнота 0.59  ->  точность 0.63, порог 0.952"),
        (0.8, "Полнота 0.81  ->  точность 0.48, порог 0.871"),
        (0.95, "Полнота 0.94  ->  точность 0.08, порог 0.303"),
    ],
)
def test_the_three_operating_points_are_lecture_4s(lecture4, wanted, printed):
    labels, scores = lecture4
    point = nearest_recall(labels, scores, wanted)

    assert f"Полнота {point.recall:.2f}  ->  точность {point.precision:.2f}, порог {point.threshold:.3f}" == printed


@pytest.mark.parametrize(
    ("key", "printed"),
    [
        ("plain", "PR-AUC 0.615   полнота 0.367   точность 0.880"),
        ("weighted", "PR-AUC 0.622   полнота 0.333   точность 0.952"),
        ("undersampled", "PR-AUC 0.624   полнота 0.650   точность 0.476"),
    ],
)
def test_the_three_remedies_print_lecture_6s_lines(lecture6, key, printed):
    labels, scores = lecture6
    counts = confusion(labels, scores[key], 0.5)
    area = average_precision(labels, scores[key])
    line = f"PR-AUC {area:.3f}   полнота {counts.recall:.3f}   точность {counts.precision:.3f}"

    assert line == printed


def test_class_weights_lower_recall_in_lecture_6(lecture6):
    # The finding the page reports: the remedy works against its purpose here.
    labels, scores = lecture6

    assert confusion(labels, scores["weighted"], 0.5).recall < confusion(labels, scores["plain"], 0.5).recall


def test_the_remedies_barely_change_the_ranking(lecture6):
    labels, scores = lecture6
    areas = [average_precision(labels, scores[key]) for key in REMEDIES]

    assert max(areas) - min(areas) < 0.01


def test_secom_gives_the_rgr_baselines():
    secom = next(population for population in populations() if population.name.startswith("SECOM"))

    # Задание на РГР: «93,4 % — столько даёт ответ «годен» всегда», PR-AUC 0,066.
    assert f"{secom.majority_accuracy * 100:.1f}" == "93.4"
    assert f"{secom.pr_auc_baseline:.3f}" == "0.066"


def test_a_dearer_miss_never_raises_the_cheapest_threshold(lecture4):
    labels, scores = lecture4
    thresholds = [cheapest_threshold(labels, scores, miss)[0] for miss in (1, 3, 10, 30, 100)]

    assert thresholds == sorted(thresholds, reverse=True)


def test_the_cheapest_threshold_is_really_the_cheapest(lecture4):
    labels, scores = lecture4
    best, cost = cheapest_threshold(labels, scores, 10)

    for threshold in np.linspace(0, 1, 101):
        assert confusion(labels, scores, threshold).cost(10) >= cost
    assert confusion(labels, scores, best).cost(10) == cost
