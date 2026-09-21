"""Cohen's kappa by hand, and the chance model the interface draws."""

import math

import pytest
from sklearn.metrics import cohen_kappa_score

from labels_playground.agreement import (
    CATEGORIES,
    agreement,
    agreement_matrix,
    chance_agreement,
    chance_sweep,
    expected_raw_agreement,
    kappa_reading,
    label_shares,
)
from labels_playground.corpus import load_annotations

# Lab 3, the agreement cell, verbatim.
LAB_FIRST = ["разметка", "разметка", "разметка", "точность", "точность",
             "точность", "терминология", "стиль", "терминология", "нет"]  # fmt: skip
LAB_SECOND = ["разметка", "разметка", "разметка", "точность", "точность",
              "терминология", "терминология", "стиль", "стиль", "нет"]  # fmt: skip


def test_the_labs_example_gives_the_kappa_the_lab_prints():
    """Executing lab 3 prints «Каппа Коэна: 0.744»."""
    stats = agreement(LAB_FIRST, LAB_SECOND)

    assert stats.observed == pytest.approx(0.8)
    assert stats.expected == pytest.approx(0.22)
    assert stats.kappa == pytest.approx(0.744, abs=0.0005)


def test_the_shipped_annotations_reproduce_the_same_kappa():
    annotations = load_annotations()

    assert list(annotations["category"]) == LAB_FIRST
    assert list(annotations["category_second"]) == LAB_SECOND


@pytest.mark.parametrize(
    "first, second",
    [
        (LAB_FIRST, LAB_SECOND),
        (["нет"] * 8 + ["стиль", "точность"], ["нет"] * 9 + ["стиль"]),
        (["a", "b", "a", "b"], ["b", "a", "b", "a"]),
        (["a", "b", "c", "a", "b", "c"], ["a", "b", "c", "c", "a", "b"]),
    ],
)
def test_kappa_by_hand_agrees_with_sklearn(first, second):
    assert agreement(first, second).kappa == pytest.approx(cohen_kappa_score(first, second))


def test_a_rater_who_never_looks_scores_zero():
    """Always answering «нет» matches often and means nothing."""
    first = ["нет"] * 7 + ["точность", "стиль", "разметка"]
    lazy = ["нет"] * 10

    stats = agreement(first, lazy)
    assert stats.observed == pytest.approx(0.7)
    assert stats.kappa == pytest.approx(0.0)


def test_kappa_is_undefined_when_both_raters_use_one_label():
    stats = agreement(["нет"] * 5, ["нет"] * 5)

    assert math.isnan(stats.kappa)
    assert "не определена" in kappa_reading(stats.kappa)


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError):
        agreement(["нет"], ["нет", "стиль"])


def test_the_matrix_has_agreement_on_its_diagonal():
    matrix = agreement_matrix(LAB_FIRST, LAB_SECOND)

    assert int(matrix.values.sum()) == 10
    assert int(sum(matrix.loc[label, label] for label in matrix.index)) == 8
    assert list(matrix.index) == [label for label in CATEGORIES if label in set(LAB_FIRST)]


def test_label_shares_sum_to_one():
    for share in (0.0, 0.5, 0.8, 0.95):
        assert label_shares(share).sum() == pytest.approx(1.0)


def test_chance_agreement_climbs_with_the_dominant_label():
    """At 80 % error-free segments, strangers agree two times in three."""
    assert chance_agreement(0.8) == pytest.approx(0.648)
    assert chance_agreement(0.9) == pytest.approx(0.812)
    assert chance_agreement(0.5) < chance_agreement(0.8) < chance_agreement(0.9)


def test_kappa_recovers_the_true_agreement_whatever_the_skew():
    """The point of the chance tab: raw agreement moves with the skew, kappa does not."""
    sweep = chance_sweep(0.3, [0.2, 0.5, 0.8, 0.95])

    assert list(sweep["kappa"]) == pytest.approx([0.3] * 4)
    assert list(sweep["raw"]) == sorted(sweep["raw"])
    assert sweep["raw"].iloc[-1] > 0.9


def test_raw_agreement_of_strangers_is_pure_chance():
    assert expected_raw_agreement(0.0, 0.8) == pytest.approx(chance_agreement(0.8))
    assert expected_raw_agreement(1.0, 0.8) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kappa, reading",
    [
        (0.1, "согласия практически нет"),
        (0.35, "слабое"),
        (0.5, "умеренное"),
        (0.744, "существенное"),
        (0.9, "почти полное"),
    ],
)
def test_kappa_is_read_on_the_labs_scale(kappa, reading):
    assert kappa_reading(kappa) == reading
