"""MQM penalties against BLEU, on the ten segments lab 3 annotates itself."""

import numpy as np
import pytest

from labels_playground.corpus import load_annotations, load_corpus
from labels_playground.mqm import (
    MISSED_CRITICAL,
    REJECTED_CORRECT,
    acceptance,
    acceptance_errors,
    acceptance_sweep,
    clean_thresholds,
    lab_flags,
    penalties,
    penalty_by_type,
    rank_correlation,
)


@pytest.fixture(scope="module")
def annotations():
    return load_annotations()


@pytest.fixture(scope="module")
def median():
    return float(load_corpus()["bleu"].median())


def test_penalties_use_the_mqm_weights(annotations):
    assert list(penalties(annotations["severity"])) == [25, 25, 25, 25, 25, 5, 5, 5, 1, 0]


def test_section_7_flags_exactly_the_two_segments_the_lab_prints(annotations, median):
    """Executing lab 3 marks s002 «не заметила» and s065 «наказала», and nothing else."""
    flags = lab_flags(annotations, median)
    flagged = dict(zip(annotations.loc[flags != "", "id"], flags[flags != ""], strict=True))

    assert flagged == {"s002": "метрика не заметила", "s065": "метрика наказала"}


def test_at_the_median_one_mistake_of_each_kind(annotations, median):
    errors = acceptance_errors(annotations, round(median, 2))

    assert errors[MISSED_CRITICAL] == 1
    assert errors[REJECTED_CORRECT] == 1
    assert errors["accepted"] == 2


def test_no_bleu_threshold_is_right_on_all_ten(annotations):
    """The claim the acceptance tab is built on.

    Passing no critical error needs a threshold above s002's 0.630; accepting
    the correct s065 needs one at or below 0.156. Both cannot hold.
    """
    sweep = acceptance_sweep(annotations, np.round(np.arange(0, 1.0001, 0.01), 2))

    assert clean_thresholds(sweep).empty


def test_the_only_safe_thresholds_accept_nothing(annotations):
    sweep = acceptance_sweep(annotations, np.round(np.arange(0, 1.0001, 0.01), 2))
    safe = sweep[sweep["missed_critical"] == 0]

    assert safe["threshold"].min() == pytest.approx(0.63)
    assert (safe["accepted"] == 0).all()


def test_acceptance_labels_each_segment(annotations):
    outcome = dict(zip(annotations["id"], acceptance(annotations, 0.37), strict=True))

    assert outcome["s002"] == MISSED_CRITICAL
    assert outcome["s065"] == REJECTED_CORRECT
    assert outcome["s006"] == "принят"
    assert outcome["s012"] == "отклонён"


def test_bleu_does_not_rank_the_errors(annotations):
    """On these ten the rank correlation is +0.18: no relationship either way."""
    correlation = rank_correlation(annotations["bleu"], penalties(annotations["severity"]))

    assert correlation == pytest.approx(0.178, abs=0.001)
    assert abs(correlation) < 0.3


def test_rank_correlation_matches_the_definition_on_a_clean_case():
    assert rank_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert rank_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_interface_carries_most_of_the_penalty(annotations):
    by_type = penalty_by_type(annotations).set_index("type")

    assert by_type.loc["интерфейс", "penalty"] == 86
    assert by_type.loc["маркетинг", "penalty"] == 55
    assert by_type.loc["документация", "errors"] == 0
