"""Regression of translation length, checked against the slide «Сколько места займёт перевод»."""

import pandas as pd
import pytest

from intro_playground.corpus import CONTENT_TYPES, load_corpus
from intro_playground.length import (
    expansion_by_type,
    fit_line,
    lengths,
    mean_expansion,
    overflow_share,
    total_expansion,
)


@pytest.fixture(scope="module")
def table():
    return lengths(load_corpus())


def test_the_corpus_is_the_courses(table):
    """Same text as the corpus the translation-metrics playground checks against lab 3."""
    ours = load_corpus()
    theirs = pd.read_csv("mt_playground/data/loc_corpus.csv")[["id", "type", "en", "ru_ref"]]
    pd.testing.assert_frame_equal(ours, theirs)


def test_the_line_is_the_slides(table):
    """The figure prints ru = 0.91·en + 6.8 and R² = 0.79 (факты.json: 0.906, 6.8, 0.791)."""
    line = fit_line(table)

    assert line.slope == pytest.approx(0.906, abs=0.0005)
    assert line.intercept == pytest.approx(6.8, abs=0.05)
    assert line.r2 == pytest.approx(0.791, abs=0.0005)


def test_russian_is_ten_per_cent_longer_per_segment_but_seven_in_total(table):
    assert mean_expansion(table) == pytest.approx(0.103, abs=0.0005)
    assert total_expansion(table) == pytest.approx(0.066, abs=0.0005)


def test_the_spread_between_types_is_wider_than_the_correction(table):
    """The slide: interface +20 %, documentation +2 %."""
    by_type = expansion_by_type(table, CONTENT_TYPES).set_index("type")

    assert by_type.loc["интерфейс", "mean"] == pytest.approx(0.196, abs=0.0005)
    assert by_type.loc["документация", "mean"] == pytest.approx(0.022, abs=0.0005)
    assert by_type["segments"].tolist() == [40, 40, 40, 40]


def test_the_rule_of_thumb_leaves_half_the_buttons_short(table):
    shares = overflow_share(table, 0.15, CONTENT_TYPES)

    assert shares["интерфейс"] == pytest.approx(0.5)
    assert shares["документация"] == pytest.approx(0.175)


def test_nothing_overflows_a_generous_enough_margin(table):
    assert overflow_share(table, 10.0, CONTENT_TYPES).max() == 0.0
