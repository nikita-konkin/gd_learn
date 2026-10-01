"""The cleaning chain must reproduce lecture 2, and the order switch must do what the note says."""

import struct

import numpy as np
import pytest

from data_playground.cleaning import (
    AFTER,
    BEFORE,
    MEAN,
    MEDIAN,
    ReadingsSettings,
    clean_lecture_table,
    estimate,
    inspect,
    lecture_table,
    make_readings,
)
from data_playground.speed import memory, results_agree, sum_of_squares


def test_the_inspection_prints_what_the_lecture_prints():
    found = inspect(lecture_table())

    assert found.missing == {"узел": 1, "сигнал": 1, "статус": 0}
    assert found.duplicates == 1
    assert found.statuses == ("ok", "OK", "fail")


def test_the_lectures_order_fills_both_gaps_with_the_clean_median():
    cleaned = clean_lecture_table(MEDIAN, AFTER)

    assert cleaned.table["узел"].tolist() == ["A", "B", "C", "A"]
    assert cleaned.table["статус"].tolist() == ["ok", "ok", "fail", "ok"]
    assert cleaned.table["сигнал"].round(2).tolist() == [-45.0, -52.3, -48.65, -48.65]
    # «Пропусков осталось: 0»
    assert cleaned.missing_left == 0
    assert cleaned.filled == (False, False, True, True)


def test_after_the_outliers_are_gone_mean_and_median_agree_on_two_values():
    assert clean_lecture_table(MEAN, AFTER).fill_value == pytest.approx(-48.65)


def test_the_median_computed_too_early_is_wrong_but_quietly():
    cleaned = clean_lecture_table(MEDIAN, BEFORE)

    assert cleaned.fill_value == pytest.approx(-52.3)
    assert cleaned.impossible_left == 0


def test_the_mean_computed_too_early_inserts_impossible_values():
    # The lecture's note, measured: the cleaning creates a new error.
    cleaned = clean_lecture_table(MEAN, BEFORE)

    assert round(cleaned.fill_value, 2) == -332.43
    assert cleaned.impossible_left == 2


def test_the_larger_table_plants_what_it_says():
    readings = make_readings(ReadingsSettings())

    assert len(readings.table) == 1000
    assert int(readings.missing_mask.sum()) == 88
    assert int(readings.outlier_mask.sum()) == 59
    assert not np.any(readings.missing_mask & readings.outlier_mask)


@pytest.mark.parametrize(
    ("statistic", "order", "error"),
    [(MEDIAN, AFTER, "+0.00"), (MEAN, AFTER, "+0.02"), (MEDIAN, BEFORE, "-0.07"), (MEAN, BEFORE, "-8.06")],
)
def test_on_a_thousand_rows_only_the_early_mean_does_real_damage(statistic, order, error):
    result = estimate(make_readings(ReadingsSettings()), statistic, order)

    assert f"{result.error:+.2f}" == error


def test_enough_outliers_push_the_early_mean_below_the_physical_limit():
    result = estimate(make_readings(ReadingsSettings(outliers=0.10)), MEAN, BEFORE)

    assert result.fill_value < -120
    assert result.impossible_left == result.filled


def test_memory_matches_the_lecture_on_64_bit_python():
    if struct.calcsize("P") != 8:
        pytest.skip("the lecture's numbers are those of a 64-bit build")
    used = memory()

    assert f"{used.list_bytes / 1024**2:.1f}" == "3.4"
    assert f"{used.array_bytes / 1024**2:.1f}" == "0.8"
    assert f"{used.ratio:.0f}" == "5"


def test_the_three_sums_agree_and_numpy_wins():
    timings = sum_of_squares(100_000, repeats=1)

    assert results_agree(timings)
    generator, elementwise, dot = timings
    assert elementwise.seconds < generator.seconds
    assert dot.seconds < generator.seconds
