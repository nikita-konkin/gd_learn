"""One control, one cause, for the data-preparation playground."""

import dataclasses
import inspect as introspect

import numpy as np

from data_playground.cleaning import ReadingsSettings, clean_signal, make_readings


def test_the_data_settings_hold_nothing_about_cleaning():
    fields = {field.name for field in dataclasses.fields(ReadingsSettings)}

    assert fields == {"rows", "missing", "outliers", "seed"}


def test_cleaning_takes_the_table_and_two_choices_only():
    assert list(introspect.signature(clean_signal).parameters) == ["table", "statistic", "order"]


def test_the_share_of_gaps_moves_only_the_gaps():
    few = make_readings(ReadingsSettings(missing=0.05))
    many = make_readings(ReadingsSettings(missing=0.25))

    assert np.array_equal(few.true_signal, many.true_signal)
    assert np.array_equal(few.outlier_mask, many.outlier_mask)
    assert few.table["узел"].equals(many.table["узел"])
    assert many.missing_mask.sum() > few.missing_mask.sum()


def test_the_share_of_outliers_moves_the_outliers_and_never_adds_gaps():
    few = make_readings(ReadingsSettings(outliers=0.02))
    many = make_readings(ReadingsSettings(outliers=0.12))

    assert np.array_equal(few.true_signal, many.true_signal)
    # An outlier takes precedence over a gap, so gaps can only disappear.
    assert not np.any(many.missing_mask & ~few.missing_mask)
    assert many.outlier_mask.sum() > few.outlier_mask.sum()


def test_cleaning_does_not_modify_its_input():
    readings = make_readings(ReadingsSettings())
    before = readings.table.copy()

    clean_signal(readings.table)

    assert readings.table.equals(before)
