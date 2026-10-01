"""Numbers as the courses print them: Russian plurals and the notebooks' rounding."""

import pytest

from playground_common.wording import as_printed, features, observations, plural, round_as_pandas, segments


@pytest.mark.parametrize(
    "count, expected",
    [
        (1, "1 сегмент"),
        (21, "21 сегмент"),
        (41, "41 сегмент"),
        (101, "101 сегмент"),
        (2, "2 сегмента"),
        (24, "24 сегмента"),
        (82, "82 сегмента"),
        (103, "103 сегмента"),
        (0, "0 сегментов"),
        (5, "5 сегментов"),
        (87, "87 сегментов"),
        (128, "128 сегментов"),
    ],
)
def test_the_noun_agrees_with_the_number(count, expected):
    assert segments(count) == expected


@pytest.mark.parametrize("count", [11, 12, 13, 14, 111, 112, 114])
def test_the_teens_always_take_the_many_form(count):
    assert plural(count, "one", "few", "many") == "many"


def test_features_and_observations_agree_with_their_number():
    assert features(1) == "1 признак"
    assert features(20) == "20 признаков"
    assert features(2000) == "2000 признаков"
    assert observations(300) == "300 наблюдений"
    assert observations(102) == "102 наблюдения"


def test_ties_are_rounded_the_way_the_lab_prints_them():
    """98/160 and 86/160: the lab's pandas tables show 0.612 and 0.538."""
    assert as_printed(98 / 160) == "0.612"
    assert as_printed(86 / 160) == "0.538"
    assert as_printed(0.53125) == "0.531"


def test_as_printed_keeps_the_trailing_zero_and_takes_other_precisions():
    assert as_printed(0.51) == "0.510"
    assert as_printed(0.7102, 4) == "0.7102"


def test_round_as_pandas_matches_dataframe_round():
    import pandas as pd

    values = [98 / 160, 86 / 160, 0.53125, 0.4375, 0.0625]
    assert [round_as_pandas(value) for value in values] == pd.Series(values).round(3).tolist()
