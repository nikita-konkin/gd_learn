"""Russian numerals: «41 сегмент», «82 сегмента», «87 сегментов»."""

import pytest

from labels_playground.wording import plural, segments


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


def test_ties_are_rounded_the_way_the_lab_prints_them():
    """98/160 and 86/160: the lab's pandas tables show 0.612 and 0.538."""
    from labels_playground.wording import as_printed

    assert as_printed(98 / 160) == "0.612"
    assert as_printed(86 / 160) == "0.538"
    assert as_printed(0.53125) == "0.531"
