"""Russian numerals: the noun after a number changes with the number.

«41 сегмент», «82 сегмента», «87 сегментов». A format string with the plural
baked in reads fine for the default settings and breaks the moment a student
picks another classifier — which is how the browser, not the tests, caught it.
"""

from __future__ import annotations


def plural(count: int, one: str, few: str, many: str) -> str:
    """The form of a noun after ``count``: one for 1, 21, 101; few for 2–4, 22–24; many otherwise.

    Holds for the nominative and, for inanimate nouns, the accusative.
    """
    count = abs(int(count))
    if count % 10 == 1 and count % 100 != 11:
        return one
    if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
        return few
    return many


def segments(count: int) -> str:
    return f"{count} {plural(count, 'сегмент', 'сегмента', 'сегментов')}"
