"""Russian numerals: the noun after a number changes with the number.

«41 сегмент», «82 сегмента», «87 сегментов». A format string with the plural
baked in reads fine for the default settings and breaks the moment a student
picks another classifier — which is how the browser, not the tests, caught it.
"""

from __future__ import annotations

import numpy as np


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


def as_printed(value: float) -> str:
    """Three decimals, rounded the way the lab's pandas tables round them.

    Accuracies over 160 segments are multiples of 1/160, and ties are common:
    naive Bayes on character n-grams scores 98/160 = 0.6125. numpy rounds half
    to even and the lab prints 0.612; a format string alone would print 0.613.
    """
    return f"{np.round(value, 3):.3f}"
