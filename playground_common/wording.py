"""Numbers as the courses' Russian text and their notebooks print them.

Two traps, each caught in a browser rather than by a test the first time:

* The noun after a number changes with the number: «41 сегмент», «82 сегмента»,
  «87 сегментов». A format string with the plural baked in reads fine for the
  default settings and breaks the moment a student moves a slider.
* Accuracies over 160 segments are multiples of 1/160, so rounding ties are
  common: 98/160 = 0.6125. The notebooks' pandas tables round half to even and
  print 0.612; a format string goes by the binary value and prints 0.613.
  Cross-validated accuracies over five folds land on ties just as often, and the
  ИСТ-51 lectures print what pandas prints too.
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


def features(count: int) -> str:
    return f"{count} {plural(count, 'признак', 'признака', 'признаков')}"


def observations(count: int) -> str:
    return f"{count} {plural(count, 'наблюдение', 'наблюдения', 'наблюдений')}"


def round_as_pandas(value: float, decimals: int = 3) -> float:
    """Round half to even, as ``DataFrame.round`` does in the lab notebooks."""
    return float(np.round(value, decimals))


def as_printed(value: float, decimals: int = 3) -> str:
    """Three decimals, as the lab's tables print them: ``as_printed(98 / 160) == "0.612"``."""
    return f"{round_as_pandas(value, decimals):.{decimals}f}"
