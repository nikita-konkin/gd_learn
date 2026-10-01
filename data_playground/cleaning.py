"""A cleaning pipeline on six rows with planted defects, and a larger table to try it on.

The six rows carry four defects: missing values, a duplicate, inconsistent case
and a physically impossible reading. The usual warning is about order: compute
the fill statistic before removing the outliers, and the filling inserts a new
error instead of fixing the old one. This module makes the order a parameter,
so the warning can be checked instead of believed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Below this a received signal strength is physically impossible.
OUTLIER_LIMIT = -120.0
# The impossible value planted in the six rows, and in the larger table as well.
SENTINEL = -900.0

MEDIAN = "медианой"
MEAN = "средним"
STATISTICS = (MEDIAN, MEAN)

AFTER = "после удаления выбросов"
BEFORE = "до удаления выбросов"
ORDERS = (AFTER, BEFORE)

TRUE_LEVEL = -50.0
TRUE_SPREAD = 6.0
NODES = ("A", "B", "C")


def defect_table() -> pd.DataFrame:
    """Six readings, defects included."""
    return pd.DataFrame(
        {
            "узел": ["A", "B", "B", "C", None, "A"],
            "сигнал": [-45.0, -52.3, -52.3, None, -47.1, -900.0],  # -900 is the planted outlier
            "статус": ["ok", "OK", "OK", "fail", "ok", "ok"],
        }
    )


@dataclass(frozen=True)
class Inspection:
    """What an inspection before cleaning reports."""

    missing: dict[str, int]
    duplicates: int
    statuses: tuple[str, ...]


def inspect(raw: pd.DataFrame) -> Inspection:
    return Inspection(
        missing={column: int(count) for column, count in raw.isna().sum().items()},
        duplicates=int(raw.duplicated().sum()),
        statuses=tuple(raw["статус"].dropna().unique()) if "статус" in raw else (),
    )


@dataclass(frozen=True)
class Cleaned:
    """A cleaned table, the value used to fill gaps, and which cells were filled."""

    table: pd.DataFrame
    fill_value: float
    filled: tuple[bool, ...]

    @property
    def impossible_left(self) -> int:
        """Readings below the physical limit that survived — or were created by — cleaning."""
        return int((self.table["сигнал"] < OUTLIER_LIMIT).sum())

    @property
    def missing_left(self) -> int:
        return int(self.table.isna().sum().sum())


def _statistic(values: pd.Series, statistic: str) -> float:
    return float(values.median() if statistic == MEDIAN else values.mean())


def clean_signal(table: pd.DataFrame, statistic: str = MEDIAN, order: str = AFTER) -> Cleaned:
    """Turn impossible readings into gaps and fill every gap with one statistic.

    ``order`` decides one thing only: whether that statistic is computed from
    the signal after the impossible readings were removed (the right order)
    or before (the mistake its note warns about). Either way every gap —
    original or created by removing an outlier — gets the same value.
    """
    result = table.copy()
    signal = result["сигнал"]
    if order == BEFORE:
        fill_value = _statistic(signal, statistic)
    signal = signal.mask(signal < OUTLIER_LIMIT)
    if order == AFTER:
        fill_value = _statistic(signal, statistic)
    filled = signal.isna()
    result["сигнал"] = signal.fillna(fill_value)
    return Cleaned(table=result, fill_value=fill_value, filled=tuple(bool(flag) for flag in filled))


def clean_defect_table(statistic: str = MEDIAN, order: str = AFTER) -> Cleaned:
    """The whole chain: duplicates, case, rows without a node, then the signal."""
    tidy = (
        defect_table()
        .drop_duplicates()
        .assign(статус=lambda frame: frame["статус"].str.lower())
        .dropna(subset=["узел"])
    )
    return clean_signal(tidy, statistic, order)


@dataclass(frozen=True)
class ReadingsSettings:
    """What decides the larger table — and nothing about how it is cleaned."""

    rows: int = 1000
    missing: float = 0.10
    outliers: float = 0.05
    seed: int = 0


@dataclass(frozen=True)
class Readings:
    table: pd.DataFrame
    true_signal: np.ndarray
    missing_mask: np.ndarray
    outlier_mask: np.ndarray

    @property
    def true_mean(self) -> float:
        return float(self.true_signal.mean())


def make_readings(settings: ReadingsSettings) -> Readings:
    """Sensor readings with gaps and impossible values planted at the requested shares.

    Every random draw is made up front, at full length, whatever the shares are.
    Moving the share of gaps therefore changes which cells are gaps and nothing
    else: the true signal, the nodes and the uniform numbers behind the outliers
    stay put. A cell cannot be both a gap and an outlier, so an outlier takes
    precedence — raising the outlier share can only turn a gap into an outlier.
    """
    generator = np.random.default_rng(settings.seed)
    true_signal = generator.normal(TRUE_LEVEL, TRUE_SPREAD, settings.rows)
    nodes = generator.choice(NODES, settings.rows)
    outlier_draw = generator.random(settings.rows)
    missing_draw = generator.random(settings.rows)

    outlier_mask = outlier_draw < settings.outliers
    missing_mask = (missing_draw < settings.missing) & ~outlier_mask

    observed = true_signal.copy()
    observed[outlier_mask] = SENTINEL
    observed[missing_mask] = np.nan
    table = pd.DataFrame({"узел": nodes, "сигнал": observed})
    return Readings(table=table, true_signal=true_signal, missing_mask=missing_mask, outlier_mask=outlier_mask)


@dataclass(frozen=True)
class Estimate:
    """How far the cleaned table's mean lands from the truth."""

    true_mean: float
    cleaned_mean: float
    fill_value: float
    filled: int
    impossible_left: int

    @property
    def error(self) -> float:
        return self.cleaned_mean - self.true_mean


def estimate(readings: Readings, statistic: str, order: str) -> Estimate:
    cleaned = clean_signal(readings.table, statistic, order)
    return Estimate(
        true_mean=readings.true_mean,
        cleaned_mean=float(cleaned.table["сигнал"].mean()),
        fill_value=cleaned.fill_value,
        filled=sum(cleaned.filled),
        impossible_left=cleaned.impossible_left,
    )
