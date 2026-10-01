"""Classification metrics in NumPy, written to give scikit-learn's numbers exactly.

The page never imports scikit-learn: everything it shows is computed here from
held-out labels and model scores prepared offline. The tests compare every
function with its scikit-learn counterpart, so "exactly" is checked, not hoped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"

# Three ways of handling the imbalance.
FAILURES = "Отказы оборудования"
RARE_CLASS = "Редкий класс"
SECOM = "SECOM: брак полупроводников (РГР)"

REMEDIES = {
    "plain": "Без учёта дисбаланса",
    "weighted": "Веса классов",
    "undersampled": "Уменьшение большинства 3:1",
}


def load_failures() -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(DATA / "failures_scores.csv", float_precision="round_trip")
    return table["label"].to_numpy(), table["score"].to_numpy()


def load_rare_class() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    table = pd.read_csv(DATA / "rare_scores.csv", float_precision="round_trip")
    return table["label"].to_numpy(), {key: table[key].to_numpy() for key in REMEDIES}


@dataclass(frozen=True)
class Confusion:
    """Counts at one threshold, and what follows from them."""

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def total(self) -> int:
        return self.true_positive + self.false_positive + self.false_negative + self.true_negative

    @property
    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / self.total

    @property
    def precision(self) -> float:
        """Share of alarms that were real. Zero when nothing was flagged, as ``zero_division=0`` gives."""
        flagged = self.true_positive + self.false_positive
        return self.true_positive / flagged if flagged else 0.0

    @property
    def recall(self) -> float:
        """Share of real failures found."""
        real = self.true_positive + self.false_negative
        return self.true_positive / real if real else 0.0

    @property
    def f1(self) -> float:
        both = self.precision + self.recall
        return 2 * self.precision * self.recall / both if both else 0.0

    def cost(self, miss: float, alarm: float = 1.0) -> float:
        return miss * self.false_negative + alarm * self.false_positive


def confusion(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Confusion:
    """Counts when everything scored at or above ``threshold`` is called a failure."""
    predicted = scores >= threshold
    actual = labels == 1
    return Confusion(
        true_positive=int(np.sum(predicted & actual)),
        false_positive=int(np.sum(predicted & ~actual)),
        false_negative=int(np.sum(~predicted & actual)),
        true_negative=int(np.sum(~predicted & ~actual)),
    )


def majority(labels: np.ndarray) -> Confusion:
    """The model that always answers "no failure"."""
    return confusion(labels, np.zeros(len(labels)), threshold=1.0)


def precision_recall_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``sklearn.metrics.precision_recall_curve`` without scikit-learn.

    Same algorithm, same ordering, same final (1, 0) point: thresholds are the
    distinct scores, ascending; precision and recall are one element longer.
    """
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    distinct = np.where(np.diff(sorted_scores))[0]
    cut = np.r_[distinct, sorted_labels.size - 1]
    true_positives = np.cumsum(sorted_labels)[cut]
    false_positives = 1 + cut - true_positives
    thresholds = sorted_scores[cut]

    flagged = true_positives + false_positives
    precision = np.divide(true_positives, flagged, out=np.zeros(len(true_positives)), where=flagged != 0)
    recall = np.ones(len(true_positives)) if true_positives[-1] == 0 else true_positives / true_positives[-1]
    return np.r_[precision[::-1], 1.0], np.r_[recall[::-1], 0.0], thresholds[::-1]


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """``sklearn.metrics.average_precision_score``: the step-wise area under the PR curve."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(-np.sum(np.diff(recall) * precision[:-1]))


@dataclass(frozen=True)
class Operating:
    """A point on the curve: where asking for a recall lands."""

    wanted: float
    recall: float
    precision: float
    threshold: float


def nearest_recall(labels: np.ndarray, scores: np.ndarray, wanted: float) -> Operating:
    """The curve point whose recall is closest to ``wanted``."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    index = int(np.argmin(np.abs(recall[:-1] - wanted)))
    return Operating(wanted, float(recall[index]), float(precision[index]), float(thresholds[index]))


def cheapest_threshold(labels: np.ndarray, scores: np.ndarray, miss: float, alarm: float = 1.0) -> tuple[float, float]:
    """The threshold that minimises total cost, and that cost.

    Every distinct score is a candidate, plus one above them all (flag nothing).
    """
    candidates = np.r_[np.unique(scores), np.inf]
    costs = [confusion(labels, scores, threshold).cost(miss, alarm) for threshold in candidates]
    best = int(np.argmin(costs))
    return float(candidates[best]), float(costs[best])


@dataclass(frozen=True)
class Population:
    """A data set described only by how many of its records are positive."""

    name: str
    total: int
    positive: int
    source: str

    @property
    def share(self) -> float:
        return self.positive / self.total

    @property
    def majority_accuracy(self) -> float:
        """Accuracy of always answering the common class."""
        return 1 - self.share

    @property
    def pr_auc_baseline(self) -> float:
        """PR-AUC of a model that ranks at random: the share of the positive class."""
        return self.share


def populations() -> tuple[Population, ...]:
    failure_labels, _ = load_failures()
    rare_labels, _ = load_rare_class()
    return (
        Population(FAILURES, len(failure_labels), int(failure_labels.sum()), "отложенная часть"),
        Population(RARE_CLASS, len(rare_labels), int(rare_labels.sum()), "отложенная часть"),
        # The RGR's data set, by its counts alone: «104 брака, дисбаланс около 14:1».
        Population(SECOM, 1567, 104, "весь набор"),
    )
