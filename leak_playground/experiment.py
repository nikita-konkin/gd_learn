"""The two pipelines, the features each one picks, and what the picks are worth.

The experiment: select the ``k`` best features by ANOVA F, then
cross-validate a logistic regression. Done once on the
whole sample the selection leaks the held-out rows into the fit; done inside a
pipeline it is redone on each training block and nothing leaks.

Everything here takes plain arrays and returns plain numbers, so it is testable
without a browser and cheap to cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Pure noise: two balanced classes, nothing to predict.
TRUTH = 0.5

MAX_ITER = 1000


def _model() -> LogisticRegression:
    return LogisticRegression(max_iter=MAX_ITER)


def _splitter(folds: int) -> StratifiedKFold:
    """The one splitter both halves of the experiment use.

    ``cross_val_score(cv=5)`` builds exactly this for a classifier. Spelling it
    out means the blocks a fold-by-fold selection sees are provably the blocks
    the scores came from.
    """
    return StratifiedKFold(n_splits=folds)


@dataclass(frozen=True)
class Scores:
    """Both pipelines' accuracy, per block and averaged."""

    leaky_folds: tuple[float, ...]
    honest_folds: tuple[float, ...]

    @property
    def leaky(self) -> float:
        return float(np.mean(self.leaky_folds))

    @property
    def honest(self) -> float:
        return float(np.mean(self.honest_folds))

    @property
    def gap(self) -> float:
        """How much the leak adds. On pure noise all of it is invented."""
        return self.leaky - self.honest


@dataclass(frozen=True)
class Selection:
    """The whole-sample selection, and how much of it survives a change of block.

    ``recurrence[i]`` is the number of blocks in which feature ``indices[i]`` was
    picked again when the selection was redone on that block's training rows.
    A feature carrying real signal is picked by every block; a feature that was
    merely lucky is picked by few.
    """

    indices: tuple[int, ...]
    f_scores: tuple[float, ...]
    p_values: tuple[float, ...]
    recurrence: tuple[int, ...]
    folds: int
    pairwise_overlap: tuple[int, ...]
    significant_at_05: int
    significant_after_bonferroni: int
    tested: int

    @property
    def mean_pairwise_overlap(self) -> float:
        """Average size of the intersection between two blocks' selections."""
        return float(np.mean(self.pairwise_overlap)) if self.pairwise_overlap else 0.0


def leaky_score(matrix: np.ndarray, labels: np.ndarray, k: int, folds: int) -> tuple[float, ...]:
    """Select on everything, then cross-validate. The wrong way, block by block.

    The scaler and the model are the honest branch's own, so the two branches
    differ in one thing only: where the selection is made. A scaler in one
    branch alone would mix a second difference into the comparison.
    """
    selected = SelectKBest(f_classif, k=k).fit_transform(matrix, labels)
    estimator = Pipeline([("scale", StandardScaler()), ("model", _model())])
    return tuple(cross_val_score(estimator, selected, labels, cv=_splitter(folds)))


def honest_score(matrix: np.ndarray, labels: np.ndarray, k: int, folds: int) -> tuple[float, ...]:
    """Select inside the pipeline, so each block selects from its own rows only."""
    estimator = Pipeline(
        [
            ("select", SelectKBest(f_classif, k=k)),
            ("scale", StandardScaler()),
            ("model", _model()),
        ]
    )
    return tuple(cross_val_score(estimator, matrix, labels, cv=_splitter(folds)))


def scores(matrix: np.ndarray, labels: np.ndarray, k: int, folds: int) -> Scores:
    return Scores(
        leaky_folds=leaky_score(matrix, labels, k, folds),
        honest_folds=honest_score(matrix, labels, k, folds),
    )


def _picks(matrix: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indices chosen, with the F statistic and p-value of every column."""
    selector = SelectKBest(f_classif, k=k).fit(matrix, labels)
    return np.flatnonzero(selector.get_support()), selector.scores_, selector.pvalues_


def selection(matrix: np.ndarray, labels: np.ndarray, k: int, folds: int) -> Selection:
    """What the whole-sample selection picked, and how reproducible those picks are."""
    indices, f_scores, p_values = _picks(matrix, labels, k)

    per_fold: list[set[int]] = []
    for train_index, _ in _splitter(folds).split(matrix, labels):
        fold_indices, _, _ = _picks(matrix[train_index], labels[train_index], k)
        per_fold.append(set(fold_indices.tolist()))

    recurrence = tuple(sum(index in picks for picks in per_fold) for index in indices.tolist())
    pairwise = tuple(
        len(first & second) for position, first in enumerate(per_fold) for second in per_fold[position + 1 :]
    )

    finite = p_values[np.isfinite(p_values)]
    tested = int(finite.size)
    bonferroni = 0.05 / tested if tested else 0.0

    return Selection(
        indices=tuple(int(index) for index in indices),
        f_scores=tuple(float(f_scores[index]) for index in indices),
        p_values=tuple(float(p_values[index]) for index in indices),
        recurrence=recurrence,
        folds=folds,
        pairwise_overlap=pairwise,
        significant_at_05=int((finite < 0.05).sum()),
        significant_after_bonferroni=int((finite < bonferroni).sum()),
        tested=tested,
    )


def chance_overlap(k: int, features: int) -> float:
    """Picks two independent selections would share if both chose at random."""
    return k * k / features if features else 0.0


def expected_false_positives(features: int, alpha: float = 0.05) -> float:
    """How many of ``features`` pure-noise columns clear ``alpha`` by luck alone."""
    return features * alpha


def all_p_values(matrix: np.ndarray, labels: np.ndarray) -> tuple[float, ...]:
    """The p-value of every column's ANOVA F test against the labels.

    On pure noise these are uniform on [0, 1]: a flat histogram is what "no
    signal" looks like, and the bar below 0.05 is a twentieth of the columns
    rather than a discovery.
    """
    _, p_values = f_classif(matrix, labels)
    return tuple(float(value) for value in p_values)
