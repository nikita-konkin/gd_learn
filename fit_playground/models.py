"""Model complexity three ways, each taken from the lecture that shows it.

* Lecture 4: polynomials of growing degree on twenty-five noisy points of a
  sine. The lecture reports the error on the training points only and states
  that the flexible model "на новых точках окажется хуже случайного". Here the
  new points are drawn, so the claim is measured.
* Lecture 5: nearest neighbours and a decision tree on two moons, where the
  number of neighbours and the depth set the flexibility.
* Lecture 5: Ridge and Lasso on the diabetes table, where the penalty does.

Every function returns plain numbers; the interface only draws them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA = Path(__file__).resolve().parent / "data" / "diabetes.csv"

# ----------------------------------------------------------------- lecture 4

POLY_SEED = 3
POLY_POINTS = 25
POLY_NOISE = 0.25
LECTURE_DEGREES = (1, 4, 17)
MAX_DEGREE = 20
# New points come from their own generator: drawing them from the lecture's
# would change nothing about the training data, but tying them to it would make
# one seed decide two things.
FRESH_SEED = 1003
FRESH_POINTS = 200


def truth(values: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * values)


def polynomial_data() -> tuple[np.ndarray, np.ndarray]:
    """Lecture 4's twenty-five points, drawn exactly as the lecture draws them."""
    generator = np.random.default_rng(POLY_SEED)
    features = np.sort(generator.uniform(0, 1, POLY_POINTS))[:, None]
    target = truth(features).ravel() + generator.normal(0, POLY_NOISE, features.shape[0])
    return features, target


def fresh_data() -> tuple[np.ndarray, np.ndarray]:
    """New points from the same law, never seen in training."""
    generator = np.random.default_rng(FRESH_SEED)
    features = generator.uniform(0, 1, FRESH_POINTS)[:, None]
    target = truth(features).ravel() + generator.normal(0, POLY_NOISE, FRESH_POINTS)
    return features, target


@dataclass(frozen=True)
class PolynomialFit:
    degree: int
    train_error: float
    fresh_error: float
    grid: tuple[float, ...]
    curve: tuple[float, ...]


def fit_polynomial(degree: int) -> PolynomialFit:
    features, target = polynomial_data()
    fresh_x, fresh_y = fresh_data()
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression()).fit(features, target)
    grid = np.linspace(0, 1, 300)[:, None]
    return PolynomialFit(
        degree=degree,
        train_error=float(mean_squared_error(target, model.predict(features))),
        fresh_error=float(mean_squared_error(fresh_y, model.predict(fresh_x))),
        grid=tuple(grid.ravel().tolist()),
        curve=tuple(model.predict(grid).tolist()),
    )


def mean_prediction_error() -> float:
    """Error on the new points of the dullest model: always answer the training mean."""
    _, target = polynomial_data()
    _, fresh_y = fresh_data()
    return float(np.mean((fresh_y - target.mean()) ** 2))


# ----------------------------------------------------------------- lecture 5, neighbours and trees

MOONS_POINTS = 300
MOONS_NOISE = 0.25
LECTURE_NEIGHBOURS = (1, 15, 99)
LECTURE_DEPTH = 3
FOLDS = 5
MESH = 120


def moons() -> tuple[np.ndarray, np.ndarray]:
    return make_moons(n_samples=MOONS_POINTS, noise=MOONS_NOISE, random_state=0)


@dataclass(frozen=True)
class Classifier:
    """Accuracy on the points it learned from, accuracy on held-out blocks, and its map."""

    train_accuracy: float
    cv_accuracy: float
    xs: tuple[float, ...]
    ys: tuple[float, ...]
    zone: tuple[tuple[float, ...], ...]

    @property
    def gap(self) -> float:
        """What the training accuracy overstates."""
        return self.train_accuracy - self.cv_accuracy


def _classifier(model) -> Classifier:
    features, target = moons()
    fitted = model.fit(features, target)
    xs = np.linspace(features[:, 0].min() - 0.5, features[:, 0].max() + 0.5, MESH)
    ys = np.linspace(features[:, 1].min() - 0.5, features[:, 1].max() + 0.5, MESH)
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    zone = fitted.predict_proba(np.c_[mesh_x.ravel(), mesh_y.ravel()])[:, 1].reshape(mesh_x.shape)
    return Classifier(
        train_accuracy=float(fitted.score(features, target)),
        cv_accuracy=float(cross_val_score(model, features, target, cv=FOLDS).mean()),
        xs=tuple(xs.tolist()),
        ys=tuple(ys.tolist()),
        zone=tuple(tuple(row) for row in zone.tolist()),
    )


def neighbours(count: int) -> Classifier:
    return _classifier(KNeighborsClassifier(n_neighbors=count))


def tree(depth: int | None) -> Classifier:
    return _classifier(DecisionTreeClassifier(max_depth=depth, random_state=0))


# ----------------------------------------------------------------- lecture 5, penalties

ALPHAS = tuple(float(alpha) for alpha in np.logspace(-2, 2, 30))
ZERO = 1e-8


def diabetes() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Lecture 5's split of the diabetes table."""
    table = pd.read_csv(DATA, float_precision="round_trip")
    features, target = table.drop(columns="target"), table["target"]
    return train_test_split(features, target, test_size=0.3, random_state=42)


@dataclass(frozen=True)
class Penalty:
    alpha: float
    ridge_weights: tuple[float, ...]
    lasso_weights: tuple[float, ...]
    ridge_r2: float
    lasso_r2: float

    @property
    def lasso_zeroed(self) -> int:
        return sum(1 for weight in self.lasso_weights if abs(weight) < ZERO)


def penalty(alpha: float) -> Penalty:
    train_x, test_x, train_y, test_y = diabetes()
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(train_x, train_y)
    lasso = make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=5000)).fit(train_x, train_y)
    return Penalty(
        alpha=alpha,
        ridge_weights=tuple(float(weight) for weight in ridge[-1].coef_),
        lasso_weights=tuple(float(weight) for weight in lasso[-1].coef_),
        ridge_r2=float(ridge.score(test_x, test_y)),
        lasso_r2=float(lasso.score(test_x, test_y)),
    )


def feature_names() -> tuple[str, ...]:
    return tuple(diabetes()[0].columns)
