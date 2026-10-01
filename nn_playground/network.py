"""Lecture 7's two-layer network on NumPy, plus three ways to get the backward pass wrong.

``initialise``, ``forward``, ``loss`` and ``backward`` are the lecture's code.
The playground's addition is ``BUGS``: each replaces one line of ``backward``
with a mistake people really make. The lecture warns that such a network
"всё равно будет обучаться, но медленно и не в ту сторону", and that from the
outside it looks like bad data. The gradient check is what tells them apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data" / "moons.csv"

NO_BUG = "нет, как в лекции"
RELU_BUG = "забыта производная ReLU"
SIZE_BUG = "градиент не поделён на размер выборки"
SIGN_BUG = "перепутан знак градиента"
BUGS = (NO_BUG, RELU_BUG, SIZE_BUG, SIGN_BUG)

# The lecture's numerical check: a 2-5-1 network, seed 1, on the first fifty
# training rows, central differences with this step.
CHECK_HIDDEN = 5
CHECK_SEED = 1
CHECK_ROWS = 50
CHECK_STEP = 1e-5
# Relative disagreement above which a component is reported as wrong.
TOLERANCE = 1e-6


def load_moons() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train and test parts of lecture 7's two moons, in the lecture's row order."""
    table = pd.read_csv(DATA, float_precision="round_trip")
    train = table[table["part"] == "train"]
    test = table[table["part"] == "test"]
    return (
        train[["x1", "x2"]].to_numpy(),
        test[["x1", "x2"]].to_numpy(),
        train["label"].to_numpy(),
        test["label"].to_numpy(),
    )


def initialise(inputs: int, hidden: int, outputs: int, seed: int = 0) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)
    # The scale matters: weights that start too large saturate the units.
    return {
        "W1": generator.normal(0, np.sqrt(2 / inputs), (inputs, hidden)),
        "b1": np.zeros(hidden),
        "W2": generator.normal(0, np.sqrt(2 / hidden), (hidden, outputs)),
        "b2": np.zeros(outputs),
    }


def forward(params: dict[str, np.ndarray], batch: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    hidden_raw = batch @ params["W1"] + params["b1"]
    hidden = np.maximum(0, hidden_raw)  # ReLU
    output_raw = hidden @ params["W2"] + params["b2"]
    # The lecture's sigmoid, as written. A broken backward pass drives the
    # outputs far enough for exp to overflow; the result is still a correct 0
    # or 1, so the warning is silenced rather than the formula changed.
    with np.errstate(over="ignore"):
        probability = 1 / (1 + np.exp(-output_raw)).ravel()
    return probability, {"hidden_raw": hidden_raw, "hidden": hidden}


def loss(probability: np.ndarray, labels: np.ndarray) -> float:
    """Cross-entropy; the clip guards against log(0)."""
    probability = np.clip(probability, 1e-9, 1 - 1e-9)
    return float(-np.mean(labels * np.log(probability) + (1 - labels) * np.log(1 - probability)))


def backward(params, batch, labels, probability, cache, bug: str = NO_BUG) -> dict[str, np.ndarray]:
    """Lecture 7's gradients by the chain rule, with at most one line broken."""
    size = batch.shape[0]
    output_grad = (probability - labels).reshape(-1, 1)
    if bug != SIZE_BUG:
        output_grad = output_grad / size
    if bug == SIGN_BUG:
        output_grad = -output_grad
    weights_second_grad = cache["hidden"].T @ output_grad
    bias_second_grad = output_grad.sum(axis=0)
    hidden_grad = output_grad @ params["W2"].T
    if bug == RELU_BUG:
        hidden_raw_grad = hidden_grad
    else:
        hidden_raw_grad = hidden_grad * (cache["hidden_raw"] > 0)  # derivative of ReLU
    weights_first_grad = batch.T @ hidden_raw_grad
    bias_first_grad = hidden_raw_grad.sum(axis=0)
    return {"W1": weights_first_grad, "b1": bias_first_grad, "W2": weights_second_grad, "b2": bias_second_grad}


@dataclass(frozen=True)
class Component:
    """One weight of the check network: what the formula says and what the slope says."""

    name: str
    analytic: float
    numeric: float

    @property
    def relative_error(self) -> float:
        scale = abs(self.analytic) + abs(self.numeric)
        return abs(self.analytic - self.numeric) / scale if scale > 1e-12 else 0.0

    @property
    def agrees(self) -> bool:
        return self.relative_error < TOLERANCE


def gradient_check(train_x: np.ndarray, train_y: np.ndarray, bug: str = NO_BUG) -> tuple[Component, ...]:
    """The lecture's numerical check, run on every weight instead of one.

    The lecture compares ``W1[0, 0]`` only, and that one comparison is enough to
    trust correct formulas. Checking all twenty-one weights is what catches the
    mistakes that leave some components right.
    """
    params = initialise(2, CHECK_HIDDEN, 1, seed=CHECK_SEED)
    batch, labels = train_x[:CHECK_ROWS], train_y[:CHECK_ROWS]
    probability, cache = forward(params, batch)
    gradients = backward(params, batch, labels, probability, cache, bug)

    components = []
    for key in ("W1", "b1", "W2", "b2"):
        for index in np.ndindex(params[key].shape):
            plus = {name: value.copy() for name, value in params.items()}
            minus = {name: value.copy() for name, value in params.items()}
            plus[key][index] += CHECK_STEP
            minus[key][index] -= CHECK_STEP
            numeric = (loss(forward(plus, batch)[0], labels) - loss(forward(minus, batch)[0], labels)) / (
                2 * CHECK_STEP
            )
            label = f"{key}[{', '.join(str(position) for position in index)}]"
            components.append(Component(label, float(gradients[key][index]), float(numeric)))
    return tuple(components)


@dataclass(frozen=True)
class Training:
    """What a training run produced."""

    history: tuple[float, ...]
    params: dict[str, np.ndarray]
    test_accuracy: float

    @property
    def initial_loss(self) -> float:
        return self.history[0]

    @property
    def final_loss(self) -> float:
        return self.history[-1]


def train(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    hidden: int = 16,
    learning_rate: float = 0.5,
    epochs: int = 400,
    seed: int = 0,
    bug: str = NO_BUG,
) -> Training:
    """Lecture 7's training loop: forward, loss, gradients, one step — ``epochs`` times."""
    params = initialise(2, hidden, 1, seed=seed)
    history = []
    for _ in range(epochs):
        probability, cache = forward(params, train_x)
        history.append(loss(probability, train_y))
        gradients = backward(params, train_x, train_y, probability, cache, bug)
        for key in params:
            params[key] -= learning_rate * gradients[key]

    test_probability, _ = forward(params, test_x)
    accuracy = float(((test_probability >= 0.5).astype(int) == test_y).mean())
    return Training(history=tuple(history), params=params, test_accuracy=accuracy)
