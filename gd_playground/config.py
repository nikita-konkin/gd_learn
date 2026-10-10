from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DATA_DISTRIBUTIONS = ("Linear", "Quadratic", "Cubic", "Sine")
MODEL_TYPES = ("Linear", "Quadratic", "Cubic")
OPTIMIZERS = ("Batch GD", "SGD", "Mini-batch SGD")
LOSS_FUNCTIONS = ("MSE", "MAE")

# The values above are compared in code and tests, so they stay English.
# These are the labels the student sees.
DISTRIBUTION_NAMES = {
    "Linear": "прямая",
    "Quadratic": "парабола",
    "Cubic": "кубическая кривая",
    "Sine": "синусоида",
}
MODEL_NAMES = {
    "Linear": "прямая (1-я степень)",
    "Quadratic": "парабола (2-я степень)",
    "Cubic": "кубическая (3-я степень)",
}
OPTIMIZER_NAMES = {
    "Batch GD": "все точки сразу (Batch GD)",
    "SGD": "одну случайную точку (SGD)",
    "Mini-batch SGD": "пакет точек (мини-пакетный SGD)",
}
LEARNING_RATE_OPTIONS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
CONVERGENCE_OPTIONS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)


@dataclass
class Defaults:
    model_type: str = "Linear"
    distribution: str = "Linear"
    optimizer: str = "Batch GD"
    loss_function: str = "MSE"
    n_points: int = 30
    noise: float = 1.2
    seed: int = 7
    learning_rate: float = 0.01
    batch_size: int = 8
    iterations_run: int = 100
    iterations_animation: int = 120
    convergence_tolerance: float = 1e-6
    w0: float = 17.0
    w1: float = 0.8
    w2: float = 0.0
    w3: float = 0.0
    history: list[dict] | None = None
    animation_history: list[dict] | None = None
    divergence_iteration: int | None = None
    surface_param_x: str = "w0"
    surface_param_y: str = "w1"
    surface_resolution: int = 35
    data: pd.DataFrame | None = None


def default_state() -> dict:
    return Defaults().__dict__.copy()
