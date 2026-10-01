"""Площадка «Нейросеть на NumPy» — тема «Обучение и оптимизация», модуль 3 курса Б.1.2.2.

A two-layer network with one line of the backward pass made breakable, and a
numerical gradient check run on every weight.
"""

from nn_playground.app import main
from nn_playground.network import (
    BUGS,
    NO_BUG,
    RELU_BUG,
    SIGN_BUG,
    SIZE_BUG,
    Component,
    Training,
    backward,
    forward,
    gradient_check,
    initialise,
    load_moons,
    loss,
    train,
)
from nn_playground.plotting import boundary_figure, history_figure

__all__ = [
    "BUGS",
    "NO_BUG",
    "RELU_BUG",
    "SIGN_BUG",
    "SIZE_BUG",
    "Component",
    "Training",
    "backward",
    "boundary_figure",
    "forward",
    "gradient_check",
    "history_figure",
    "initialise",
    "load_moons",
    "loss",
    "main",
    "train",
]
