from __future__ import annotations

import numpy as np


def degree_for_model(model_type: str) -> int:
    return {"Linear": 1, "Quadratic": 2, "Cubic": 3}.get(model_type, 1)


def active_param_names(degree: int) -> list[str]:
    return [f"w{index}" for index in range(int(degree) + 1)]


def param_index(param_name: str) -> int:
    return int(param_name[1:])


def design_matrix(x, degree: int) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    columns = [np.ones_like(values)]
    for power in range(1, int(degree) + 1):
        columns.append(values**power)
    return np.vstack(columns).T


def predict_values(x, params, degree: int, transform: tuple[float, float] | None = None) -> np.ndarray:
    basis = np.asarray(x, dtype=float)
    if transform is not None:
        center, scale = transform
        basis = (basis - center) / scale
    return design_matrix(basis, degree) @ np.asarray(params, dtype=float)[: degree + 1]


def equation_text(params, degree: int, variable_name: str = "x") -> str:
    coeffs = np.asarray(params, dtype=float)
    if degree == 1:
        return f"y = {coeffs[1]:.3f}{variable_name} + {coeffs[0]:.3f}"
    if degree == 2:
        return (
            f"y = {coeffs[2]:.3f}{variable_name}^2 + "
            f"{coeffs[1]:.3f}{variable_name} + {coeffs[0]:.3f}"
        )
    return (
        f"y = {coeffs[3]:.3f}{variable_name}^3 + "
        f"{coeffs[2]:.3f}{variable_name}^2 + "
        f"{coeffs[1]:.3f}{variable_name} + {coeffs[0]:.3f}"
    )
