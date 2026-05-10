from __future__ import annotations

import math

import numpy as np

from gd_playground.config import LOSS_FUNCTIONS
from gd_playground.data import clamp_batch_size, divergence_limits, feature_transform_from_data
from gd_playground.model import predict_values


def loss_function_options() -> list[str]:
    return list(LOSS_FUNCTIONS)


def optimization_loss_label(loss_function: str, prefix: str = "Observed") -> str:
    return f"{prefix} {loss_function}"


def loss_axis_title(loss_function: str) -> str:
    return f"Loss ({loss_function})"


def params_from_row(row: dict) -> np.ndarray:
    return np.array([row["w0"], row["w1"], row["w2"], row["w3"]], dtype=float)


def mse_loss(x, y, params, degree: int, transform: tuple[float, float] | None = None) -> float:
    residuals = predict_values(x, params, degree, transform) - np.asarray(y, dtype=float)
    return float(np.mean(residuals**2))


def mae_loss(x, y, params, degree: int, transform: tuple[float, float] | None = None) -> float:
    residuals = predict_values(x, params, degree, transform) - np.asarray(y, dtype=float)
    return float(np.mean(np.abs(residuals)))


def optimization_loss(
    x,
    y,
    params,
    degree: int,
    loss_function: str = "MSE",
    transform: tuple[float, float] | None = None,
) -> float:
    if loss_function == "MAE":
        return mae_loss(x, y, params, degree, transform)
    return mse_loss(x, y, params, degree, transform)


def gradient(
    x,
    y,
    params,
    degree: int,
    loss_function: str = "MSE",
    transform: tuple[float, float] | None = None,
) -> np.ndarray:
    basis = np.asarray(x, dtype=float)
    if transform is not None:
        center, scale = transform
        basis = (basis - center) / scale

    params = np.asarray(params, dtype=float)
    y = np.asarray(y, dtype=float)
    columns = [np.ones_like(basis)]
    for power in range(1, int(degree) + 1):
        columns.append(basis**power)
    design = np.vstack(columns).T

    errors = design @ params[: degree + 1] - y
    if loss_function == "MAE":
        active_gradient = (1.0 / len(basis)) * (design.T @ np.sign(errors))
    else:
        active_gradient = (2.0 / len(basis)) * (design.T @ errors)

    full_gradient = np.zeros_like(params)
    full_gradient[: degree + 1] = active_gradient
    return full_gradient


def full_dataset_metrics(
    data,
    params,
    degree: int,
    loss_function: str = "MSE",
    transform: tuple[float, float] | None = None,
) -> dict[str, float]:
    if transform is None:
        transform = feature_transform_from_data(data)

    x_values = data["x"].to_numpy(dtype=float)
    observed = data["y"].to_numpy(dtype=float)
    true_values = data["y_true"].to_numpy(dtype=float)

    observed_optimization_loss = optimization_loss(
        x_values, observed, params, degree, loss_function, transform
    )
    true_optimization_loss = optimization_loss(
        x_values, true_values, params, degree, loss_function, transform
    )
    observed_mse = mse_loss(x_values, observed, params, degree, transform)
    true_mse = mse_loss(x_values, true_values, params, degree, transform)
    full_gradient = gradient(x_values, observed, params, degree, loss_function, transform)

    return {
        "optimization_loss": observed_optimization_loss,
        "true_optimization_loss": true_optimization_loss,
        "mse": observed_mse,
        "rmse": math.sqrt(observed_mse),
        "mae": mae_loss(x_values, observed, params, degree, transform),
        "true_mse": true_mse,
        "true_rmse": math.sqrt(true_mse),
        "grad_norm": float(np.linalg.norm(full_gradient[: degree + 1])),
        "param_norm": float(np.linalg.norm(np.asarray(params, dtype=float)[: degree + 1])),
    }


def _history_row(iteration: int, params, metrics: dict[str, float]) -> dict[str, float]:
    return {
        "iteration": iteration,
        "w0": float(params[0]),
        "w1": float(params[1]),
        "w2": float(params[2]),
        "w3": float(params[3]),
        "optimization_loss": metrics["optimization_loss"],
        "true_optimization_loss": metrics["true_optimization_loss"],
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "true_mse": metrics["true_mse"],
        "true_rmse": metrics["true_rmse"],
        "grad_norm": metrics["grad_norm"],
        "param_norm": metrics["param_norm"],
    }


def _diverged(params, metrics: dict[str, float], degree: int, limits: dict[str, float]) -> bool:
    active_params = np.asarray(params, dtype=float)[: degree + 1]
    metric_values = np.array(
        [
            metrics["optimization_loss"],
            metrics["true_optimization_loss"],
            metrics["mse"],
            metrics["true_mse"],
            metrics["grad_norm"],
        ],
        dtype=float,
    )
    return (
        not np.all(np.isfinite(active_params))
        or not np.all(np.isfinite(metric_values))
        or metrics["mse"] > limits["mse_limit"]
        or metrics["grad_norm"] > limits["grad_norm_limit"]
    )


def _batch_indices(
    optimizer: str,
    order: np.ndarray,
    cursor: int,
    batch_size: int,
    shuffle_each_epoch: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    sample_count = len(order)
    if optimizer == "Batch GD":
        return order, cursor

    if cursor == 0 and shuffle_each_epoch:
        rng.shuffle(order)

    if optimizer == "SGD":
        next_index = order[cursor]
        return np.array([next_index], dtype=int), (cursor + 1) % sample_count

    end = min(cursor + batch_size, sample_count)
    indices = order[cursor:end]
    if len(indices) == 0:
        return order[:batch_size], 0
    next_cursor = 0 if end >= sample_count else end
    return indices, next_cursor


def optimizer_history(
    data,
    params,
    degree: int,
    optimizer: str,
    learning_rate: float,
    iterations: int,
    batch_size: int,
    shuffle_each_epoch: bool,
    seed: int,
    loss_function: str = "MSE",
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    x_all = data["x"].to_numpy(dtype=float)
    y_all = data["y"].to_numpy(dtype=float)
    n_points = len(x_all)
    batch_size = clamp_batch_size(batch_size, n_points)
    transform = feature_transform_from_data(data)
    limits = divergence_limits(data)

    params = np.asarray(params, dtype=float).copy()
    order = np.arange(n_points, dtype=int)
    cursor = 0
    history: list[dict[str, float]] = []

    for iteration in range(int(iterations) + 1):
        metrics = full_dataset_metrics(data, params, degree, loss_function, transform)
        history.append(_history_row(iteration, params, metrics))

        if _diverged(params, metrics, degree, limits) or iteration == int(iterations):
            break

        batch_indices, cursor = _batch_indices(
            optimizer, order, cursor, batch_size, shuffle_each_epoch, rng
        )
        step_gradient = gradient(
            x_all[batch_indices],
            y_all[batch_indices],
            params,
            degree,
            loss_function,
            transform,
        )
        params -= float(learning_rate) * step_gradient
        if degree < 3:
            params[degree + 1 :] = 0.0

    return history


def recompute_history_metrics(history, data, degree: int, loss_function: str = "MSE") -> list[dict]:
    transform = feature_transform_from_data(data)
    recomputed_history = []
    for row in history or []:
        params = params_from_row(row)
        metrics = full_dataset_metrics(data, params, degree, loss_function, transform)
        recomputed_history.append({**row, **metrics})
    return recomputed_history


def prepare_history_for_display(history, data, degree: int, loss_function: str = "MSE"):
    fixed_history = recompute_history_metrics(history, data, degree, loss_function)
    if not fixed_history:
        return [], None

    transform = feature_transform_from_data(data)
    limits = divergence_limits(data)
    x_probe = np.linspace(-4.3, 4.3, 25)
    stable_history = []

    for row in fixed_history:
        params = params_from_row(row)
        metric_values = np.array(
            [
                row["optimization_loss"],
                row["true_optimization_loss"],
                row["mse"],
                row["rmse"],
                row["mae"],
                row["true_mse"],
                row["true_rmse"],
                row["grad_norm"],
            ],
            dtype=float,
        )

        with np.errstate(over="ignore", invalid="ignore"):
            y_probe = predict_values(x_probe, params, degree, transform)

        if (
            not np.all(np.isfinite(params))
            or not np.all(np.isfinite(metric_values))
            or row["mse"] > limits["mse_limit"]
            or row["grad_norm"] > limits["grad_norm_limit"]
            or not np.all(np.isfinite(y_probe))
            or np.max(np.abs(y_probe)) > limits["prediction_limit"]
        ):
            return stable_history, row["iteration"]

        stable_history.append(row)

    return stable_history, None


def convergence_iteration(history, tolerance: float, metric_key: str = "optimization_loss") -> int | None:
    if not history or len(history) < 2:
        return None

    for index in range(1, len(history)):
        delta = abs(history[index - 1][metric_key] - history[index][metric_key])
        if delta < tolerance:
            return history[index]["iteration"]
    return None
