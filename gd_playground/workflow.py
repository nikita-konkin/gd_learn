from __future__ import annotations

from gd_playground.data import feature_transform_from_data
from gd_playground.model import degree_for_model
from gd_playground.state import params_vector, write_params
from gd_playground.training import (
    convergence_iteration,
    full_dataset_metrics,
    optimizer_history,
    params_from_row,
    prepare_history_for_display,
)


def history_summary(history, tolerance: float) -> dict | None:
    if not history:
        return None

    return {
        "convergence_iteration": convergence_iteration(history, tolerance, "optimization_loss"),
        "max_iteration": history[-1]["iteration"],
        "best_optimization_loss": min(row["optimization_loss"] for row in history),
        "best_true_optimization_loss": min(row["true_optimization_loss"] for row in history),
        "best_observed_mse": min(row["mse"] for row in history),
        "best_true_mse": min(row["true_mse"] for row in history),
        "last_optimization_loss": history[-1]["optimization_loss"],
    }


def run_history(
    data,
    params,
    degree: int,
    optimizer: str,
    learning_rate: float,
    iterations: int,
    batch_size: int,
    shuffle_each_epoch: bool,
    seed: int,
    loss_function: str,
):
    raw_history = optimizer_history(
        data=data,
        params=params,
        degree=degree,
        optimizer=optimizer,
        learning_rate=learning_rate,
        iterations=iterations,
        batch_size=batch_size,
        shuffle_each_epoch=shuffle_each_epoch,
        seed=seed,
        loss_function=loss_function,
    )
    return prepare_history_for_display(raw_history, data, degree, loss_function)


def apply_run(session_state, data, degree: int, shuffle_each_epoch: bool):
    history, divergence_iteration = run_history(
        data=data,
        params=params_vector(session_state),
        degree=degree,
        optimizer=session_state["optimizer"],
        learning_rate=session_state["learning_rate"],
        iterations=session_state["iterations_run"],
        batch_size=session_state["batch_size"],
        shuffle_each_epoch=shuffle_each_epoch,
        seed=int(session_state["seed"]) + 123,
        loss_function=session_state["loss_function"],
    )
    session_state["history"] = history
    session_state["animation_history"] = None
    session_state["divergence_iteration"] = divergence_iteration
    if history:
        write_params(params_from_row(history[-1]), session_state)
    return history, divergence_iteration


def apply_step(session_state, data, degree: int, shuffle_each_epoch: bool):
    history, divergence_iteration = run_history(
        data=data,
        params=params_vector(session_state),
        degree=degree,
        optimizer=session_state["optimizer"],
        learning_rate=session_state["learning_rate"],
        iterations=1,
        batch_size=session_state["batch_size"],
        shuffle_each_epoch=shuffle_each_epoch,
        seed=int(session_state["seed"]) + 456,
        loss_function=session_state["loss_function"],
    )

    previous_history = session_state.get("history") or []
    next_iteration = previous_history[-1]["iteration"] + 1 if previous_history else 1
    if len(history) > 1:
        appended_row = history[1].copy()
        appended_row["iteration"] = next_iteration
        session_state["history"] = previous_history + [appended_row]
        session_state["divergence_iteration"] = None
        write_params(params_from_row(session_state["history"][-1]), session_state)
    else:
        session_state["history"] = previous_history
        session_state["divergence_iteration"] = (
            next_iteration if divergence_iteration is not None else None
        )

    session_state["animation_history"] = None
    return session_state["history"], session_state["divergence_iteration"]


def apply_animation_build(session_state, data, degree: int, shuffle_each_epoch: bool):
    history, divergence_iteration = run_history(
        data=data,
        params=params_vector(session_state),
        degree=degree,
        optimizer=session_state["optimizer"],
        learning_rate=session_state["learning_rate"],
        iterations=session_state["iterations_animation"],
        batch_size=session_state["batch_size"],
        shuffle_each_epoch=shuffle_each_epoch,
        seed=int(session_state["seed"]) + 789,
        loss_function=session_state["loss_function"],
    )
    session_state["history"] = history
    session_state["animation_history"] = history
    session_state["divergence_iteration"] = divergence_iteration
    return history, divergence_iteration


def current_metrics(session_state, data, degree: int) -> tuple[dict, tuple[float, float]]:
    transform = feature_transform_from_data(data)
    metrics = full_dataset_metrics(
        data,
        params_vector(session_state),
        degree,
        session_state["loss_function"],
        transform,
    )
    return metrics, transform


def current_degree(session_state) -> int:
    return degree_for_model(session_state["model_type"])
