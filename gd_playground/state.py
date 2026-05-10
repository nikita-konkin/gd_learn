from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np

from gd_playground.config import default_state
from gd_playground.data import generate_data, should_regenerate_data


def _resolve_session_state(
    session_state: MutableMapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    if session_state is not None:
        return session_state
    import streamlit as st

    return st.session_state


def init_state(session_state: MutableMapping[str, Any] | None = None) -> MutableMapping[str, Any]:
    state = _resolve_session_state(session_state)
    for key, value in default_state().items():
        if key not in state:
            state[key] = value
    return state


def clear_training_outputs(session_state: MutableMapping[str, Any] | None = None) -> None:
    state = _resolve_session_state(session_state)
    state["history"] = None
    state["animation_history"] = None
    state["divergence_iteration"] = None


def reset_all_state(session_state: MutableMapping[str, Any] | None = None) -> MutableMapping[str, Any]:
    state = _resolve_session_state(session_state)
    state.clear()
    state.update(default_state())
    return state


def params_vector(session_state: MutableMapping[str, Any] | None = None) -> np.ndarray:
    state = _resolve_session_state(session_state)
    return np.array(
        [
            float(state.get("w0", 0.0)),
            float(state.get("w1", 0.0)),
            float(state.get("w2", 0.0)),
            float(state.get("w3", 0.0)),
        ],
        dtype=float,
    )


def write_params(params, session_state: MutableMapping[str, Any] | None = None) -> None:
    state = _resolve_session_state(session_state)
    params = np.asarray(params, dtype=float)
    state["w0"] = float(params[0])
    state["w1"] = float(params[1])
    state["w2"] = float(params[2])
    state["w3"] = float(params[3])


def ensure_distinct_surface_params(
    param_names: list[str],
    session_state: MutableMapping[str, Any] | None = None,
) -> tuple[str, str]:
    state = _resolve_session_state(session_state)
    x_name = state.get("surface_param_x", param_names[0])
    y_name = state.get("surface_param_y", param_names[1] if len(param_names) > 1 else param_names[0])

    if x_name not in param_names:
        x_name = param_names[0]
    if y_name not in param_names or y_name == x_name:
        alternatives = [name for name in param_names if name != x_name]
        y_name = alternatives[0] if alternatives else x_name

    state["surface_param_x"] = x_name
    state["surface_param_y"] = y_name
    return x_name, y_name


def remember_data_controls(session_state: MutableMapping[str, Any] | None = None) -> None:
    state = _resolve_session_state(session_state)
    state["distribution_name"] = state["distribution"]
    state["noise_value"] = state["noise"]
    state["seed_value"] = state["seed"]


def refresh_dataset(
    session_state: MutableMapping[str, Any] | None = None,
    force_regenerate: bool = False,
    increment_seed: bool = False,
):
    state = _resolve_session_state(session_state)
    if increment_seed:
        state["seed"] = int(state["seed"]) + 1

    needs_regeneration = force_regenerate or should_regenerate_data(
        data=state.get("data"),
        n_points=state["n_points"],
        distribution=state["distribution"],
        noise=state["noise"],
        seed=state["seed"],
        previous_distribution=state.get("distribution_name"),
        previous_noise=state.get("noise_value"),
        previous_seed=state.get("seed_value"),
    )

    if needs_regeneration:
        state["data"] = generate_data(
            state["n_points"],
            state["distribution"],
            state["noise"],
            state["seed"],
        )
        clear_training_outputs(state)

    remember_data_controls(state)
    return state["data"]


def sync_training_config(session_state: MutableMapping[str, Any] | None = None) -> bool:
    state = _resolve_session_state(session_state)
    changed = (
        state.get("model_type_name") != state["model_type"]
        or state.get("optimizer_name") != state["optimizer"]
        or state.get("loss_function_name") != state["loss_function"]
    )
    if changed:
        clear_training_outputs(state)

    state["model_type_name"] = state["model_type"]
    state["optimizer_name"] = state["optimizer"]
    state["loss_function_name"] = state["loss_function"]
    return changed
