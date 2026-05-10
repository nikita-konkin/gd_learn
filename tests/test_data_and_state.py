import numpy as np

from gd_playground.config import Defaults
from gd_playground.data import (
    clamp_batch_size,
    divergence_limits,
    feature_transform,
    feature_transform_from_data,
    generate_data,
    normalized_feature_text,
    should_regenerate_data,
)
from gd_playground.state import (
    clear_training_outputs,
    ensure_distinct_surface_params,
    init_state,
    params_vector,
    refresh_dataset,
    reset_all_state,
    sync_training_config,
    write_params,
)


def test_generate_data_is_seeded_and_shape_stable():
    first = generate_data(12, "Quadratic", 0.5, 7)
    second = generate_data(12, "Quadratic", 0.5, 7)

    assert list(first.columns) == ["x", "y", "y_true"]
    assert first.equals(second)


def test_should_regenerate_data_detects_seed_change():
    data = generate_data(12, "Linear", 0.5, 7)

    assert should_regenerate_data(
        data=data,
        n_points=12,
        distribution="Linear",
        noise=0.5,
        seed=8,
        previous_distribution="Linear",
        previous_noise=0.5,
        previous_seed=7,
    )


def test_should_regenerate_data_stays_false_for_same_controls():
    data = generate_data(12, "Quadratic", 0.5, 7)

    assert not should_regenerate_data(
        data=data,
        n_points=12,
        distribution="Quadratic",
        noise=0.5,
        seed=7,
        previous_distribution="Quadratic",
        previous_noise=0.5,
        previous_seed=7,
    )


def test_clamp_batch_size_stays_above_sgd_mode():
    assert clamp_batch_size(1, 30) == 2
    assert clamp_batch_size(99, 5) == 5


def test_feature_transform_handles_constant_input():
    center, scale = feature_transform([4.0, 4.0, 4.0])

    assert center == 4.0
    assert scale == 1.0
    assert normalized_feature_text((center, scale)) == "z = (x - 4.000) / 1.000"


def test_divergence_limits_are_positive():
    data = generate_data(10, "Cubic", 0.1, 4)
    limits = divergence_limits(data)

    assert limits["prediction_limit"] > 0.0
    assert limits["mse_limit"] > limits["prediction_limit"]
    assert limits["grad_norm_limit"] > 0.0


def test_init_reset_and_param_helpers_work_on_plain_dict():
    state = {}
    init_state(state)
    write_params(np.array([1.0, 2.0, 3.0, 4.0]), state)

    assert params_vector(state).tolist() == [1.0, 2.0, 3.0, 4.0]

    clear_training_outputs(state)
    assert state["history"] is None
    assert state["animation_history"] is None
    assert state["divergence_iteration"] is None

    reset_all_state(state)
    defaults = Defaults()
    assert state["model_type"] == defaults.model_type
    assert state["w0"] == defaults.w0


def test_refresh_dataset_updates_seed_tracks_controls_and_clears_history():
    state = Defaults().__dict__.copy()
    state["history"] = [{"iteration": 0}]
    state["animation_history"] = [{"iteration": 0}]
    state["divergence_iteration"] = 3

    data = refresh_dataset(state, force_regenerate=True, increment_seed=True)

    assert state["seed"] == Defaults().seed + 1
    assert state["distribution_name"] == state["distribution"]
    assert state["noise_value"] == state["noise"]
    assert state["seed_value"] == state["seed"]
    assert state["history"] is None
    assert state["animation_history"] is None
    assert state["divergence_iteration"] is None
    assert data is state["data"]


def test_sync_training_config_reports_changes_and_clears_training_outputs():
    state = Defaults().__dict__.copy()
    init_state(state)
    state["history"] = [{"iteration": 0}]
    state["model_type_name"] = "Linear"
    state["optimizer_name"] = "Batch GD"
    state["loss_function_name"] = "MSE"
    state["model_type"] = "Quadratic"

    changed = sync_training_config(state)

    assert changed
    assert state["history"] is None
    assert state["model_type_name"] == "Quadratic"


def test_ensure_distinct_surface_params_repairs_invalid_state():
    state = {"surface_param_x": "bad", "surface_param_y": "w0"}

    x_name, y_name = ensure_distinct_surface_params(["w0", "w1", "w2"], state)

    assert x_name == "w0"
    assert y_name == "w1"
    assert state["surface_param_y"] == "w1"


def test_feature_transform_from_data_matches_feature_transform():
    data = generate_data(8, "Sine", 0.0, 5)

    assert feature_transform_from_data(data) == feature_transform(data["x"])
