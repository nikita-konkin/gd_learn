import numpy as np
import pytest

from gd_playground.data import feature_transform_from_data, generate_data
from gd_playground.model import active_param_names, degree_for_model, design_matrix, equation_text, param_index, predict_values
from gd_playground.training import (
    convergence_iteration,
    full_dataset_metrics,
    gradient,
    loss_axis_title,
    loss_function_options,
    mae_loss,
    mse_loss,
    optimization_loss,
    optimization_loss_label,
    optimizer_history,
    params_from_row,
    prepare_history_for_display,
    recompute_history_metrics,
)


def test_degree_and_parameter_helpers_match_supported_models():
    assert degree_for_model("Linear") == 1
    assert degree_for_model("Quadratic") == 2
    assert degree_for_model("Cubic") == 3
    assert active_param_names(2) == ["w0", "w1", "w2"]
    assert param_index("w3") == 3


def test_design_matrix_and_prediction_are_polynomial():
    design = design_matrix([0.0, 2.0], 2)
    params = np.array([1.0, 3.0, 5.0, 0.0])
    prediction = predict_values([0.0, 2.0], params, 2)

    assert design.tolist() == [[1.0, 0.0, 0.0], [1.0, 2.0, 4.0]]
    assert prediction.tolist() == [1.0, 27.0]


def test_equation_text_uses_ascii_power_notation():
    params = np.array([1.0, 2.0, 3.0, 4.0])

    assert equation_text(params, 1) == "y = 2.000x + 1.000"
    assert equation_text(params, 2) == "y = 3.000x^2 + 2.000x + 1.000"
    assert equation_text(params, 3) == "y = 4.000x^3 + 3.000x^2 + 2.000x + 1.000"


def test_loss_label_helpers_reflect_selected_optimizer_loss():
    assert loss_function_options() == ["MSE", "MAE"]
    assert optimization_loss_label("MAE") == "Observed MAE"
    assert optimization_loss_label("MSE", "True-function") == "True-function MSE"
    assert loss_axis_title("MAE") == "Loss (MAE)"


def test_loss_functions_and_gradient_match_simple_linear_case():
    x_values = np.array([0.0, 1.0])
    y_values = np.array([0.0, 1.0])
    params = np.array([0.0, 0.0, 0.0, 0.0])

    assert mse_loss(x_values, y_values, params, 1) == pytest.approx(0.5)
    assert mae_loss(x_values, y_values, params, 1) == pytest.approx(0.5)
    assert optimization_loss(x_values, y_values, params, 1, "MAE") == pytest.approx(0.5)
    assert gradient(x_values, y_values, params, 1, "MSE").tolist() == pytest.approx([-1.0, -1.0, 0.0, 0.0])
    assert gradient(x_values, y_values, params, 1, "MAE").tolist() == pytest.approx([-0.5, -0.5, 0.0, 0.0])


def test_full_dataset_metrics_include_true_error_and_gradient_norm():
    data = generate_data(6, "Linear", 0.0, 3)
    center, scale = feature_transform_from_data(data)
    params = np.array([12.0 + 2.2 * center, 2.2 * scale, 0.0, 0.0])

    metrics = full_dataset_metrics(data, params, degree=1)

    assert metrics["true_mse"] == pytest.approx(0.0)
    assert metrics["true_rmse"] == pytest.approx(0.0)
    assert metrics["grad_norm"] >= 0.0
    assert metrics["param_norm"] > 0.0


def test_recompute_history_metrics_overrides_stale_values():
    data = generate_data(6, "Linear", 0.0, 3)
    history = [
        {"iteration": 0, "w0": 1.0, "w1": 0.5, "w2": 0.0, "w3": 0.0, "mse": 9999.0},
        {"iteration": 1, "w0": 12.0, "w1": 2.2, "w2": 9.0, "w3": 7.0, "mse": 8888.0},
    ]

    fixed = recompute_history_metrics(history, data, degree=1)

    assert fixed[0]["mse"] != 9999.0
    assert fixed[1]["mse"] != 8888.0
    assert fixed[0]["true_mse"] >= 0.0


def test_optimizer_history_can_optimize_mae():
    data = generate_data(20, "Linear", 0.0, 7)
    history = optimizer_history(
        data=data,
        params=np.array([0.0, 0.0, 0.0, 0.0]),
        degree=1,
        optimizer="Batch GD",
        learning_rate=0.01,
        iterations=40,
        batch_size=1,
        shuffle_each_epoch=False,
        seed=7,
        loss_function="MAE",
    )

    assert history[-1]["optimization_loss"] <= history[0]["optimization_loss"]


def test_optimizer_history_stays_finite_for_scaled_cubic_minibatch():
    data = generate_data(30, "Linear", 1.2, 8)
    history = optimizer_history(
        data=data,
        params=np.array([17.0, 0.8, 0.0, 0.0]),
        degree=3,
        optimizer="Mini-batch SGD",
        learning_rate=0.01,
        iterations=600,
        batch_size=2,
        shuffle_each_epoch=True,
        seed=797,
    )

    assert np.isfinite([row["mse"] for row in history]).all()
    assert np.isfinite([row["true_mse"] for row in history]).all()


def test_prepare_history_for_display_trims_divergent_rows():
    data = generate_data(6, "Linear", 0.0, 3)
    history = [
        {"iteration": 0, "w0": 12.0, "w1": 2.2, "w2": 0.0, "w3": 0.0},
        {"iteration": 1, "w0": 0.0, "w1": 0.0, "w2": 0.0, "w3": 100.0},
    ]

    stable, divergence_iteration = prepare_history_for_display(history, data, degree=3)

    assert [row["iteration"] for row in stable] == [0]
    assert divergence_iteration == 1


def test_convergence_iteration_uses_selected_metric():
    history = [
        {"iteration": 0, "optimization_loss": 10.0},
        {"iteration": 1, "optimization_loss": 5.0},
        {"iteration": 2, "optimization_loss": 4.9999999},
    ]

    assert convergence_iteration(history, 1e-5) == 2
    assert convergence_iteration(history[:1], 1e-5) is None


def test_params_from_row_builds_dense_parameter_vector():
    row = {"w0": 1, "w1": 2, "w2": 3, "w3": 4}
    assert params_from_row(row).tolist() == [1.0, 2.0, 3.0, 4.0]
