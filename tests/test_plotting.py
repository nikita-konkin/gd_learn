import numpy as np

from gd_playground.data import feature_transform_from_data, generate_data
from gd_playground.plotting import (
    animated_dashboard_figure,
    animation_slider_label,
    loss_figure,
    loss_surface_figure,
    metrics_text,
    model_figure,
    parameter_axis_values,
)
from gd_playground.training import optimizer_history


def test_animation_slider_labels_are_sparse():
    labels = [animation_slider_label(i, 260) for i in range(261)]

    assert labels[0] == "0"
    assert labels[-1] == "260"
    assert labels[1] == ""
    assert labels[22] != ""


def test_metrics_text_includes_scaled_feature_summary():
    data = generate_data(6, "Linear", 0.0, 3)
    transform = feature_transform_from_data(data)
    row = {
        "iteration": 0,
        "w0": 1.0,
        "w1": 0.5,
        "w2": 0.0,
        "w3": 0.0,
        "optimization_loss": 1.0,
        "true_optimization_loss": 0.5,
        "mse": 1.0,
        "true_mse": 0.5,
        "rmse": 1.0,
        "mae": 0.75,
        "grad_norm": 0.25,
    }

    summary = metrics_text(row, 1, transform, "MSE")

    assert "Iteration: 0" in summary
    assert "Scaled feature:" in summary
    assert "Model:" in summary


def test_model_figure_draws_one_residual_segment_per_point():
    data = generate_data(7, "Linear", 0.1, 5)
    figure = model_figure(data, np.array([10.0, 2.0, 0.0, 0.0]), degree=1)

    assert len(figure.data) == len(data) + 3


def test_parameter_axis_values_pad_small_ranges():
    history = [{"w0": 1.0}, {"w0": 1.0}]
    axis = parameter_axis_values(history, np.array([1.0, 0.0, 0.0, 0.0]), "w0", 5)

    assert len(axis) == 5
    assert axis[0] < 1.0 < axis[-1]


def test_loss_figure_uses_selected_loss_and_reports_divergence():
    data = generate_data(6, "Linear", 0.0, 3)
    history = [
        {"iteration": 0, "w0": 12.0, "w1": 2.2, "w2": 0.0, "w3": 0.0},
        {"iteration": 1, "w0": 0.0, "w1": 0.0, "w2": 0.0, "w3": 100.0},
    ]

    figure = loss_figure(history, data, degree=3, tolerance=1e-6, loss_function="MAE")

    assert figure.layout.yaxis.title.text == "Loss (MAE)"
    assert figure.data[0].name == "Observed MAE"
    assert "Diverged at iteration 1" in figure.layout.title.text


def test_animation_figure_returns_empty_state():
    data = generate_data(5, "Sine", 0.1, 11)
    figure = animated_dashboard_figure(data, [], degree=1, tolerance=1e-6)

    assert figure.layout.title.text == "Gradient Descent Animation"
    assert figure.layout.annotations[0].text == "Build animation to show model-fit and loss playback."


def test_animation_controls_are_horizontal_and_below_plot():
    data = generate_data(8, "Quadratic", 0.1, 5)
    history = optimizer_history(
        data=data,
        params=np.array([16.0, 1.0, -0.3, 0.0]),
        degree=2,
        optimizer="Mini-batch SGD",
        learning_rate=0.001,
        iterations=20,
        batch_size=2,
        shuffle_each_epoch=True,
        seed=42,
    )

    figure = animated_dashboard_figure(data, history, degree=2, tolerance=1e-6)

    assert figure.layout.updatemenus[0].direction == "right"
    assert figure.layout.updatemenus[0].y < 0
    assert figure.layout.sliders[0].y < 0
    assert len(figure.layout.annotations) == 2


def test_loss_surface_figure_shows_surface_and_descent_path():
    data = generate_data(12, "Linear", 0.1, 5)
    history = optimizer_history(
        data=data,
        params=np.array([0.0, 0.0, 0.0, 0.0]),
        degree=1,
        optimizer="Batch GD",
        learning_rate=0.01,
        iterations=10,
        batch_size=1,
        shuffle_each_epoch=False,
        seed=5,
        loss_function="MAE",
    )

    figure = loss_surface_figure(
        data=data,
        history=history,
        current_params=np.array([history[-1]["w0"], history[-1]["w1"], history[-1]["w2"], history[-1]["w3"]]),
        degree=1,
        loss_function="MAE",
        x_param="w0",
        y_param="w1",
        resolution=15,
    )

    assert figure.data[0].type == "surface"
    assert figure.data[1].name == "Gradient descent path"
    assert figure.layout.scene.zaxis.title.text == "MAE"
    assert list(figure.data[1].z) == [row["optimization_loss"] for row in history]
