import numpy as np

import gradient_descent_playground_v3 as public_app
from gd_playground.config import Defaults
from gd_playground.data import generate_data
from gd_playground.workflow import (
    apply_animation_build,
    apply_run,
    apply_step,
    current_degree,
    current_metrics,
    history_summary,
    run_history,
)


def build_state():
    return Defaults().__dict__.copy()


def test_run_history_returns_stable_history_and_divergence_marker():
    state = build_state()
    data = generate_data(20, "Linear", 0.2, 3)

    history, divergence_iteration = run_history(
        data=data,
        params=np.array([0.0, 0.0, 0.0, 0.0]),
        degree=1,
        optimizer="Batch GD",
        learning_rate=0.01,
        iterations=20,
        batch_size=2,
        shuffle_each_epoch=False,
        seed=7,
        loss_function="MSE",
    )

    assert history
    assert divergence_iteration is None


def test_apply_run_updates_session_state_and_params():
    state = build_state()
    state["optimizer"] = "Batch GD"
    state["learning_rate"] = 0.01
    state["iterations_run"] = 20
    data = generate_data(20, "Linear", 0.0, 7)

    history, divergence_iteration = apply_run(state, data, degree=1, shuffle_each_epoch=False)

    assert divergence_iteration is None
    assert state["history"] == history
    assert state["animation_history"] is None
    assert state["w0"] == history[-1]["w0"]
    assert state["w1"] == history[-1]["w1"]


def test_apply_step_appends_one_new_iteration():
    state = build_state()
    data = generate_data(20, "Linear", 0.0, 7)

    apply_step(state, data, degree=1, shuffle_each_epoch=False)
    first_length = len(state["history"])
    apply_step(state, data, degree=1, shuffle_each_epoch=False)

    assert first_length == 1
    assert len(state["history"]) == 2
    assert [row["iteration"] for row in state["history"]] == [1, 2]


def test_apply_animation_build_populates_animation_history():
    state = build_state()
    state["iterations_animation"] = 25
    data = generate_data(20, "Quadratic", 0.3, 5)

    history, divergence_iteration = apply_animation_build(state, data, degree=2, shuffle_each_epoch=True)

    assert state["animation_history"] == history
    assert state["history"] == history
    assert divergence_iteration == state["divergence_iteration"]


def test_history_summary_reports_best_values():
    history = [
        {"iteration": 0, "optimization_loss": 10.0, "true_optimization_loss": 9.0, "mse": 10.0, "true_mse": 9.0},
        {"iteration": 1, "optimization_loss": 5.0, "true_optimization_loss": 4.0, "mse": 5.0, "true_mse": 4.0},
    ]

    summary = history_summary(history, tolerance=1e-6)

    assert summary["max_iteration"] == 1
    assert summary["best_optimization_loss"] == 5.0
    assert summary["best_true_mse"] == 4.0


def test_current_metrics_and_degree_read_state_consistently():
    state = build_state()
    state["model_type"] = "Quadratic"
    data = generate_data(12, "Quadratic", 0.1, 5)

    metrics, transform = current_metrics(state, data, degree=2)

    assert current_degree(state) == 2
    assert metrics["mse"] >= 0.0
    assert len(transform) == 2


def test_public_entrypoint_reexports_core_functions():
    data = public_app.generate_data(5, "Linear", 0.0, 1)
    figure = public_app.loss_figure([], data, degree=1, tolerance=1e-6)

    assert public_app.Defaults().model_type == "Linear"
    assert figure.layout.title.text == "Loss vs Iterations"
