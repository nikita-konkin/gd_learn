from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gd_playground.data import feature_transform_from_data, normalized_feature_text
from gd_playground.model import equation_text, param_index, predict_values
from gd_playground.training import (
    convergence_iteration,
    loss_axis_title,
    optimization_loss,
    optimization_loss_label,
    params_from_row,
    prepare_history_for_display,
)


def animation_slider_label(iteration: int, max_iteration: int, max_labels: int = 12) -> str:
    if max_iteration <= 0:
        return str(iteration)

    interval = max(1, int(math.ceil(max_iteration / max_labels)))
    if iteration in (0, max_iteration) or iteration % interval == 0:
        return str(iteration)
    return ""


def metrics_text(row: dict, degree: int, transform: tuple[float, float], loss_function: str = "MSE") -> str:
    params = params_from_row(row)
    return (
        f"Iteration: {row['iteration']}\n"
        f"{optimization_loss_label(loss_function)}: {row['optimization_loss']:.6f}\n"
        f"{optimization_loss_label(loss_function, 'True-function')}: "
        f"{row['true_optimization_loss']:.6f}\n"
        f"Observed MSE: {row['mse']:.6f}\n"
        f"True MSE: {row['true_mse']:.6f}\n"
        f"RMSE: {row['rmse']:.6f}\n"
        f"MAE: {row['mae']:.6f}\n"
        f"|grad(loss)|: {row['grad_norm']:.6f}\n"
        f"Model: {equation_text(params, degree, 'z')}\n"
        f"Scaled feature: {normalized_feature_text(transform)}"
    )


def model_figure(data, params, degree: int, title_suffix: str = "") -> go.Figure:
    transform = feature_transform_from_data(data)
    x_curve = np.linspace(-4.3, 4.3, 200)
    y_model = predict_values(x_curve, params, degree, transform)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=data["x"],
            y=data["y_true"],
            mode="lines",
            name="Underlying distribution",
            line=dict(width=3, dash="dot"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=data["x"],
            y=data["y"],
            mode="markers",
            name="Data points",
            marker=dict(size=10),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x_curve,
            y=y_model,
            mode="lines",
            name=equation_text(params, degree, "z"),
            line=dict(width=4),
        )
    )

    point_predictions = predict_values(data["x"].to_numpy(dtype=float), params, degree, transform)
    for x_value, observed, predicted in zip(data["x"], data["y"], point_predictions, strict=True):
        figure.add_trace(
            go.Scatter(
                x=[x_value, x_value],
                y=[observed, predicted],
                mode="lines",
                line=dict(width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    figure.update_layout(
        title=f"Model Fit {title_suffix}".strip(),
        height=540,
        margin=dict(l=30, r=30, t=70, b=30),
        xaxis_title="x",
        yaxis_title="y",
        xaxis=dict(range=[-4.5, 4.5], gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
        hovermode="closest",
    )
    return figure


def parameter_axis_values(history, reference_params, param_name: str, resolution: int) -> np.ndarray:
    index = param_index(param_name)
    values = [float(reference_params[index])]
    for row in history or []:
        values.append(float(row[param_name]))

    low = min(values)
    high = max(values)
    span = high - low
    if span < 1e-9:
        span = max(abs(low), 1.0)
    padding = 0.25 * span
    return np.linspace(low - padding, high + padding, int(resolution))


def loss_surface_figure(
    data,
    history,
    current_params,
    degree: int,
    loss_function: str,
    x_param: str,
    y_param: str,
    resolution: int = 35,
) -> go.Figure:
    transform = feature_transform_from_data(data)
    x_axis = parameter_axis_values(history, current_params, x_param, resolution)
    y_axis = parameter_axis_values(history, current_params, y_param, resolution)
    x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)

    base_params = np.asarray(current_params, dtype=float).copy()
    x_index = param_index(x_param)
    y_index = param_index(y_param)
    z_mesh = np.zeros_like(x_mesh, dtype=float)

    x_values = data["x"].to_numpy(dtype=float)
    observed = data["y"].to_numpy(dtype=float)
    for row_index in range(x_mesh.shape[0]):
        for col_index in range(x_mesh.shape[1]):
            candidate = base_params.copy()
            candidate[x_index] = x_mesh[row_index, col_index]
            candidate[y_index] = y_mesh[row_index, col_index]
            z_mesh[row_index, col_index] = optimization_loss(
                x_values, observed, candidate, degree, loss_function, transform
            )

    figure = go.Figure()
    figure.add_trace(
        go.Surface(
            x=x_mesh,
            y=y_mesh,
            z=z_mesh,
            colorscale="Viridis",
            opacity=0.85,
            showscale=False,
            name="Loss surface",
            hovertemplate=(
                f"{x_param}=%{{x:.4f}}<br>"
                f"{y_param}=%{{y:.4f}}<br>"
                f"{loss_function}=%{{z:.6f}}<extra></extra>"
            ),
        )
    )

    if history:
        figure.add_trace(
            go.Scatter3d(
                x=[row[x_param] for row in history],
                y=[row[y_param] for row in history],
                z=[row["optimization_loss"] for row in history],
                mode="lines+markers",
                name="Gradient descent path",
                line=dict(width=6, color="#ff8c42"),
                marker=dict(size=4, color="#ffd166"),
                hovertemplate=(
                    f"Iteration=%{{text}}<br>{x_param}=%{{x:.4f}}<br>{y_param}=%{{y:.4f}}<br>"
                    f"{loss_function}=%{{z:.6f}}<extra></extra>"
                ),
                text=[row["iteration"] for row in history],
            )
        )
        final_row = history[-1]
        figure.add_trace(
            go.Scatter3d(
                x=[final_row[x_param]],
                y=[final_row[y_param]],
                z=[final_row["optimization_loss"]],
                mode="markers",
                name="Current point",
                marker=dict(size=7, color="#ef476f", symbol="diamond"),
                text=[final_row["iteration"]],
                hovertemplate=(
                    f"Iteration=%{{text}}<br>{x_param}=%{{x:.4f}}<br>{y_param}=%{{y:.4f}}<br>"
                    f"{loss_function}=%{{z:.6f}}<extra></extra>"
                ),
            )
        )
    else:
        current_loss = optimization_loss(
            x_values, observed, current_params, degree, loss_function, transform
        )
        figure.add_trace(
            go.Scatter3d(
                x=[base_params[x_index]],
                y=[base_params[y_index]],
                z=[current_loss],
                mode="markers",
                name="Current point",
                marker=dict(size=7, color="#ef476f", symbol="diamond"),
                hovertemplate=(
                    f"{x_param}=%{{x:.4f}}<br>{y_param}=%{{y:.4f}}<br>"
                    f"{loss_function}=%{{z:.6f}}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title=f"3D Loss Surface - {loss_function} over {x_param} and {y_param}",
        height=620,
        margin=dict(l=0, r=0, t=70, b=0),
        scene=dict(
            xaxis_title=x_param,
            yaxis_title=y_param,
            zaxis_title=loss_function,
        ),
    )
    return figure


def _empty_state_figure(title: str, height: int, message: str) -> go.Figure:
    return go.Figure().update_layout(
        title=title,
        height=height,
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
            }
        ],
    )


def loss_figure(
    history,
    data,
    degree: int,
    tolerance: float,
    divergence_iteration: int | None = None,
    loss_function: str = "MSE",
) -> go.Figure:
    history, detected_divergence = prepare_history_for_display(history, data, degree, loss_function)
    divergence_iteration = detected_divergence if detected_divergence is not None else divergence_iteration

    if not history:
        message = "Run training or build animation to show loss."
        if divergence_iteration is not None:
            message = (
                f"Training diverged at iteration {divergence_iteration}. "
                "Lower the learning rate or simplify the model."
            )
        return _empty_state_figure("Loss vs Iterations", 360, message)

    history_frame = pd.DataFrame(history)
    convergence_at = convergence_iteration(history, tolerance, metric_key="optimization_loss")

    if divergence_iteration is not None:
        subtitle = (
            f"Diverged at iteration {divergence_iteration}; "
            f"showing stable history through iteration {int(history_frame['iteration'].max())}"
        )
    elif convergence_at is not None:
        subtitle = f"Estimated convergence at iteration {convergence_at} (delta {loss_function} < {tolerance:g})"
    else:
        subtitle = (
            f"No convergence detected within {int(history_frame['iteration'].max())} iterations "
            f"(delta {loss_function} < {tolerance:g})"
        )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history_frame["iteration"],
            y=history_frame["optimization_loss"],
            mode="lines+markers",
            name=optimization_loss_label(loss_function),
            line=dict(width=3),
            marker=dict(size=4),
            hovertemplate=(
                f"Iteration=%{{x}}<br>{optimization_loss_label(loss_function)}=%{{y:.6f}}"
                "<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=history_frame["iteration"],
            y=history_frame["true_optimization_loss"],
            mode="lines",
            name=optimization_loss_label(loss_function, "True-function"),
            line=dict(width=3, dash="dash"),
            hovertemplate=(
                f"Iteration=%{{x}}<br>"
                f"{optimization_loss_label(loss_function, 'True-function')}=%{{y:.6f}}<extra></extra>"
            ),
        )
    )

    if convergence_at is not None:
        convergence_loss = history_frame.loc[
            history_frame["iteration"] == convergence_at, "optimization_loss"
        ].iloc[0]
        figure.add_trace(
            go.Scatter(
                x=[convergence_at],
                y=[convergence_loss],
                mode="markers",
                name="Convergence point",
                marker=dict(size=14, symbol="star"),
                hovertemplate=(
                    f"Convergence iteration=%{{x}}<br>{loss_function}=%{{y:.6f}}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title=f"Loss vs Iterations<br><sup>{subtitle}</sup>",
        height=360,
        margin=dict(l=30, r=30, t=80, b=30),
        xaxis_title="Iteration",
        yaxis_title=loss_axis_title(loss_function),
        xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.2)", tickformat=".3~s"),
    )
    return figure


def animated_dashboard_figure(
    data,
    history,
    degree: int,
    tolerance: float,
    divergence_iteration: int | None = None,
    loss_function: str = "MSE",
) -> go.Figure:
    history, detected_divergence = prepare_history_for_display(history, data, degree, loss_function)
    divergence_iteration = detected_divergence if detected_divergence is not None else divergence_iteration

    if not history:
        message = "Build animation to show model-fit and loss playback."
        if divergence_iteration is not None:
            message = (
                f"Training diverged at iteration {divergence_iteration}. "
                "Lower the learning rate or simplify the model."
            )
        return _empty_state_figure("Gradient Descent Animation", 760, message)

    transform = feature_transform_from_data(data)
    x_curve = np.linspace(-4.3, 4.3, 200)
    history_frame = pd.DataFrame(history)
    first_row = history[0]
    first_params = params_from_row(first_row)
    max_iteration = int(history_frame["iteration"].max())
    convergence_at = convergence_iteration(history, tolerance, metric_key="optimization_loss")

    if divergence_iteration is not None:
        convergence_text = (
            f"Diverged at iteration {divergence_iteration}; showing stable frames only"
        )
    elif convergence_at is None:
        convergence_text = f"No convergence detected within {max_iteration} iterations"
    else:
        convergence_text = f"Estimated convergence: iteration {convergence_at}"

    figure = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.62, 0.38],
        specs=[[{"type": "xy"}], [{"type": "xy"}]],
        subplot_titles=("Model fit", f"Loss vs Iterations - {convergence_text}"),
        vertical_spacing=0.16,
    )

    figure.add_trace(
        go.Scatter(
            x=data["x"],
            y=data["y_true"],
            mode="lines",
            name="Underlying distribution",
            line=dict(width=3, dash="dot"),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=data["x"],
            y=data["y"],
            mode="markers",
            name="Data points",
            marker=dict(size=9),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=x_curve,
            y=predict_values(x_curve, first_params, degree, transform),
            mode="lines",
            name=equation_text(first_params, degree, "z"),
            line=dict(width=4),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=history_frame["iteration"],
            y=history_frame["optimization_loss"],
            mode="lines",
            name=optimization_loss_label(loss_function),
            line=dict(width=3),
            hovertemplate=(
                f"Iteration=%{{x}}<br>{optimization_loss_label(loss_function)}=%{{y:.6f}}"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=history_frame["iteration"],
            y=history_frame["true_optimization_loss"],
            mode="lines",
            name=optimization_loss_label(loss_function, "True-function"),
            line=dict(width=3, dash="dash"),
            hovertemplate=(
                f"Iteration=%{{x}}<br>"
                f"{optimization_loss_label(loss_function, 'True-function')}=%{{y:.6f}}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=[first_row["iteration"]],
            y=[first_row["optimization_loss"]],
            mode="markers",
            name="Current iteration",
            marker=dict(size=14, symbol="circle"),
            hovertemplate=f"Current iteration=%{{x}}<br>{loss_function}=%{{y:.6f}}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    if convergence_at is not None:
        convergence_loss = history_frame.loc[
            history_frame["iteration"] == convergence_at, "optimization_loss"
        ].iloc[0]
        figure.add_trace(
            go.Scatter(
                x=[convergence_at],
                y=[convergence_loss],
                mode="markers",
                name="Convergence point",
                marker=dict(size=15, symbol="star"),
                hovertemplate=(
                    f"Convergence iteration=%{{x}}<br>{loss_function}=%{{y:.6f}}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    base_annotations = list(figure.layout.annotations)
    frames = []
    for row in history:
        params = params_from_row(row)
        frames.append(
            go.Frame(
                name=str(row["iteration"]),
                data=[
                    go.Scatter(
                        x=x_curve,
                        y=predict_values(x_curve, params, degree, transform),
                        name=equation_text(params, degree, "z"),
                    ),
                    go.Scatter(
                        x=[row["iteration"]],
                        y=[row["optimization_loss"]],
                    ),
                ],
                traces=[2, 5],
                layout=go.Layout(
                    title_text=(
                        f"Gradient Descent Animation - Iteration {row['iteration']}, "
                        f"{loss_function} {row['optimization_loss']:.6f}"
                    ),
                    annotations=base_annotations,
                ),
            )
        )

    figure.frames = frames
    figure.update_xaxes(
        title_text="x",
        range=[-4.5, 4.5],
        gridcolor="rgba(128,128,128,0.2)",
        row=1,
        col=1,
    )
    figure.update_yaxes(title_text="y", gridcolor="rgba(128,128,128,0.2)", row=1, col=1)
    figure.update_xaxes(
        title_text="Iteration",
        gridcolor="rgba(128,128,128,0.2)",
        row=2,
        col=1,
    )
    figure.update_yaxes(
        title_text=loss_axis_title(loss_function),
        gridcolor="rgba(128,128,128,0.2)",
        tickformat=".3~s",
        row=2,
        col=1,
    )
    figure.update_layout(
        title=(
            f"Gradient Descent Animation - Iteration 0, "
            f"{loss_function} {first_row['optimization_loss']:.6f}"
        ),
        height=760,
        margin=dict(l=30, r=30, t=90, b=130),
        hovermode="closest",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "showactive": False,
                "x": 0.0,
                "y": -0.09,
                "xanchor": "left",
                "yanchor": "top",
                "pad": {"t": 0, "r": 12},
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 45, "redraw": False},
                                "transition": {"duration": 20},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Iteration: ", "font": {"size": 12}},
                "x": 0.14,
                "y": -0.09,
                "xanchor": "left",
                "len": 0.86,
                "pad": {"t": 0, "b": 0},
                "steps": [
                    {
                        "label": animation_slider_label(row["iteration"], max_iteration),
                        "method": "animate",
                        "args": [
                            [str(row["iteration"])],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    }
                    for row in history
                ],
            }
        ],
    )
    return figure
