"""Figures for the network playground: the learning curve and the decision boundary."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from nn_playground.network import forward
from playground_common.palette import ACCENT, BORDER, TRUTH, layout

MESH = 160


def history_figure(history: tuple[float, ...], reference: tuple[float, ...] | None = None) -> go.Figure:
    """Loss per epoch. With a bug switched on, the correct run is drawn behind it for scale."""
    figure = go.Figure()
    if reference is not None:
        figure.add_trace(
            go.Scatter(
                y=list(reference),
                mode="lines",
                name="без ошибки",
                line={"color": TRUTH, "width": 2, "dash": "dash"},
            )
        )
    figure.add_trace(
        go.Scatter(
            y=list(history),
            mode="lines",
            name="этот прогон",
            line={"color": ACCENT if reference is not None else BORDER, "width": 3},
        )
    )
    figure.add_hline(
        y=float(np.log(2)),
        line={"color": TRUTH, "width": 1, "dash": "dot"},
        annotation_text="ln 2: ответ 0.5 для всех",
        annotation_position="top right",
    )
    top = max(1.0, min(max(history), 3.0))
    figure.update_layout(
        **layout(
            height=320,
            xaxis={"title": "эпоха"},
            yaxis={"title": "потери", "range": [0, top]},
            legend={"orientation": "h", "y": 1.12, "x": 0},
        )
    )
    return figure


def boundary_figure(params, test_x: np.ndarray, test_y: np.ndarray) -> go.Figure:
    """The network's probability over the plane, with the held-out points on top."""
    xs = np.linspace(-2, 3, MESH)
    ys = np.linspace(-1.5, 2, MESH)
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    zone = forward(params, np.c_[mesh_x.ravel(), mesh_y.ravel()])[0].reshape(mesh_x.shape)

    figure = go.Figure(
        go.Contour(
            x=xs,
            y=ys,
            z=zone,
            zmin=0,
            zmax=1,
            colorscale=[[0, "#B5CEDE"], [0.5, "#F4F4F4"], [1, "#E8B98E"]],
            contours={"start": 0, "end": 1, "size": 0.1},
            line={"width": 0},
            colorbar={"title": "P(класс 1)", "thickness": 12},
            hovertemplate="P = %{z:.2f}<extra></extra>",
        )
    )
    for label, colour in ((0, BORDER), (1, ACCENT)):
        mask = test_y == label
        figure.add_trace(
            go.Scatter(
                x=test_x[mask, 0],
                y=test_x[mask, 1],
                mode="markers",
                name=f"класс {label}",
                marker={"color": colour, "size": 5, "line": {"width": 0.5, "color": "white"}},
                hoverinfo="skip",
            )
        )
    figure.update_layout(
        **layout(
            height=360,
            xaxis={"title": "x1", "range": [-2, 3]},
            yaxis={"title": "x2", "range": [-1.5, 2]},
            legend={"orientation": "h", "y": 1.1, "x": 0},
        )
    )
    return figure
