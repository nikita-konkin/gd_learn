"""Figures for the model-complexity playground."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fit_playground.models import Classifier, PolynomialFit, truth
from playground_common.palette import ACCENT, BORDER, TRUTH, layout

SERIES = ("#2E5A73", "#C4762F", "#4A7590", "#8C8C8C", "#7A9E7E", "#B5534A", "#6B5B95", "#D4A017", "#3B8EA5", "#9C6644")


def polynomial_figure(fit: PolynomialFit, train_x, train_y, fresh_x, fresh_y) -> go.Figure:
    """The fitted curve, its training points, and new points from the same law."""
    grid = np.asarray(fit.grid)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=fresh_x.ravel(),
            y=fresh_y,
            mode="markers",
            name="новые точки",
            marker={"color": TRUTH, "size": 5, "opacity": 0.5},
        )
    )
    figure.add_trace(
        go.Scatter(x=grid, y=truth(grid), mode="lines", name="истинная связь", line={"color": TRUTH, "dash": "dash"})
    )
    figure.add_trace(
        go.Scatter(x=grid, y=list(fit.curve), mode="lines", name="модель", line={"color": ACCENT, "width": 3})
    )
    figure.add_trace(
        go.Scatter(
            x=train_x.ravel(),
            y=train_y,
            mode="markers",
            name="обучающие точки",
            marker={"color": BORDER, "size": 9},
        )
    )
    figure.update_layout(
        **layout(
            height=340,
            xaxis={"title": "x", "range": [0, 1]},
            yaxis={"title": "y", "range": [-2, 2]},
            legend={"orientation": "h", "y": 1.14, "x": 0},
        )
    )
    return figure


def degree_curve_figure(degrees, train_errors, fresh_errors, baseline: float, chosen: int) -> go.Figure:
    """Error against degree on the points learned from and on new ones, log scale."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=list(degrees),
            y=list(train_errors),
            mode="lines+markers",
            name="на обучении",
            line={"color": BORDER, "width": 3},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=list(degrees),
            y=list(fresh_errors),
            mode="lines+markers",
            name="на новых точках",
            line={"color": ACCENT, "width": 3},
        )
    )
    figure.add_hline(
        y=baseline,
        line={"color": TRUTH, "dash": "dash"},
        annotation_text="ответ «среднее» для всех",
        annotation_position="top left",
    )
    figure.add_vline(x=chosen, line={"color": TRUTH, "width": 1, "dash": "dot"})
    figure.update_layout(
        **layout(
            height=320,
            xaxis={"title": "степень многочлена", "dtick": 2},
            yaxis={"title": "средний квадрат ошибки", "type": "log"},
            legend={"orientation": "h", "y": 1.14, "x": 0},
        )
    )
    return figure


def boundary_figure(model: Classifier, features: np.ndarray, target: np.ndarray, title: str) -> go.Figure:
    """Probability of class 1 over the plane, with the training points on top."""
    figure = go.Figure(
        go.Contour(
            x=list(model.xs),
            y=list(model.ys),
            z=[list(row) for row in model.zone],
            zmin=0,
            zmax=1,
            colorscale=[[0, "#B5CEDE"], [0.5, "#F4F4F4"], [1, "#E8B98E"]],
            contours={"start": 0, "end": 1, "size": 0.1},
            line={"width": 0},
            showscale=False,
            hoverinfo="skip",
        )
    )
    for label, colour in ((0, BORDER), (1, ACCENT)):
        mask = target == label
        figure.add_trace(
            go.Scatter(
                x=features[mask, 0],
                y=features[mask, 1],
                mode="markers",
                name=f"класс {label}",
                marker={"color": colour, "size": 6, "line": {"width": 0.5, "color": "white"}},
                hoverinfo="skip",
            )
        )
    figure.update_layout(
        **layout(
            height=320,
            title={"text": title, "font": {"size": 14}},
            showlegend=False,
            xaxis={"title": "x1"},
            yaxis={"title": "x2"},
        )
    )
    return figure


def paths_figure(alphas, ridge_paths, lasso_paths, names, chosen: float) -> go.Figure:
    """Two panels of weight trajectories, with the chosen penalty marked."""
    figure = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Ridge", "Lasso"))
    ridge = np.asarray(ridge_paths)
    lasso = np.asarray(lasso_paths)
    for index, name in enumerate(names):
        colour = SERIES[index % len(SERIES)]
        for column, values in ((1, ridge), (2, lasso)):
            figure.add_trace(
                go.Scatter(
                    x=list(alphas),
                    y=values[:, index].tolist(),
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    showlegend=column == 1,
                    line={"color": colour, "width": 2},
                    hovertemplate=f"{name}: " + "%{y:.1f} при alpha %{x:.3g}<extra></extra>",
                ),
                row=1,
                col=column,
            )
    for column in (1, 2):
        figure.add_vline(x=chosen, line={"color": TRUTH, "dash": "dash"}, row=1, col=column)
        figure.update_xaxes(type="log", title_text="сила штрафа alpha", row=1, col=column)
    figure.update_yaxes(title_text="вес признака", row=1, col=1)
    figure.update_layout(**layout(height=360, legend={"orientation": "h", "y": -0.25, "x": 0}))
    return figure
