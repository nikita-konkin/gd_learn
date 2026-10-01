"""Figures for the metric playground."""

from __future__ import annotations

import plotly.graph_objects as go

from metric_playground.metrics import Confusion
from playground_common.palette import ACCENT, BORDER, TRUTH, layout

REMEDY_COLOURS = {"plain": TRUTH, "weighted": BORDER, "undersampled": ACCENT}


def confusion_figure(counts: Confusion) -> go.Figure:
    """The four counts as a 2×2 grid: rows are reality, columns are the model's answer."""
    grid = [
        [counts.true_negative, counts.false_positive],
        [counts.false_negative, counts.true_positive],
    ]
    names = [["верно: исправен", "ложная тревога"], ["пропущен отказ", "найден отказ"]]
    figure = go.Figure(
        go.Heatmap(
            z=grid,
            x=["модель: исправен", "модель: отказ"],
            y=["на деле: исправен", "на деле: отказ"],
            text=[[f"{names[row][column]}<br><b>{grid[row][column]}</b>" for column in range(2)] for row in range(2)],
            texttemplate="%{text}",
            colorscale=[[0, "#EDF3F7"], [1, "#B5CEDE"]],
            showscale=False,
            hoverinfo="skip",
        )
    )
    figure.update_layout(**layout(height=260, margin={"l": 120, "r": 20, "t": 20, "b": 40}))
    figure.update_yaxes(autorange="reversed", type="category")
    figure.update_xaxes(type="category", side="top")
    return figure


def curve_figure(curves: dict[str, tuple], points: dict[str, tuple[float, float]], baseline: float) -> go.Figure:
    """Precision against recall for one or more models, with each model's current point marked."""
    figure = go.Figure()
    for key, (name, precision, recall) in curves.items():
        colour = REMEDY_COLOURS.get(key, BORDER)
        figure.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=name,
                line={"color": colour, "width": 2, "shape": "hv"},
                hovertemplate="полнота %{x:.2f}, точность %{y:.2f}<extra>" + name + "</extra>",
            )
        )
        if key in points:
            recall_now, precision_now = points[key]
            figure.add_trace(
                go.Scatter(
                    x=[recall_now],
                    y=[precision_now],
                    mode="markers",
                    marker={"color": colour, "size": 13, "line": {"width": 2, "color": "white"}},
                    showlegend=False,
                    hovertemplate="при выбранном пороге: полнота %{x:.2f}, точность %{y:.2f}<extra></extra>",
                )
            )
    figure.add_hline(
        y=baseline,
        line={"color": TRUTH, "dash": "dash"},
        annotation_text=f"случайное ранжирование: {baseline:.3f}",
        annotation_position="bottom left",
    )
    figure.update_layout(
        **layout(
            height=340,
            xaxis={"title": "полнота", "range": [0, 1.02]},
            yaxis={"title": "точность", "range": [0, 1.05]},
            legend={"orientation": "h", "y": 1.12, "x": 0},
        )
    )
    return figure
