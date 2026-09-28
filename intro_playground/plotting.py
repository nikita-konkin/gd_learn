"""Figures for the intro-lecture playground, in the lecture figures' colours."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from intro_playground.length import Line

# The course figure script's palette, so a figure here and one on the slide read the same.
TYPE_COLORS = {
    "интерфейс": "#2B4C6F",
    "документация": "#3E7D5A",
    "маркетинг": "#B03A2B",
    "юридический": "#6B5B95",
}
INK = "#1B2433"
ASH = "#8B9299"
REFERENCE = "#B03A2B"
PMI_COLOR = "#3E7D5A"

LAYOUT = {"template": "plotly_white", "margin": {"l": 10, "r": 10, "t": 50, "b": 10}}


def depth_figure(sweep: pd.DataFrame, depth: int, baseline: float, reference: float) -> go.Figure:
    """Accuracy at every depth, the chosen depth marked, the two yardsticks drawn."""
    figure = go.Figure(
        go.Scatter(
            x=sweep["depth"],
            y=sweep["accuracy"],
            mode="lines+markers",
            line={"color": INK, "width": 3},
            marker={"size": [14 if value == depth else 8 for value in sweep["depth"]]},
            hovertemplate="глубина %{x}: %{y:.3f}<extra></extra>",
            showlegend=False,
        )
    )
    figure.add_hline(
        y=reference,
        line={"color": REFERENCE, "dash": "dash", "width": 1.5},
        annotation_text="символьные n-граммы, Л.р. № 1",
        annotation_position="bottom right",
    )
    figure.add_hline(
        y=baseline,
        line={"color": ASH, "dash": "dot", "width": 1},
        annotation_text="самый частый класс",
        annotation_position="bottom right",
    )
    figure.update_layout(
        title="Точность и глубина дерева",
        xaxis={"title": "глубина дерева", "dtick": 1},
        yaxis={"title": "точность", "range": [0, 1]},
        **LAYOUT,
    )
    return figure


def length_figure(table: pd.DataFrame, line: Line, ids) -> go.Figure:
    """Every segment by its source and translation length, coloured by content type."""
    figure = go.Figure()
    for content_type, color in TYPE_COLORS.items():
        part = table[table["type"] == content_type]
        figure.add_trace(
            go.Scatter(
                x=part["en"],
                y=part["ru"],
                mode="markers",
                name=content_type,
                marker={"color": color, "size": 8, "opacity": 0.75, "line": {"color": "white", "width": 0.5}},
                customdata=[[ids[index], ratio - 1] for index, ratio in part["ratio"].items()],
                hovertemplate="%{customdata[0]}: %{x:.0f} → %{y:.0f} симв. (%{customdata[1]:+.0%})<extra></extra>",
            )
        )
    edge = [0.0, float(table["en"].max()) * 1.03]
    figure.add_trace(
        go.Scatter(
            x=edge,
            y=[line.predict(value) for value in edge],
            mode="lines",
            name=f"регрессия: ru = {line.slope:.2f}·en + {line.intercept:.1f}",
            line={"color": INK, "width": 2},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=edge, y=edge, mode="lines", name="равная длина", line={"color": ASH, "dash": "dash"}, hoverinfo="skip"
        )
    )
    figure.update_layout(
        title="Длина перевода против длины оригинала",
        xaxis_title="оригинал, символов",
        yaxis_title="перевод, символов",
        legend={"orientation": "h", "y": -0.2},
        **LAYOUT,
    )
    return figure


def overflow_figure(shares: pd.Series, margin: float) -> go.Figure:
    """Share of each type's segments that overrun the reserved room."""
    figure = go.Figure(
        go.Bar(
            x=list(shares.index),
            y=list(shares.values),
            marker_color=[TYPE_COLORS[name] for name in shares.index],
            text=[f"{value:.0%}" for value in shares.values],
            textposition="outside",
            hovertemplate="%{x}: %{y:.0%}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"Не влезает при запасе {margin:.0%}",
        yaxis={"title": "доля сегментов", "range": [0, 1.1], "tickformat": ".0%"},
        **LAYOUT,
    )
    return figure


def pairs_figure(top: pd.DataFrame, value: str, title: str, color: str, axis: str) -> go.Figure:
    """Horizontal bars, strongest pair on top."""
    rows = top.iloc[::-1]
    figure = go.Figure(
        go.Bar(
            x=rows[value],
            y=rows["pair"],
            orientation="h",
            marker_color=color,
            customdata=rows[["count", "pmi"]].to_numpy(),
            hovertemplate="%{y}: вхождений %{customdata[0]}, PMI %{customdata[1]:.2f}<extra></extra>",
        )
    )
    figure.update_layout(title=title, xaxis_title=axis, height=380, **LAYOUT)
    return figure
