"""Figures for the leakage playground.

Three, and each one is meant to make something visible that a single averaged
number hides: that the gap holds in every block, that the chosen features do
not survive a change of block, and that the p-values are as flat as pure noise
demands.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from leak_playground.experiment import Selection
from playground_common.palette import BORDER, HONEST, LEAKY, TRUTH, layout


def scores_figure(leaky_folds: tuple[float, ...], honest_folds: tuple[float, ...]) -> go.Figure:
    """Accuracy per block, both pipelines, with the truth drawn across.

    Averages are easy to dismiss as a fluke of one split. Per-block bars are not:
    the leaky pipeline sits above the truth line in every block.
    """
    blocks = [f"блок {number}" for number in range(1, len(leaky_folds) + 1)]
    figure = go.Figure()
    for name, values, colour in (
        ("с утечкой", leaky_folds, LEAKY),
        ("честно", honest_folds, HONEST),
    ):
        figure.add_trace(
            go.Bar(
                x=blocks,
                y=list(values),
                name=name,
                marker_color=colour,
                hovertemplate="%{x}: %{y:.3f}<extra>" + name + "</extra>",
            )
        )
    figure.add_hline(
        y=0.5,
        line={"color": TRUTH, "width": 2, "dash": "dash"},
        annotation_text="истина 0.500",
        annotation_position="top left",
    )
    figure.update_layout(
        **layout(
            height=340,
            barmode="group",
            yaxis={"title": "доля правильных", "range": [0, 1]},
            xaxis={"type": "category"},
            legend={"orientation": "h", "y": 1.12, "x": 0},
        )
    )
    return figure


def recurrence_figure(selected: Selection) -> go.Figure:
    """How many blocks re-pick each feature the whole-sample selection chose.

    A column carrying real signal is re-picked by every block, so the height of
    these bars is the closest thing to evidence the selection can offer. It is
    not proof: on the lecture's pure noise one of the twenty does reach the top
    by luck. What the figure makes visible is how few get anywhere near it.
    """
    order = np.argsort(selected.recurrence)[::-1]
    labels = [f"#{selected.indices[position]}" for position in order]
    values = [selected.recurrence[position] for position in order]

    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=BORDER,
            hovertemplate="признак %{x}: отобран заново в %{y} блоках<extra></extra>",
        )
    )
    figure.add_hline(
        y=selected.folds,
        line={"color": LEAKY, "width": 2, "dash": "dash"},
        annotation_text=f"переотобран во всех {selected.folds} блоках",
        annotation_position="top right",
    )
    figure.update_layout(
        **layout(
            height=320,
            yaxis={
                "title": "в скольких блоках отобран заново",
                "range": [0, selected.folds + 0.6],
                "dtick": 1,
            },
            xaxis={
                "title": "номер признака, отобранного на всей выборке",
                "tickangle": -60,
                # Without this plotly reads "#8", "#25" as numbers and lays them
                # out on a linear axis 0..2000, where twenty bars collapse into
                # hairlines. The labels are names, not quantities.
                "type": "category",
            },
        )
    )
    return figure


def pvalues_figure(p_values: np.ndarray, alpha: float = 0.05) -> go.Figure:
    """The p-values of every column, with the count expected by chance drawn in.

    A flat histogram is the signature of no signal at all. The leftmost bar is
    not evidence: it is the share of a uniform distribution below ``alpha``.
    """
    finite = p_values[np.isfinite(p_values)]
    bins = 20
    counts, edges = np.histogram(finite, bins=bins, range=(0.0, 1.0))
    centres = (edges[:-1] + edges[1:]) / 2
    expected = finite.size / bins

    figure = go.Figure(
        go.Bar(
            x=centres,
            y=counts.tolist(),
            width=1.0 / bins * 0.9,
            marker_color=[LEAKY if centre < alpha else BORDER for centre in centres],
            hovertemplate="p около %{x:.2f}: %{y} признаков<extra></extra>",
        )
    )
    figure.add_hline(
        y=expected,
        line={"color": TRUTH, "width": 2, "dash": "dash"},
        annotation_text=f"ожидаемо без сигнала: {expected:.0f} на столбец",
        annotation_position="top right",
    )
    figure.update_layout(
        **layout(
            height=300,
            yaxis={"title": "признаков"},
            xaxis={"title": "p-значение теста «признак связан с целью»", "range": [0, 1]},
        )
    )
    return figure
