"""Figures for the data-preparation playground."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from data_playground.cleaning import OUTLIER_LIMIT
from data_playground.speed import Timing
from playground_common.palette import ACCENT, BORDER, TRUTH, layout
from playground_common.wording import duration


def cleaned_histogram(signal: np.ndarray, filled: np.ndarray, true_mean: float) -> go.Figure:
    """The cleaned signal, with the filled cells drawn apart from the measured ones.

    A gap filled with a sensible value disappears into the bulk of the
    distribution. A gap filled with a statistic polluted by outliers stands out
    as a single spike far from it — the error the cleaning itself introduced.
    """
    figure = go.Figure()
    for name, values, colour in (
        ("измерено", signal[~filled], BORDER),
        ("заполнено", signal[filled], ACCENT),
    ):
        figure.add_trace(
            go.Histogram(
                x=values,
                name=name,
                marker_color=colour,
                xbins={"size": 1.0},
                opacity=0.85,
                hovertemplate="%{x} дБм: %{y}<extra>" + name + "</extra>",
            )
        )
    figure.add_vline(
        x=true_mean,
        line={"color": TRUTH, "width": 2, "dash": "dash"},
        annotation_text="истинное среднее",
        annotation_position="top left",
    )
    if signal.min() < OUTLIER_LIMIT:
        figure.add_vline(
            x=OUTLIER_LIMIT,
            line={"color": ACCENT, "width": 1, "dash": "dot"},
            annotation_text="физический предел",
            annotation_position="bottom right",
        )
    low = min(float(signal.min()) - 5, true_mean - 30)
    figure.update_layout(
        **layout(
            height=320,
            barmode="overlay",
            xaxis={"title": "сигнал после очистки, дБм", "range": [low, true_mean + 30]},
            yaxis={"title": "строк"},
            legend={"orientation": "h", "y": 1.12, "x": 0},
        )
    )
    return figure


def timing_figure(timings: tuple[Timing, ...]) -> go.Figure:
    """Milliseconds per method, on a log axis: the gap spans orders of magnitude."""
    labels = [duration(timing.seconds) for timing in timings]
    figure = go.Figure(
        go.Bar(
            x=[timing.seconds * 1000 for timing in timings],
            y=[timing.method for timing in timings],
            orientation="h",
            marker_color=[ACCENT, BORDER, BORDER],
            text=labels,
            textposition="outside",
            customdata=labels,
            hovertemplate="%{y}: %{customdata}<extra></extra>",
        )
    )
    figure.update_layout(
        **layout(
            height=220,
            margin={"l": 220, "r": 60, "t": 20, "b": 50},
            # One tick per power of ten: left to itself, plotly labels every digit
            # of a narrow log range, «4 5 6 7 8 9 0.1 2 3 …», which reads as noise.
            xaxis={"title": "время, мс (логарифмическая шкала)", "type": "log", "dtick": 1},
            yaxis={"autorange": "reversed", "type": "category"},
        )
    )
    return figure
