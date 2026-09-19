"""Figures for the translation-memory playground."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tm_playground.clustering import REPRESENTATION_LABELS

SEGMENT_COLOR = "#B8BFCC"
LEVENSHTEIN_COLOR = "#B03A2B"
COSINE_COLOR = "#4C72B0"
THRESHOLD_COLOR = "#8C8C8C"
AGREEMENT_COLOR = "#2E8B57"

# The lab's palette, so the map here and the map in the notebook read the same.
TYPE_COLORS = {
    "интерфейс": "#4C72B0",
    "документация": "#55A868",
    "маркетинг": "#C44E52",
    "юридический": "#8172B2",
}


def _wrap(text: str, width: int = 48) -> str:
    """Break a segment across lines so a hover box stays readable."""
    words = str(text).split()
    lines, current = [], ""
    for word in words:
        if current and len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return "<br>".join(lines)


def divergence_figure(
    levenshtein_scores: np.ndarray,
    cosine_scores: np.ndarray,
    segments: list[str],
    threshold: float,
) -> go.Figure:
    """Every memory segment placed by both measures at once.

    One point per segment: how many characters it shares with the query across,
    how many weighted n-grams it shares with it up. Agreement would put the
    cloud on a diagonal. What you actually get is the two winners sitting in
    different corners, and those corners are the lab's exercise.
    """
    levenshtein_scores = np.asarray(levenshtein_scores, dtype=float)
    cosine_scores = np.asarray(cosine_scores, dtype=float)
    best_levenshtein = int(np.argmax(levenshtein_scores))
    best_cosine = int(np.argmax(cosine_scores))

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=levenshtein_scores,
            y=cosine_scores,
            mode="markers",
            name="сегменты памяти",
            marker={"color": SEGMENT_COLOR, "size": 8, "line": {"width": 0.5, "color": "white"}},
            customdata=[_wrap(segment) for segment in segments],
            hovertemplate="%{customdata}<br>Левенштейн %{x:.1f}%<br>косинус %{y:.3f}<extra></extra>",
        )
    )

    highlights = [
        (best_levenshtein, "выбор Левенштейна", LEVENSHTEIN_COLOR, "diamond"),
        (best_cosine, "выбор косинуса", COSINE_COLOR, "star"),
    ]
    for index, name, colour, symbol in highlights:
        figure.add_trace(
            go.Scatter(
                x=[levenshtein_scores[index]],
                y=[cosine_scores[index]],
                mode="markers",
                name=name,
                marker={"color": colour, "size": 18, "symbol": symbol,
                        "line": {"width": 1.5, "color": "white"}},
                customdata=[_wrap(segments[index])],
                hovertemplate="%{customdata}<br>Левенштейн %{x:.1f}%<br>косинус %{y:.3f}<extra></extra>",
            )
        )

    figure.add_vline(
        x=threshold,
        line_dash="dash",
        line_color=THRESHOLD_COLOR,
        annotation_text=f"порог {threshold:.0f}%",
        annotation_position="top left",
    )
    title = (
        "Меры согласны" if best_levenshtein == best_cosine else "Меры выбрали разные сегменты"
    )
    figure.update_layout(
        title=title,
        xaxis_title="процент совпадения по Левенштейну",
        yaxis_title="косинус по символьным n-граммам",
        xaxis_range=[0, 100],
        height=440,
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "y": -0.18},
    )
    return figure


def threshold_figure(sweep: pd.DataFrame, current: float) -> go.Figure:
    """How many of the queries still get a suggestion as the cut-off rises."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=sweep["threshold"],
            y=sweep["offered"],
            mode="lines+markers",
            name="подсказка показана",
            line={"color": LEVENSHTEIN_COLOR, "width": 3},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=sweep["threshold"],
            y=sweep["agreeing"],
            mode="lines+markers",
            name="и косинус выбрал то же",
            line={"color": AGREEMENT_COLOR, "width": 3, "dash": "dot"},
        )
    )
    figure.add_vline(
        x=current,
        line_color=THRESHOLD_COLOR,
        line_width=2,
        annotation_text=f"{current:.0f}%",
        annotation_position="top right",
    )
    figure.update_layout(
        title=f"Порог решает, сколько подсказок увидит переводчик (всего запросов: {int(sweep['total'].iloc[0])})",
        xaxis_title="порог совпадения, %",
        yaxis_title="запросов с подсказкой",
        yaxis_range=[0, int(sweep["total"].iloc[0])],
        height=380,
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "y": -0.25},
    )
    return figure


def agreement_figure(table: pd.DataFrame) -> go.Figure:
    """ARI per representation, with the spread across starting points behind it."""
    labels = [REPRESENTATION_LABELS.get(name, name) for name in table["representation"]]
    figure = go.Figure(
        go.Bar(
            x=table["ari"],
            y=labels,
            orientation="h",
            marker_color=COSINE_COLOR,
            error_x={
                "type": "data",
                "symmetric": False,
                "array": table["ari_high"] - table["ari"],
                "arrayminus": table["ari"] - table["ari_low"],
                "visible": True,
                "thickness": 1.5,
            },
            hovertemplate="%{y}: ARI %{x:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Совпадение кластеров с разметкой: не ноль, но и не результат",
        xaxis_title="ARI (1.0 — разметка воспроизведена, 0.0 — случайное разбиение)",
        xaxis_range=[0, 1],
        height=320,
        margin={"l": 10, "r": 30, "t": 60, "b": 40},
        yaxis={"autorange": "reversed"},
    )
    return figure


def map_figure(coordinates: np.ndarray, content_types) -> go.Figure:
    """The corpus on a plane, coloured by the labelling clustering could not find."""
    content_types = list(content_types)
    figure = go.Figure()
    for content_type, colour in TYPE_COLORS.items():
        mask = [index for index, value in enumerate(content_types) if value == content_type]
        if not mask:
            continue
        figure.add_trace(
            go.Scatter(
                x=coordinates[mask, 0],
                y=coordinates[mask, 1],
                mode="markers",
                name=content_type,
                marker={"color": colour, "size": 9, "line": {"width": 0.6, "color": "white"}},
                hovertemplate=f"{content_type}<extra></extra>",
            )
        )
    figure.update_layout(
        title="Карта корпуса: две главные компоненты",
        xaxis_title="компонента 1",
        yaxis_title="компонента 2",
        height=460,
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
        legend={"title": "тип контента"},
    )
    return figure
