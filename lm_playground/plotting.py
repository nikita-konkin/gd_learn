"""Графики playground'а языковой модели."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lm_playground.evaluation import OrderResult

TRAIN_COLOR = "#8C8C8C"
HELDOUT_COLOR = "#4C72B0"
COPIED_COLOR = "#B03A2B"
RAW_COLOR = "#8C8C8C"
PREPARED_COLOR = "#4C72B0"


def overfitting_figure(results: list[OrderResult], current_order: int) -> go.Figure:
    """Перплексия и доля списанного как функции порядка модели.

    Две оси намеренно на одном полотне: вывод работы в том, что минимум
    отложенной перплексии и взлёт доли списанного происходят рядом.
    """
    orders = [result.order for result in results]
    heldout = [result.heldout_perplexity for result in results]

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=orders,
            y=[result.train_perplexity for result in results],
            name="перплексия на обучении",
            mode="lines+markers",
            line={"color": TRAIN_COLOR, "dash": "dot"},
            hovertemplate="n=%{x}: %{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=orders,
            y=heldout,
            name="перплексия на отложенных",
            mode="lines+markers",
            line={"color": HELDOUT_COLOR, "width": 3},
            hovertemplate="n=%{x}: %{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=orders,
            y=[result.copied_fraction * 100 for result in results],
            name="списано дословно, %",
            mode="lines+markers",
            line={"color": COPIED_COLOR, "width": 3},
            hovertemplate="n=%{x}: %{y:.1f} %<extra></extra>",
        ),
        secondary_y=True,
    )

    best_order = orders[heldout.index(min(heldout))]
    figure.add_vline(
        x=best_order,
        line_dash="dash",
        line_color=HELDOUT_COLOR,
        annotation_text=f"лучший n = {best_order}",
        annotation_position="top left",
    )
    if current_order != best_order:
        figure.add_vline(
            x=current_order,
            line_color="#2E8B57",
            annotation_text=f"выбран n = {current_order}",
            annotation_position="top right",
        )

    figure.update_layout(
        title="Чем длиннее контекст, тем складнее текст — и тем больше списано",
        xaxis_title="порядок модели n (символов контекста)",
        height=420,
        margin={"l": 60, "r": 60, "t": 60, "b": 40},
        legend={"orientation": "h", "y": -0.2},
    )
    figure.update_yaxes(title_text="перплексия", secondary_y=False)
    figure.update_yaxes(title_text="списано дословно, %", range=[0, 105], secondary_y=True)
    return figure


def distribution_figure(
    raw: dict[str, float],
    prepared: dict[str, float],
    top: int = 12,
) -> go.Figure:
    """Распределение следующего символа до и после ручек сэмплирования."""
    ranked = sorted(raw, key=lambda key: raw[key], reverse=True)[:top]
    labels = ["⏎" if character == "\n" else ("␣" if character == " " else character) for character in ranked]

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=labels,
            y=[raw[character] for character in ranked],
            name="модель",
            marker_color=RAW_COLOR,
            opacity=0.55,
            hovertemplate="%{x}: %{y:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=labels,
            y=[prepared.get(character, 0.0) for character in ranked],
            name="после температуры и отсечения",
            marker_color=PREPARED_COLOR,
            hovertemplate="%{x}: %{y:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        barmode="group",
        title="Из чего выбирается следующий символ",
        xaxis_title="символ",
        yaxis_title="вероятность",
        height=340,
        margin={"l": 60, "r": 20, "t": 50, "b": 40},
        legend={"orientation": "h", "y": -0.25},
    )
    return figure
