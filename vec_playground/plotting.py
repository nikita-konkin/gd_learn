"""Графики playground'а векторизации."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from vec_playground.evaluation import SweepPoint

WORD_COLOR = "#8C8C8C"
CHAR_COLOR = "#4C72B0"
BASELINE_COLOR = "#B03A2B"
CURRENT_COLOR = "#2E8B57"


def ngram_sweep_figure(points: list[SweepPoint], baseline: float) -> go.Figure:
    """Точность как функция длины n-граммы, отдельно для слов и символов.

    Главный результат работы № 1 в виде кривой: словарные признаки упираются
    в потолок, символьные проходят выше.
    """
    figure = go.Figure()
    for analyzer, colour in (("слова", WORD_COLOR), ("символы внутри слов", CHAR_COLOR)):
        series = [point for point in points if point.analyzer == analyzer]
        if not series:
            continue
        figure.add_trace(
            go.Scatter(
                x=[point.ngram_max for point in series],
                y=[point.accuracy for point in series],
                error_y={
                    "type": "data",
                    "array": [point.deviation for point in series],
                    "visible": True,
                    "color": colour,
                    "thickness": 1,
                },
                mode="lines+markers",
                name=analyzer,
                line={"color": colour, "width": 3},
                customdata=[point.features for point in series],
                hovertemplate="до %{x}-грамм: %{y:.3f}<br>признаков %{customdata}<extra></extra>",
            )
        )

    figure.add_hline(
        y=baseline,
        line_dash="dash",
        line_color=BASELINE_COLOR,
        annotation_text=f"baseline {baseline:.2f}",
        annotation_position="bottom right",
    )
    figure.update_layout(
        title="Символьные n-граммы обгоняют словарные",
        xaxis_title="максимальная длина n-граммы",
        yaxis_title="точность (5-кратная кросс-валидация)",
        yaxis_range=[0, 1],
        height=420,
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "y": -0.2},
    )
    return figure


def comparison_figure(table: pd.DataFrame, baseline: float, current: float | None = None) -> go.Figure:
    """Столбики по конфигурациям — та же картинка, что в разделе 8 работы."""
    figure = go.Figure(
        go.Bar(
            x=table["точность"],
            y=table["конфигурация"],
            orientation="h",
            marker_color=CHAR_COLOR,
            error_x={"type": "data", "array": table["разброс"], "visible": True, "thickness": 1},
            customdata=table["признаков"],
            hovertemplate="%{y}: %{x:.3f}<br>признаков %{customdata}<extra></extra>",
        )
    )
    figure.add_vline(
        x=baseline,
        line_dash="dash",
        line_color=BASELINE_COLOR,
        annotation_text=f"baseline {baseline:.2f}",
    )
    if current is not None:
        figure.add_vline(
            x=current,
            line_color=CURRENT_COLOR,
            line_width=3,
            annotation_text=f"ваша настройка {current:.3f}",
            annotation_position="top left",
        )
    figure.update_layout(
        title="Признаки решают больше, чем модель",
        xaxis_title="точность",
        xaxis_range=[0, 1],
        height=420,
        margin={"l": 10, "r": 30, "t": 60, "b": 40},
        yaxis={"autorange": "reversed"},
    )
    return figure


def confusion_figure(matrix: pd.DataFrame) -> go.Figure:
    """Матрица ошибок: что именно модель путает."""
    figure = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=list(matrix.columns),
            y=list(matrix.index),
            colorscale="Blues",
            showscale=False,
            text=matrix.values,
            texttemplate="%{text}",
            hovertemplate="правильно %{y}, модель сказала %{x}: %{z}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Матрица ошибок",
        xaxis_title="модель сказала",
        yaxis_title="правильный ответ",
        height=380,
        margin={"l": 100, "r": 20, "t": 60, "b": 60},
        yaxis={"autorange": "reversed"},
    )
    return figure
