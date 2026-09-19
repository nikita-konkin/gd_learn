"""Figures for the playground. Plotly throughout, as in the gradient one."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

OK_COLOR = "#2E8B57"
WARN_COLOR = "#B03A2B"
BASE_COLOR = "#8C8C8C"
ACCENT_COLOR = "#4C72B0"


def metric_comparison_figure(current: dict[str, float], baseline: dict[str, float]) -> go.Figure:
    """The edited translation's measures against the model's original output.

    TER is shown as it is. Lower is better for it, so the axis label says so
    rather than the scale being quietly flipped.
    """
    names = ["BLEU", "chrF", "TER"]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=names,
            y=[baseline[name] for name in names],
            name="выход модели",
            marker_color=BASE_COLOR,
            opacity=0.55,
            hovertemplate="%{x} исходный: %{y:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=names,
            y=[current[name] for name in names],
            name="ваш вариант",
            marker_color=ACCENT_COLOR,
            hovertemplate="%{x} сейчас: %{y:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        barmode="group",
        title="Метрики: ваш вариант против выхода модели",
        yaxis_title="значение (у TER меньше — лучше)",
        height=320,
        margin={"l": 50, "r": 20, "t": 50, "b": 40},
        legend={"orientation": "h", "y": -0.2},
    )
    return figure


def coverage_figure(table: pd.DataFrame) -> go.Figure:
    """How many segments of the corpus each instrument sends back for review."""
    colors = [WARN_COLOR if value > 20 else OK_COLOR for value in table["доля корпуса"]]
    figure = go.Figure(
        go.Bar(
            x=table["доля корпуса"],
            y=table["средство"],
            orientation="h",
            marker_color=colors,
            text=[
                f"{count} сегм. · {share:.1f} %"
                for count, share in zip(table["на проверку"], table["доля корпуса"], strict=True)
            ],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f} % корпуса<extra></extra>",
        )
    )
    figure.update_layout(
        title="Цена тревоги: сколько корпуса уйдёт на ручную проверку",
        xaxis_title="доля корпуса, %",
        xaxis_range=[0, 100],
        height=300,
        margin={"l": 10, "r": 60, "t": 50, "b": 40},
    )
    return figure


def blind_spot_figure(corpus: pd.DataFrame, bleu_threshold: float, selected_id: str | None = None) -> go.Figure:
    """BLEU against semantic similarity, with fired checks in red.

    The red points in the top right corner are the blind spot: by both measures
    the translation looks good, and the string is broken.
    """
    clean = corpus[~corpus["есть_замечания"]]
    flagged = corpus[corpus["есть_замечания"]]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=clean["BLEU"],
            y=clean["semantic"],
            mode="markers",
            name="проверки молчат",
            marker={"color": OK_COLOR, "size": 7, "opacity": 0.45},
            customdata=clean[["id", "ru_mt"]],
            hovertemplate="%{customdata[0]}<br>BLEU %{x:.2f} · семантика %{y:.2f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=flagged["BLEU"],
            y=flagged["semantic"],
            mode="markers",
            name="проверка сработала",
            marker={"color": WARN_COLOR, "size": 12, "symbol": "x", "line": {"width": 1}},
            customdata=flagged[["id", "ru_mt"]],
            hovertemplate="%{customdata[0]}<br>BLEU %{x:.2f} · семантика %{y:.2f}<extra></extra>",
        )
    )

    if selected_id is not None:
        chosen = corpus[corpus["id"] == selected_id]
        if not chosen.empty:
            figure.add_trace(
                go.Scatter(
                    x=chosen["BLEU"],
                    y=chosen["semantic"],
                    mode="markers",
                    name="выбранный сегмент",
                    marker={"color": ACCENT_COLOR, "size": 16, "symbol": "circle-open", "line": {"width": 3}},
                    hovertemplate="выбран<extra></extra>",
                )
            )

    figure.add_vline(
        x=bleu_threshold,
        line_dash="dash",
        line_color=ACCENT_COLOR,
        annotation_text=f"порог BLEU {bleu_threshold:.2f}",
        annotation_position="top",
    )
    figure.update_layout(
        title="Слепая зона: что метрики не видят",
        xaxis_title="BLEU",
        yaxis_title="семантическая близость",
        height=460,
        margin={"l": 60, "r": 20, "t": 50, "b": 40},
        legend={"orientation": "h", "y": -0.16},
    )
    return figure


def distribution_figure(corpus: pd.DataFrame, metric: str, current_value: float | None) -> go.Figure:
    """The measure's distribution across the corpus, with the current value marked."""
    figure = go.Figure(
        go.Histogram(
            x=corpus[metric],
            nbinsx=20,
            marker_color=ACCENT_COLOR,
            opacity=0.75,
            hovertemplate=f"{metric} %{{x}}: %{{y}} сегм.<extra></extra>",
        )
    )
    figure.add_vline(
        x=float(corpus[metric].median()),
        line_dash="dot",
        line_color=BASE_COLOR,
        annotation_text=f"медиана {corpus[metric].median():.2f}",
    )
    if current_value is not None:
        figure.add_vline(
            x=current_value,
            line_color=WARN_COLOR,
            line_width=3,
            annotation_text=f"ваш вариант {current_value:.2f}",
            annotation_position="top left",
        )
    figure.update_layout(
        title=f"Распределение {metric} по корпусу",
        xaxis_title=metric,
        yaxis_title="сегментов",
        height=320,
        margin={"l": 50, "r": 20, "t": 50, "b": 40},
    )
    return figure
