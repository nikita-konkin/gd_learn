"""Figures for the data-and-annotation playground."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from labels_playground.learning import CHARACTERS, FEATURE_LABELS, WORDS, Curve
from labels_playground.mqm import MISSED_CRITICAL, REJECTED_CORRECT, acceptance, penalties
from labels_playground.wording import segments

# The lab notebooks' palette, so a figure here and one there read the same.
WORDS_COLOR = "#4C72B0"
CHARACTERS_COLOR = "#B03A2B"
NEWSGROUPS_COLOR = "#55A868"
BASELINE_COLOR = "#8C8C8C"
NEUTRAL_COLOR = "#B8BFCC"
KAPPA_COLOR = "#2E8B57"
CHANCE_COLOR = "#DD8452"

FEATURE_COLORS = {WORDS: WORDS_COLOR, CHARACTERS: CHARACTERS_COLOR}
OUTCOME_COLORS = {MISSED_CRITICAL: "#B03A2B", REJECTED_CORRECT: "#DD8452"}

LAYOUT = {"template": "plotly_white", "margin": {"l": 10, "r": 10, "t": 50, "b": 10}}


def _band(curve: Curve, color: str) -> go.Scatter:
    """The fold-to-fold spread around a test curve, as a filled band."""
    upper = list(curve.test + curve.test_spread)
    lower = list(curve.test - curve.test_spread)
    return go.Scatter(
        x=list(curve.sizes) + list(curve.sizes)[::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor=color,
        opacity=0.12,
        line={"width": 0},
        hoverinfo="skip",
        showlegend=False,
    )


def learning_figure(curves: dict[str, Curve], baseline: float, catch_up: int | None) -> go.Figure:
    """Held-out accuracy against training-set size, for both feature sets.

    The training-set accuracy of the character model is drawn too, dotted: the
    gap between it and its held-out line is overfitting, and it does not close.
    """
    figure = go.Figure()
    for feature_set in (WORDS, CHARACTERS):
        curve = curves[feature_set]
        color = FEATURE_COLORS[feature_set]
        figure.add_trace(_band(curve, color))
        figure.add_trace(
            go.Scatter(
                x=curve.sizes,
                y=curve.test,
                mode="lines+markers",
                name=f"{FEATURE_LABELS[feature_set]}, на новых данных",
                line={"color": color, "width": 3},
                hovertemplate="%{x} сегментов: %{y:.3f}<extra></extra>",
            )
        )
    characters = curves[CHARACTERS]
    figure.add_trace(
        go.Scatter(
            x=characters.sizes,
            y=characters.train,
            mode="lines",
            name=f"{FEATURE_LABELS[CHARACTERS]}, на обучающих",
            line={"color": CHARACTERS_COLOR, "width": 1.5, "dash": "dot"},
            hovertemplate="%{x} сегментов: %{y:.3f}<extra></extra>",
        )
    )
    figure.add_hline(
        y=baseline,
        line={"color": BASELINE_COLOR, "dash": "dash", "width": 1},
        annotation_text="самый частый класс",
        annotation_position="bottom right",
    )
    if catch_up is not None:
        words_final = curves[WORDS].final
        figure.add_shape(
            type="line",
            x0=catch_up,
            x1=int(curves[WORDS].sizes[-1]),
            y0=words_final,
            y1=words_final,
            line={"color": "#333333", "width": 1.5, "dash": "dash"},
        )
        figure.add_annotation(
            x=catch_up,
            y=words_final,
            text=segments(catch_up),
            showarrow=True,
            arrowhead=2,
            ax=-40,
            ay=-40,
        )
    figure.update_layout(
        title="Кривая обучения: качество на новых данных по мере разметки",
        xaxis_title="обучающих сегментов",
        yaxis_title="точность",
        yaxis_range=[0, 1.05],
        legend={"orientation": "h", "y": -0.2},
        **LAYOUT,
    )
    return figure


def newsgroups_figure(newsgroups: pd.DataFrame, ours: Curve) -> go.Figure:
    """Lab 1's section 13 figure: the same method on a corpus eleven times larger."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=newsgroups["train_size"],
            y=newsgroups["accuracy"],
            mode="lines+markers",
            name="20 Newsgroups, 4 класса (посчитано в ноутбуке)",
            line={"color": NEWSGROUPS_COLOR, "width": 3},
            hovertemplate="%{x} документов: %{y:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=ours.sizes,
            y=ours.test,
            mode="lines+markers",
            name="наш корпус (посчитано здесь)",
            line={"color": CHARACTERS_COLOR, "width": 2},
            hovertemplate="%{x} сегментов: %{y:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Тот же метод — разный объём данных",
        xaxis_title="обучающих примеров",
        yaxis_title="точность",
        yaxis_range=[0, 1.0],
        legend={"orientation": "h", "y": -0.2},
        **LAYOUT,
    )
    return figure


def mqm_figure(annotations: pd.DataFrame, threshold: float, median: float) -> go.Figure:
    """Each annotated segment by its BLEU and its MQM penalty.

    If BLEU tracked the cost of an error, the points would fall from top left
    to bottom right. The threshold line splits them into accepted and rejected;
    the coloured ones are where that split contradicts the annotation.
    """
    outcome = acceptance(annotations, threshold)
    penalty = penalties(annotations["severity"])
    colors = [OUTCOME_COLORS.get(value, NEUTRAL_COLOR) for value in outcome]
    # Five critical errors share one height, three of them within 0.08 BLEU:
    # alternate their labels above and below so the ids stay readable.
    order = annotations.assign(penalty=penalty).groupby("penalty")["bleu"].rank(method="first").astype(int)
    positions = ["top center" if rank % 2 else "bottom center" for rank in order]
    hover = [
        f"<b>{row.id}</b> · {row.type}<br>эталон: {row.ru_ref}<br>перевод: {row.ru_mt}"
        f"<br>{row.severity}: {row.comment}<br>BLEU {row.bleu:.3f}"
        for row in annotations.itertuples()
    ]
    figure = go.Figure(
        go.Scatter(
            x=annotations["bleu"],
            y=penalty,
            mode="markers+text",
            text=annotations["id"],
            textposition=positions,
            marker={"size": 14, "color": colors, "line": {"color": "#333333", "width": 1}},
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        )
    )
    figure.add_vline(
        x=threshold,
        line={"color": "#333333", "width": 2},
        annotation_text=f"порог {threshold:.2f}",
        annotation_position="top",
    )
    figure.add_vline(
        x=median,
        line={"color": BASELINE_COLOR, "width": 1, "dash": "dot"},
        annotation_text="медиана корпуса",
        # Inside the plot, right of the line: below it the text lands on the 0.4 tick.
        annotation_position="bottom right",
    )
    figure.update_layout(
        title="BLEU против штрафа MQM: правее порога — принято",
        xaxis_title="BLEU",
        yaxis_title="штраф MQM",
        xaxis_range=[0, 1],
        yaxis_range=[-3, 30],
        **LAYOUT,
    )
    return figure


def acceptance_figure(sweep: pd.DataFrame, threshold: float) -> go.Figure:
    """Both kinds of acceptance error at every BLEU threshold.

    A threshold good enough to use would bring both lines to zero at once.
    """
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=sweep["threshold"],
            y=sweep["missed_critical"],
            mode="lines",
            line={"shape": "hv", "color": OUTCOME_COLORS[MISSED_CRITICAL], "width": 3},
            name="принято с критической ошибкой",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=sweep["threshold"],
            y=sweep["rejected_correct"],
            mode="lines",
            line={"shape": "hv", "color": OUTCOME_COLORS[REJECTED_CORRECT], "width": 3},
            name="отклонено без ошибок",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=sweep["threshold"],
            y=sweep["accepted"],
            mode="lines",
            line={"shape": "hv", "color": BASELINE_COLOR, "width": 1.5, "dash": "dot"},
            name="принято всего",
        )
    )
    figure.add_vline(x=threshold, line={"color": "#333333", "width": 2})
    figure.update_layout(
        title="Ошибки приёмки при каждом пороге BLEU",
        xaxis_title="порог BLEU",
        yaxis_title="сегментов из десяти",
        legend={"orientation": "h", "y": -0.25},
        **LAYOUT,
    )
    return figure


def chance_figure(sweep: pd.DataFrame, share: float) -> go.Figure:
    """Raw agreement climbs with the dominant label; kappa does not move."""
    figure = go.Figure()
    for column, name, color, dash in (
        ("raw", "совпадение ответов", CHARACTERS_COLOR, "solid"),
        ("chance", "совпадение по случайности", CHANCE_COLOR, "dash"),
        ("kappa", "каппа Коэна", KAPPA_COLOR, "solid"),
    ):
        figure.add_trace(
            go.Scatter(
                x=sweep["share_no_error"],
                y=sweep[column],
                mode="lines",
                name=name,
                line={"color": color, "width": 3, "dash": dash},
                hovertemplate="%{y:.2f}<extra></extra>",
            )
        )
    figure.add_vline(x=share, line={"color": "#333333", "width": 1.5})
    figure.update_layout(
        title="Чем больше сегментов без ошибок, тем легче совпасть случайно",
        xaxis_title="доля сегментов без ошибок",
        yaxis_title="доля / каппа",
        yaxis_range=[-0.05, 1.05],
        xaxis_tickformat=".0%",
        legend={"orientation": "h", "y": -0.25},
        **LAYOUT,
    )
    return figure


def matrix_figure(matrix: pd.DataFrame) -> go.Figure:
    """Who labelled what: the diagonal is agreement, everything else a dispute."""
    figure = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=list(matrix.columns),
            y=list(matrix.index),
            colorscale=[[0, "#FFFFFF"], [1, WORDS_COLOR]],
            showscale=False,
            text=matrix.values,
            texttemplate="%{text}",
            hovertemplate="первый: %{y}<br>второй: %{x}<br>сегментов: %{z}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Кто что поставил",
        xaxis_title="второй разметчик",
        yaxis_title="первый разметчик",
        yaxis_autorange="reversed",
        **LAYOUT,
    )
    return figure
