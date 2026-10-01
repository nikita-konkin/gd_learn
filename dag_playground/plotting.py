"""The graph, drawn level by level and coloured by what happened to each task."""

from __future__ import annotations

import plotly.graph_objects as go

from dag_playground.scheduler import FAILED, SKIPPED, SUCCESS
from playground_common.palette import ACCENT, BORDER, TRUTH, layout

STATUS_COLOURS = {SUCCESS: BORDER, FAILED: ACCENT, SKIPPED: TRUTH}
NOT_RUN = "#D0D7DE"


def _positions(levels) -> dict[str, tuple[float, float]]:
    """Level on the x axis; tasks of one level spread evenly along y."""
    positions = {}
    for column, level in enumerate(levels):
        count = len(level)
        for row, task in enumerate(level):
            positions[task] = (float(column), (count - 1) / 2 - row)
    return positions


def graph_figure(levels, edges, status: dict[str, str]) -> go.Figure:
    """Tasks as markers, dependencies as arrows, colour as the task's fate.

    Tasks in the same column share a level and could run at the same time —
    lecture 12's point that the graph shows parallelism a script hides.
    """
    positions = _positions(levels)
    figure = go.Figure()

    for source, target in edges:
        if source not in positions or target not in positions:
            continue
        (x0, y0), (x1, y1) = positions[source], positions[target]
        figure.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=1.5,
            arrowcolor=TRUTH,
            standoff=16,
            startstandoff=16,
        )

    tasks = list(positions)
    figure.add_trace(
        go.Scatter(
            x=[positions[task][0] for task in tasks],
            y=[positions[task][1] for task in tasks],
            mode="markers+text",
            text=[task.replace("_", " ") for task in tasks],
            textposition="bottom center",
            marker={
                "size": 26,
                "color": [STATUS_COLOURS.get(status.get(task), NOT_RUN) for task in tasks],
                "line": {"width": 1, "color": BORDER},
            },
            customdata=[status.get(task, "не запускалась") for task in tasks],
            hovertemplate="%{text}: %{customdata}<extra></extra>",
            showlegend=False,
        )
    )
    figure.update_layout(
        **layout(
            height=300,
            xaxis={
                "title": "шаг выполнения",
                "tickmode": "array",
                "tickvals": list(range(len(levels))),
                "ticktext": [str(number) for number in range(1, len(levels) + 1)],
                "showgrid": False,
                "zeroline": False,
                "range": [-0.5, max(len(levels) - 0.5, 0.5)],
            },
            yaxis={"visible": False, "range": [-1.6, 1.0]},
        )
    )
    return figure
