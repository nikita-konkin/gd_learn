"""The course's colours, so a playground and a lecture slide read as one set.

Copied as plain hex from ``презентации/diagrams.py`` in the course materials,
where the same constants drive the matplotlib diagrams printed in the PDFs.
Duplicated rather than imported: the materials are not a Python package, and
these files have to run inside a browser with nothing else available.
"""

from __future__ import annotations

FILL = "#E9F0F5"
FILLS = ("#EDF3F7", "#DCE8F0", "#C9DBE7", "#B5CEDE")
BORDER = "#2E5A73"
ARROW = "#4A7590"
TEXT = "#12242F"
ACCENT = "#C4762F"
ACCENT_FILL = "#FBF0E4"

# Roles the figures use, named by meaning rather than by hue, so a chart does
# not have to decide what "the orange one" stands for.
HONEST = BORDER
LEAKY = ACCENT
TRUTH = "#8C8C8C"


def layout(height: int = 360, **overrides) -> dict:
    """Plotly layout shared by every figure.

    Deliberately silent about text colour and grid colour. ``st.plotly_chart``
    applies Streamlit's own template, which follows the reader's light or dark
    preference; naming the course's near-black TEXT here painted the axis labels
    black on a dark background, where they could not be read. The palette's job
    is the colour of the *data* — those are what has to match the slides.
    """
    base = {
        "height": height,
        "margin": {"l": 60, "r": 20, "t": 40, "b": 50},
        "font": {"size": 13},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
    }
    base.update(overrides)
    return base
