"""One place for every colour, marker and mode name the culture figures use.

Colour is load-bearing here: a culture keeps the same hue in every panel, so a
reader who learns the palette once can follow one culture across the whole set.
Splitting the palette out is what makes that promise checkable in one screen.
"""

from __future__ import annotations

MODES: tuple[str, str] = ("NTP", "FA")
"""The paper's two generation strategies, in the order every panel draws them."""

MODE_TITLES = {"NTP": "Next-token probabilities", "FA": "Full answer"}

CULTURE_COLORS = {
    "arabic": "#1b9e77",
    "bengali": "#d95f02",
    "chinese": "#7570b3",
    "english": "#e7298a",
    "german": "#66a61e",
    "korean": "#e6ab02",
    "portuguese": "#a6761d",
    "spanish": "#666666",
    "turkish": "#1f78b4",
}
"""Nine qualitative hues, one per culture, stable across every panel."""

REFERENCE_COLOR = "#000000"

QUALITY_PALETTE = ("#1b9e77", "#66c2a5", "#ffd92f", "#fc8d62", "#d73027")
OUTCOME_PALETTE = {"valid": "#1b9e77", "invalid": "#fc8d62", "failed": "#d73027"}

MDS_REFERENCES: tuple[tuple[str, str], ...] = (
    ("Mixtral archived", "archived"),
    ("Mixtral fresh", "fresh"),
)
"""The two Mixtral series every matched MDS panel draws beside the culture model.

Both are references rather than controls: Mixtral is a different base model, so a
gap between it and a finetuned culture MLLM confounds the base model with the
culture tuning. Drawing the archived and the freshly re-run series together shows
how much of any gap is merely run-to-run noise.
"""

MDS_SERIES_COLORS = {"Mixtral archived": REFERENCE_COLOR, "Mixtral fresh": "#c8c8c8"}
MDS_SERIES_MARKERS = {"Mixtral archived": "o", "Mixtral fresh": "s"}

MDS_CULTURE_COLORS = {"spanish": "#1f78b4"}
"""Culture hues that override ``CULTURE_COLORS`` in the MDS plates only.

These plates spend black and grey on the two Mixtral references, so a finetuned
culture MLLM drawn in a grey cannot be told from the references it is being
compared against. ``CULTURE_COLORS["spanish"]`` is ``#666666``, so Spanish takes
a blue here. Every other culture keeps its usual hue, and ``CULTURE_COLORS``
itself is untouched -- the nine-hue convention still holds everywhere else.
"""


def is_reference(label: str) -> bool:
    """Return whether an MDS series label names a Mixtral reference."""
    return label in MDS_SERIES_COLORS


def mds_color(label: str) -> str:
    """Return the hue one MDS series is drawn in, references included."""
    return MDS_SERIES_COLORS.get(
        label, MDS_CULTURE_COLORS.get(label, CULTURE_COLORS.get(label, "#e7298a"))
    )
