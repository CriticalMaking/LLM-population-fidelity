from __future__ import annotations

from typing import Any, NamedTuple, cast

import matplotlib
import numpy as np
from matplotlib.colors import Colormap, to_hex

matplotlib.use("Agg")

SURFACE = "#ffffff"
INK = "#1a1a1a"
MUTED_INK = "#5f5f5f"
AXIS_INK = "#8a8a8a"
GRID = "#dcdcdc"
SEPARATOR = "#ffffff"

SERIF_STACK: tuple[str, ...] = ("STIXGeneral", "Nimbus Roman", "DejaVu Serif")

PLATE_RC: dict[str, Any] = {
    "figure.facecolor": SURFACE,
    "figure.dpi": 150,
    "savefig.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "font.family": "serif",
    "font.serif": list(SERIF_STACK),
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.labelcolor": INK,
    "axes.edgecolor": AXIS_INK,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "text.color": INK,
    "xtick.color": MUTED_INK,
    "ytick.color": MUTED_INK,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.6,
    "legend.columnspacing": 1.2,
    "legend.labelcolor": INK,
    "lines.linewidth": 1.4,
    "lines.markersize": 5.0,
    "patch.linewidth": 0.6,
    "patch.edgecolor": SEPARATOR,
}

matplotlib.rcParams.update(cast(Any, PLATE_RC))

PASTEL_TINT = 0.62


def tinted(ink: str, amount: float = PASTEL_TINT) -> str:
    channels = (int(ink[position : position + 2], 16) for position in (1, 3, 5))
    return "#" + "".join(f"{round(value + (255 - value) * amount):02x}" for value in channels)


class Tone(NamedTuple):
    ink: str
    fill: str


def tone(ink: str, amount: float = PASTEL_TINT) -> Tone:
    return Tone(ink, tinted(ink, amount))


CATEGORICAL_INKS: tuple[str, ...] = (
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
    "#1f78b4",
)

CATEGORICAL_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "<", ">")

CATEGORICAL_LINESTYLES: tuple[Any, ...] = (
    "solid",
    (0, (5, 2)),
    (0, (1, 1.6)),
    (0, (6, 2, 1, 2)),
    (0, (3, 1.4)),
    (0, (7, 2, 1, 2, 1, 2)),
    (0, (2, 1.2)),
    (0, (8, 3)),
    (0, (4, 1.6, 1, 1.6)),
)

REFERENCE_TONE = tone("#000000")
RERUN_TONE = tone("#c8c8c8")
UNTUNED_TONE = tone("#111111")
SURVEY_TONE = Tone("#8a8a2a", "#fde725")

QUALITY_RAMP: tuple[str, ...] = ("#1b9e77", "#66c2a5", "#ffd92f", "#fc8d62", "#d73027")

STATUS_TONES: dict[str, Tone] = {
    "valid": tone("#1b9e77"),
    "invalid": tone("#fc8d62"),
    "failed": tone("#d73027"),
}

MAGNITUDE_COLORMAP = "viridis_r"

MAGNITUDE_SPAN = (0.15, 0.85)

MODEL_TONE = tone("#440154")


def magnitude_colormap() -> Colormap:
    return matplotlib.colormaps[MAGNITUDE_COLORMAP]


def magnitude_steps(count: int, span: tuple[float, float] = MAGNITUDE_SPAN) -> list[str]:
    ramp = matplotlib.colormaps["viridis"]
    low, high = span
    positions = np.linspace(low, high, count) if count > 1 else np.array([low])
    return [to_hex(ramp(float(position))) for position in positions]


def category_colors(count: int) -> list[str]:
    ramp = matplotlib.colormaps["tab10"]
    positions = np.linspace(0, 1, max(count, 1))
    return [to_hex(ramp(float(position))) for position in positions]


METHOD_TONES: dict[str, Tone] = {
    "Linear": tone("#000000"),
    "Random": tone("#808080"),
    "NTP": tone("#d95f02"),
    "FA": tone("#7570b3"),
}

METHOD_LINESTYLES: dict[str, Any] = {
    "Linear": "solid",
    "Random": (0, (1, 1.6)),
    "NTP": (0, (6, 3)),
    "FA": (0, (7, 2, 1, 2)),
}

METHOD_MARKERS: dict[str, str] = {"NTP": "o", "FA": "s"}


def slot_tone(index: int) -> Tone:
    return tone(CATEGORICAL_INKS[index % len(CATEGORICAL_INKS)])


def slot_marker(index: int) -> str:
    return CATEGORICAL_MARKERS[index % len(CATEGORICAL_MARKERS)]


def slot_linestyle(index: int) -> Any:
    return CATEGORICAL_LINESTYLES[index % len(CATEGORICAL_LINESTYLES)]


def bar_layout(count: int) -> tuple[float, np.ndarray]:
    width = 0.8 / max(count, 1)
    offsets = (np.arange(count) - (count - 1) / 2) * width
    return width, offsets
