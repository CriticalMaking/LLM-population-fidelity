from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from machine_bias_reproduction.config import FIGURES_ROOT
from machine_bias_reproduction.figures import save_plate
from machine_bias_reproduction.plates import (
    CATEGORICAL_INKS,
    GRID,
    INK,
    MUTED_INK,
    QUALITY_RAMP,
    SEPARATOR,
    SURFACE,
    bar_layout,
    magnitude_steps,
    tinted,
)

from .fidelity import (
    COUNTRY_FAMILY,
    EVERY_CELL,
    FAMILIES,
    FAMILY_LABELS,
    POPULATION,
    SCORE_COLUMNS,
    delta_column,
    levels,
    overall_fidelity,
)
from .fidelity_baseline import HOME_SCOPE, scope_rows
from .matching import HOME
from .palette import MIXTRAL_ARCHIVED, REFERENCE_TONES, arm_hatch, arm_order, model_tone
from .population import MODEL_INDEX, MODES, across_replicates
from .population_plates import (
    LEGEND_HANDLERS,
    MARKER_EDGE,
    MARKER_SIZE,
    NOTE_SIZE,
    PLATE_TEXT,
    SPREAD_BAR,
    VIEWS,
    arm_plot_name,
    compact_handles,
    legend_below,
    legend_height,
    replicated,
    series_plot_name,
    series_style,
    spread_handle,
    spread_of,
    stacked,
)
from .registry import ARMS, BASE_ARM, is_base

FIDELITY_FIGURES = FIGURES_ROOT / "fidelity"

FIDELITY_COLORMAP = "viridis"

SHIFT_COLORMAP = LinearSegmentedColormap.from_list(
    "fidelity_shift", (QUALITY_RAMP[-1], SURFACE, QUALITY_RAMP[0])
)

REFERENCE_DASH = (0, (4, 2))

VALUE_SIZE = 12.0

CELL_VALUE_SIZE = 12.0

ROW_WIDTH = 6.9

ROW_HEIGHT = 0.62

# The component profile prints beside the shift plate at under half a column, so its
# rows carry the series and the variant on one line and sit this close together.
COMPONENT_ROW = 0.28

# Score units either side of 0 and 1 on the component profile, so a mark on the floor
# or the ceiling is drawn whole rather than cut by the axis.
COMPONENT_MARGIN = 0.03

ROW_LABEL_PAD = 10.0

# The stacked component plate: one block of rows per set of cells, a bold header tick
# over each block and a rule between blocks.
BLOCK_LABELS: dict[str, str] = {
    POPULATION: "All retained cells",
    "Germany": "German cells only",
    "Mexico": "Mexican cells only",
    "United States": "United States cells only",
}

BLOCK_GAP = 1.4

BLOCK_RULE_INK = "#b8b8b8"

# Side by side, one panel per set of cells: the blocks run across the page instead of
# down it, so the plate is as tall as its longest block rather than as tall as all of
# them, and each panel carries its own row labels.
BLOCK_PANEL_WIDTH = 5.4

BLOCK_PANEL_PAD = 1.9

Block = tuple[str, pd.DataFrame]

SCATTER_WIDTH = 5.8

SCATTER_HEIGHT = 5.6

# The shift plate prints at half a column next to the component profile, so its box is
# wider than tall and each axis spans its own reach rather than the larger of the two.
SHIFT_HEIGHT = 3.6

# Each corner note on the shift plate names both scores over two lines, so the center
# axis keeps this much of its reach to spare and no marker lands under a note.
SHIFT_CENTER_SPARE = 0.7

SHIFT_STEMS: dict[str, str] = {
    POPULATION: "fig_fidelity_shift",
    HOME_SCOPE: "fig_fidelity_shift_home",
}

SHIFT_XLABELS: dict[str, str] = {
    POPULATION: "Change in PFS against the as-released variant",
    HOME_SCOPE: "Change in PFS against the as-released variant\n(target-country cells)",
}

DELTA_PFS = delta_column("pfs")

DELTA_CENTER = delta_column("score_center")

SPREAD: tuple[str, ...] = (*SCORE_COLUMNS, "e_mean_nemd", DELTA_PFS, DELTA_CENTER)

BAR_SPREAD: dict[str, Any] = {"ecolor": MUTED_INK, "elinewidth": 1.1, "capsize": 2.6}

# The three PFS components, each named as the text names it: the word and the symbol
# the definition gives it, not the formula behind it.
COMPONENTS: tuple[tuple[str, str, str], ...] = (
    ("score_accuracy", "o", r"Accuracy ($S_{\mathrm{acc}}$)"),
    ("score_dispersion", "s", r"Adaptability ($S_{\mathrm{adapt}}$)"),
    ("score_structure", "^", r"Structure ($S_{\mathrm{struct}}$)"),
)

COMPONENT_INKS = magnitude_steps(len(COMPONENTS))

CENTER_LABEL = "Cultural center alignment (1 - C)"

PFS_LABEL = "Population Fidelity Score"

CENTER_LEGEND = r"Center alignment ($S_{\mathrm{center}}$)"

SCORE_XLABEL = "Score (0 to 1, higher is better)"

SUBGROUP_ROW = 0.38

SUBGROUP_WIDTH = 11.5

SUBGROUP_DODGE = 0.19

SUBGROUP_BOX = 0.32

SUBGROUP_MARGIN = 0.02

# One box per score on every row of the subgroup plate: (column, edge ink, fill, legend).
# PFS keeps the solid fill of its diamond and center the hollow fill of its square, so
# the two plates read with one key.
SUBGROUP_BOXES: tuple[tuple[str, str, str, str], ...] = (
    ("pfs", INK, tinted(INK, 0.5), PFS_LABEL),
    ("score_center", MUTED_INK, SURFACE, CENTER_LEGEND),
)

LevelRow = tuple[str | None, str | None, str]

SUBGROUP_MODEL_ROW = 0.42

SUBGROUP_MODEL_WIDTH = 12.5

SUBGROUP_MODEL_MARKER = 7.0

SUBGROUP_MODEL_COLUMNS = 4

# The two scores the per-model subgroup plates put side by side, one panel each, both
# on the full 0 to 1 scale so the panels read against each other.
SUBGROUP_SCORES: tuple[tuple[str, str], ...] = (
    ("pfs", PFS_LABEL),
    ("score_center", CENTER_LABEL),
)

# The box plate of one mode laid along the page: levels across, score up. The main
# text prints it at text width and a few centimetres tall, so the panel is short, the
# text a few points smaller than the other plates', and the level names stand upright
# and centred under their columns, each on one line.
SUBGROUP_WIDE_WIDTH = 11.5

SUBGROUP_WIDE_HEIGHT = 1.4

# Room the constrained layout needs under and over the panel for the level names, the
# family names and the legend, so the panel keeps SUBGROUP_WIDE_HEIGHT.
SUBGROUP_WIDE_MARGINS = 1.0

BOX_PLATE_TEXT: dict[str, Any] = {
    "font.size": 13.0,
    "axes.labelsize": 13.0,
    "xtick.labelsize": 12.0,
    "ytick.labelsize": 12.0,
    "legend.fontsize": 12.0,
}

WIDE_TICK_ROTATION = 60

BOX_TICK_ROTATION = 90

# The level names every subgroup plate writes beside its rows or under its columns: the
# pooled cells and the longest level abbreviated, so no name stands much taller or wider
# than the rest.
POOLED_TICK = "All subpop."

TICK_NAMES: dict[str, str] = {"Divorced or separated": "Divor. / separ."}

BOX_YLABEL = "Score (0 to 1)"

# Score units past 0 and 1 on the short panel, enough for a flier on the floor to be
# drawn whole.
BOX_MARGIN = 0.04

# The per-model and proprietary plates laid along the page: PFS above center, levels
# across, one sub-column per variant or model inside every level.
SUBGROUP_WIDE_PANEL = 2.6

SUBGROUP_WIDE_MARKER = 5.5

# Short y labels for the stacked panels, where the full names would collide.
SUBGROUP_WIDE_YLABELS: dict[str, str] = {
    "pfs": "Population\nFidelity Score",
    "score_center": "Center alignment\n(1 - C)",
}

# Every question of one mode stacked in one figure, the shape the appendix uses: the
# four topic views in order, one row per question for the boxes and one pair of rows
# (PFS above center) per question for the marker plates.
QUESTION_VIEWS: tuple[str, ...] = ("happiness", "politics", "religious", "trust")

QUESTION_STACK_ROW = 2.0

QUESTION_STACK_PAIR_ROW = 1.55

STACK_YLABELS: dict[str, str] = {"pfs": "PFS", "score_center": "Center (1 - C)"}

# The proprietary lineage the archive lets us read under one elicitation mode: the
# original study's GPT-3, published under FA, beside the served GPT-5.6-Terra, evaluated
# under FA alone. GPT-4T was published under NTP only, so it is scored in the archived
# table and kept off this plate; a series drawn under another mode would be starred.
PROPRIETARY_MODE = "fa"

PROPRIETARY_SERVED: tuple[str, ...] = ("terra",)

# Each archived proprietary model drawn, and the mode it is read under.
PROPRIETARY_ARCHIVED: dict[str, str] = {"GPT-3": "fa"}

PROPRIETARY_ORDER: tuple[str, ...] = ("GPT-3", "GPT-5.6-Terra")

PROPRIETARY_INKS: dict[str, str] = {"GPT-3": CATEGORICAL_INKS[8]}

PROPRIETARY_COLUMNS = 2

GROUP_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("score_accuracy", "Accuracy: 1 - mean nEMD"),
    ("score_dispersion", "Adaptability: min(A, 1/A)"),
    ("score_structure", "Structure: max(0, rho) — the usual bottleneck"),
    ("score_center", "Center: 1 - C — reported, never inside PFS"),
)

CENTER_CORNERS = (
    (0.02, 0.97, "left", "top", "Fidelity without a German center"),
    (0.98, 0.97, "right", "top", "Fidelity and center together"),
    (0.02, 0.03, "left", "bottom", "Neither"),
    (0.98, 0.03, "right", "bottom", "Center relocated,\npopulation still flat"),
)

SHIFT_CORNERS = (
    (0.02, 0.97, "left", "top", "Center improved,\nfidelity worsened"),
    (0.98, 0.97, "right", "top", "Center improved,\nfidelity improved"),
    (0.02, 0.03, "left", "bottom", "Center worsened,\nfidelity worsened"),
    (0.98, 0.03, "right", "bottom", "Center worsened,\nfidelity improved"),
)


def _rows(frame: pd.DataFrame) -> Iterator[Any]:
    return iter(frame.itertuples())


def _rank(series: str, model_key: Any, arm: Any) -> tuple[int, int, int, str]:
    if series in REFERENCE_TONES:
        return (1, 0, 0, series)
    return (0, MODEL_INDEX[model_key], 0 if is_base(arm) else 1, series)


def series_order(frame: pd.DataFrame) -> list[str]:
    unique = frame.drop_duplicates("series")
    ranked = [
        (_rank(str(row.series), row.model_key, row.arm), str(row.series)) for row in _rows(unique)
    ]
    return [name for _, name in sorted(ranked)]


def bar_style(series: str, model_key: Any, arm: Any) -> dict[str, Any]:
    if series in REFERENCE_TONES:
        reference = REFERENCE_TONES[series]
        return {"facecolor": reference.fill, "edgecolor": reference.ink}
    shade = model_tone(model_key, MODEL_INDEX[model_key])
    return {
        "facecolor": SURFACE if is_base(arm) else shade.fill,
        "edgecolor": shade.ink,
        "hatch": arm_hatch(arm),
    }


def fill_handles(frame: pd.DataFrame) -> list[Patch]:
    """One patch per variant present, so the legend names every fill the bars use."""
    handles = [Patch(facecolor=SURFACE, edgecolor=INK, label="hollow fill: as released")]
    arms = [arm for arm in arm_order(set(frame["arm"].dropna())) if not is_base(arm)]
    handles += [
        Patch(
            facecolor=GRID,
            edgecolor=INK,
            hatch=arm_hatch(arm),
            label=f"{'hatched' if arm_hatch(arm) else 'solid'} fill: {arm}",
        )
        for arm in arms
    ]
    return handles


def view_frame(
    per_question: pd.DataFrame,
    overall: pd.DataFrame,
    view: str,
    mode: str,
    family: str,
) -> pd.DataFrame:
    question = VIEWS[view]
    table = overall if question is None else per_question
    if table.empty:
        return table
    frame = table if question is None else table[table["question"].eq(question)]
    return frame[frame["mode"].eq(mode) & frame["group"].eq(family)].copy()


def _title(view: str, mode: str, family: str, headline: str) -> str:
    topic = "every topic" if VIEWS[view] is None else view
    scope = "" if family == POPULATION else f"{FAMILY_LABELS[family]}, "
    return f"{headline} — {scope}{topic}, {mode.upper()}"


def _ordered(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame.dropna(subset=[column]).sort_values(column, ascending=True)


def row_figure(count: int, width: float, row: float = ROW_HEIGHT) -> tuple[Figure, Axes]:
    figure = Figure(figsize=(width, row * count + 3.0), layout="constrained")
    axis = figure.subplots(1, 1)
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    return figure, axis


def _row_labels(axis: Axes, frame: pd.DataFrame, one_line: bool = False) -> None:
    """One tick per row, the variant under the series or, on one line, beside it."""
    shape = series_plot_name if one_line else stacked
    axis.set_yticks(range(len(frame)), [shape(series) for series in frame["series"]])
    axis.tick_params(axis="y", pad=ROW_LABEL_PAD)
    axis.set_ylim(-0.6, len(frame) - 0.4)


def _corner_notes(axis: Axes, corners: tuple[tuple[float, float, str, str, str], ...]) -> None:
    for x, y, ha, va, text in corners:
        axis.text(
            x,
            y,
            text,
            transform=axis.transAxes,
            ha=ha,
            va=va,
            fontsize=NOTE_SIZE,
            style="italic",
            color=MUTED_INK,
        )


def _reference_rule(axis: Axes, frame: pd.DataFrame, column: str, orientation: str) -> bool:
    reference = frame[frame["series"].eq(MIXTRAL_ARCHIVED)][column]
    if reference.empty or pd.isna(reference.iat[0]):
        return False
    draw = axis.axvline if orientation == "x" else axis.axhline
    draw(
        float(str(reference.iat[0])),
        color=REFERENCE_TONES[MIXTRAL_ARCHIVED].ink,
        linestyle=REFERENCE_DASH,
        linewidth=1.1,
    )
    return True


def _reference_handle(label: str) -> Line2D:
    return Line2D(
        [],
        [],
        color=REFERENCE_TONES[MIXTRAL_ARCHIVED].ink,
        linestyle=REFERENCE_DASH,
        label=label,
    )


def ranking_plate(frame: pd.DataFrame, mode: str, view: str, destination: Path) -> list[Path]:
    ordered = _ordered(frame, "pfs")
    figure, axis = row_figure(len(ordered), ROW_WIDTH)
    for position, row in enumerate(_rows(ordered)):
        spread = spread_of(row, "pfs")
        axis.barh(
            position,
            float(row.pfs),
            height=0.68,
            linewidth=1.0,
            xerr=spread,
            error_kw=BAR_SPREAD,
            **bar_style(str(row.series), row.model_key, row.arm),
        )
        note = f" ({row.binding_term})" if float(row.pfs) == 0.0 else ""
        axis.text(
            float(row.pfs) + (spread or 0.0) + 0.008,
            position,
            f"{row.pfs:.3f}{note}",
            va="center",
            fontsize=VALUE_SIZE,
            color=MUTED_INK,
        )
    drawn = _reference_rule(axis, ordered, "pfs", "x")
    _row_labels(axis, ordered)
    axis.set_xlim(0.0, min(1.0, max(float(ordered["pfs"].max()) * 1.18, 0.25)))
    axis.set_xlabel(PFS_LABEL)
    # axis.set_title(_title(view, mode, POPULATION, "Population fidelity"))
    handles: list[Any] = list(fill_handles(ordered))
    if drawn:
        handles.append(_reference_handle("reference: Mixtral archived"))
    if replicated(ordered):
        handles.append(spread_handle())
    axis.legend(handles=handles, loc="lower right", handler_map=LEGEND_HANDLERS)
    return save_plate(figure, destination / f"fig_fidelity_ranking_{view}_{mode}")


def component_row(axis: Axes, row: Any, position: float) -> None:
    scores = [float(getattr(row, column)) for column, _, _ in COMPONENTS]
    axis.plot(
        [min(scores), max(scores)],
        [position, position],
        color=GRID,
        linewidth=1.4,
        zorder=1.5,
    )
    for (column, marker, _), shade in zip(COMPONENTS, COMPONENT_INKS, strict=True):
        _spread_bar(axis, float(getattr(row, column)), position, spread_of(row, column), shade)
        axis.plot(
            float(getattr(row, column)),
            position,
            marker=marker,
            markersize=MARKER_SIZE,
            linestyle="none",
            color=shade,
            markeredgecolor=shade,
            zorder=3,
        )
    axis.plot(
        float(row.score_center),
        position,
        marker="s",
        markersize=12,
        linestyle="none",
        markerfacecolor="none",
        markeredgecolor=MUTED_INK,
        markeredgewidth=MARKER_EDGE,
        zorder=2.5,
    )
    _spread_bar(axis, float(row.pfs), position, spread_of(row, "pfs"), INK)
    axis.plot(
        float(row.pfs),
        position,
        marker="D",
        markersize=10,
        linestyle="none",
        color=INK,
        zorder=3.5,
    )


def component_axis(axis: Axes) -> None:
    axis.set_xlim(-COMPONENT_MARGIN, 1.0 + COMPONENT_MARGIN)
    axis.set_xticks(np.linspace(0.0, 1.0, 6))
    axis.set_xlabel(SCORE_XLABEL)


def component_handles(frame: pd.DataFrame) -> list[Line2D]:
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=marker,
            markersize=MARKER_SIZE,
            color=shade,
            label=label,
        )
        for (_, marker, label), shade in zip(COMPONENTS, COMPONENT_INKS, strict=True)
    ]
    handles.append(
        Line2D([], [], linestyle="none", marker="D", markersize=10, color=INK, label=PFS_LABEL)
    )
    handles.append(
        Line2D(
            [],
            [],
            linestyle="none",
            marker="s",
            markersize=12,
            markerfacecolor="none",
            markeredgecolor=MUTED_INK,
            markeredgewidth=MARKER_EDGE,
            label=CENTER_LEGEND,
        )
    )
    if replicated(frame):
        handles.append(spread_handle())
    return handles


def components_plate(frame: pd.DataFrame, mode: str, view: str, destination: Path) -> list[Path]:
    ordered = _ordered(frame, "pfs")
    figure, axis = row_figure(len(ordered), ROW_WIDTH, COMPONENT_ROW)
    for position, row in enumerate(_rows(ordered)):
        component_row(axis, row, position)
    _row_labels(axis, ordered, one_line=True)
    component_axis(axis)
    # axis.set_title(_title(view, mode, POPULATION, "What each PFS is made of"))
    legend_below(figure, component_handles(ordered), 2)
    return save_plate(figure, destination / f"fig_fidelity_components_{view}_{mode}")


def component_blocks_plate(
    blocks: Sequence[Block], label: Callable[[Any], str], stem: Path
) -> list[Path]:
    slots = sum(len(rows) + 1 for _, rows in blocks) + BLOCK_GAP * (len(blocks) - 1)
    figure, axis = row_figure(int(np.ceil(slots)), ROW_WIDTH, COMPONENT_ROW)
    ticks: list[float] = []
    labels: list[str] = []
    headers: list[bool] = []
    position = 0.0
    for index, (header, rows) in enumerate(blocks):
        if index:
            axis.axhline(position - BLOCK_GAP / 2.0, color=BLOCK_RULE_INK, linewidth=0.8)
            position -= BLOCK_GAP
        ticks.append(position)
        labels.append(header)
        headers.append(True)
        for row in _rows(rows):
            position -= 1.0
            component_row(axis, row, position)
            ticks.append(position)
            labels.append(label(row))
            headers.append(False)
    axis.set_yticks(ticks, labels)
    for tick, bold in zip(axis.get_yticklabels(), headers, strict=True):
        if bold:
            tick.set_fontweight("bold")
    axis.tick_params(axis="y", pad=ROW_LABEL_PAD)
    axis.set_ylim(position - 0.6, 0.6)
    component_axis(axis)
    drawn = pd.concat([rows for _, rows in blocks], ignore_index=True)
    legend_below(figure, component_handles(drawn), 2)
    return save_plate(figure, stem)


def condition_label(row: Any) -> str:
    arm = getattr(row, "arm", None)
    if isinstance(arm, str) and arm in ARMS:
        return f"{row.model_label} ({arm_plot_name(arm)})"
    return str(row.series)


def component_panels_plate(
    blocks: Sequence[Block], label: Callable[[Any], str], stem: Path
) -> list[Path]:
    tallest = max(len(rows) for _, rows in blocks)
    drawn = pd.concat([rows for _, rows in blocks], ignore_index=True)
    handles = component_handles(drawn)
    figure = Figure(
        figsize=(
            BLOCK_PANEL_WIDTH * len(blocks),
            COMPONENT_ROW * tallest + legend_height(handles, len(handles)) + BLOCK_PANEL_PAD,
        ),
        layout="constrained",
    )
    panels = np.atleast_1d(figure.subplots(1, len(blocks), sharex=True))
    middle = len(blocks) // 2
    for index, (axis, (header, rows)) in enumerate(zip(panels, blocks, strict=True)):
        axis.grid(axis="x", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        ticks: list[float] = []
        labels: list[str] = []
        head = (tallest - len(rows)) / 2.0
        for offset, row in enumerate(_rows(rows)):
            position = -(head + float(offset))
            component_row(axis, row, position)
            ticks.append(position)
            labels.append(label(row))
        axis.set_yticks(ticks, labels)
        axis.tick_params(axis="y", pad=ROW_LABEL_PAD)
        axis.set_ylim(-tallest + 0.4, 0.6)
        axis.set_title(header, fontweight="bold")
        component_axis(axis)
        axis.set_xticks(np.linspace(0.0, 1.0, 3))
        if index != middle:
            axis.set_xlabel("")
    legend_below(figure, handles, len(handles))
    return save_plate(figure, stem)


def released_rows(frame: pd.DataFrame) -> pd.Series:
    return frame["arm"].isna() | frame["arm"].eq(BASE_ARM)


def home_pairs(inside: pd.DataFrame, arm: str) -> pd.DataFrame:
    tuned = inside[inside["arm"].eq(arm)].dropna(subset=["pfs"])
    released = inside[released_rows(inside) & inside["model_key"].isin(tuned["model_key"])]
    released = released.dropna(subset=["pfs"])
    lead = released.set_index("model_key")["pfs"]
    kept = pd.concat([released, tuned[tuned["model_key"].isin(lead.index)]])
    order = kept.assign(lead=kept["model_key"].map(lead), tuned=kept["arm"].eq(arm))
    ordered = order.sort_values(
        ["lead", "model_key", "tuned"], ascending=[False, True, True], kind="stable"
    )
    return ordered.drop(columns=["lead", "tuned"])


def home_blocks(pooled: pd.DataFrame, countries: pd.DataFrame) -> list[Block]:
    released = pooled[released_rows(pooled)].dropna(subset=["pfs"])
    blocks: list[Block] = [
        (BLOCK_LABELS[POPULATION], released.sort_values("pfs", ascending=False, kind="stable"))
    ]
    for arm, country in HOME.items():
        pairs = home_pairs(countries[countries["level"].eq(country)], arm)
        if not pairs.empty:
            blocks.append((BLOCK_LABELS[country], pairs))
    return blocks if len(blocks) > 1 else []


def home_components_plate(
    per_question: pd.DataFrame,
    overall: pd.DataFrame,
    view: str,
    mode: str,
    destination: Path,
) -> list[Path]:
    pooled = view_frame(per_question, overall, view, mode, POPULATION)
    countries = view_frame(per_question, overall, view, mode, COUNTRY_FAMILY)
    if pooled.empty or countries.empty:
        return []
    blocks = home_blocks(pooled, countries)
    if not blocks:
        return []
    return component_panels_plate(
        blocks,
        condition_label,
        destination / f"fig_fidelity_components_home_{view}_{mode}",
    )


def _scatter_figure(
    handles: Sequence[Line2D], height: float = SCATTER_HEIGHT
) -> tuple[Figure, Axes]:
    figure = Figure(
        figsize=(SCATTER_WIDTH, height + legend_height(handles, 2)), layout="constrained"
    )
    axis = figure.subplots(1, 1)
    axis.grid(color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    legend_below(figure, handles, 2)
    return figure, axis


def center_plate(frame: pd.DataFrame, mode: str, view: str, destination: Path) -> list[Path]:
    drawable = frame.dropna(subset=["pfs", "score_center"])
    handles = [
        *compact_handles(drawable),
        Line2D(
            [],
            [],
            color=MUTED_INK,
            linestyle=(0, (1, 1.2)),
            label="as released to its finetuned variant",
        ),
    ]
    figure, axis = _scatter_figure(handles)
    for model_key, pair in drawable.groupby("model_key", dropna=True):
        if len(pair) < 2:
            continue
        key = str(model_key)
        shade = model_tone(key, MODEL_INDEX[key])
        axis.plot(
            pair["score_center"],
            pair["pfs"],
            color=shade.ink,
            linewidth=1.1,
            linestyle=(0, (1, 1.2)),
            zorder=2,
        )
    for row in _rows(drawable):
        axis.plot(
            float(row.score_center),
            float(row.pfs),
            linestyle="none",
            markersize=MARKER_SIZE,
            markeredgewidth=MARKER_EDGE,
            **series_style(str(row.series), row.model_key, row.arm),
        )
    _corner_notes(axis, CENTER_CORNERS)
    axis.set_xlabel(CENTER_LABEL)
    axis.set_ylabel(PFS_LABEL)
    # axis.set_title(_title(view, mode, POPULATION, "Fidelity against center"))
    return save_plate(figure, destination / f"fig_fidelity_center_{view}_{mode}")


def _spread_bar(axis: Axes, x: float, y: float, spread: float | None, ink: str) -> None:
    if spread is not None:
        axis.errorbar(x, y, xerr=spread, ecolor=ink, **SPREAD_BAR)


def _reach(frame: pd.DataFrame, columns: Sequence[str]) -> float:
    extent = 0.0
    for column in columns:
        values = frame[column].abs()
        if f"{column}_ci" in frame.columns:
            values = values + frame[f"{column}_ci"].fillna(0.0)
        if not values.empty:
            extent = max(extent, float(values.max()))
    return extent or 0.1


def _span(frame: pd.DataFrame, column: str, spare: float = 0.25) -> tuple[float, float]:
    """Symmetric limits around zero with a share of the column's reach to spare."""
    reach = _reach(frame, (column,)) * (1.0 + spare)
    return -reach, reach


def home_shifts(
    paired: pd.DataFrame, paired_overall: pd.DataFrame, view: str, mode: str
) -> pd.DataFrame:
    countries = view_frame(paired, paired_overall, view, mode, COUNTRY_FAMILY)
    if countries.empty:
        return countries
    return scope_rows(countries, HOME_SCOPE)


def shift_plate(
    frame: pd.DataFrame,
    mode: str,
    view: str,
    destination: Path,
    scope: str = POPULATION,
) -> list[Path]:
    drawable = frame.dropna(subset=[DELTA_PFS, DELTA_CENTER])
    handles = [
        *compact_handles(drawable),
        Line2D([], [], linestyle=REFERENCE_DASH, color=MUTED_INK, label="no change"),
    ]
    if replicated(drawable):
        handles.append(spread_handle())
    figure, axis = _scatter_figure(handles, SHIFT_HEIGHT)
    axis.axvline(0.0, linestyle=REFERENCE_DASH, color=MUTED_INK, linewidth=1.1)
    axis.axhline(0.0, linestyle=REFERENCE_DASH, color=MUTED_INK, linewidth=1.1)
    for row in _rows(drawable):
        style = series_style(str(row.series), row.model_key, row.arm)
        x = float(getattr(row, DELTA_PFS))
        y = float(getattr(row, DELTA_CENTER))
        xerr = spread_of(row, DELTA_PFS)
        yerr = spread_of(row, DELTA_CENTER)
        if xerr is not None or yerr is not None:
            axis.errorbar(x, y, xerr=xerr, yerr=yerr, ecolor=style["color"], **SPREAD_BAR)
        axis.plot(
            x, y, linestyle="none", markersize=MARKER_SIZE, markeredgewidth=MARKER_EDGE, **style
        )
    axis.set_xlim(*_span(drawable, DELTA_PFS))
    axis.set_ylim(*_span(drawable, DELTA_CENTER, SHIFT_CENTER_SPARE))
    _corner_notes(axis, SHIFT_CORNERS)
    axis.set_xlabel(SHIFT_XLABELS[scope])
    axis.set_ylabel("Change in center alignment")
    # axis.set_title(_title(view, mode, POPULATION, "What the finetuning bought"))
    return save_plate(figure, destination / f"{SHIFT_STEMS[scope]}_{view}_{mode}")


def cells_plate(frame: pd.DataFrame, mode: str, view: str, destination: Path) -> list[Path]:
    ordered = _ordered(frame, "e_mean_nemd").iloc[::-1]
    figure, axis = row_figure(len(ordered), ROW_WIDTH)
    for position, row in enumerate(_rows(ordered)):
        shade = bar_style(str(row.series), row.model_key, row.arm)
        axis.plot(
            [float(row.e_q10_nemd), float(row.e_q90_nemd)],
            [position, position],
            color=shade["edgecolor"],
            linewidth=1.2,
            zorder=2,
        )
        axis.barh(
            position,
            float(row.e_q75_nemd) - float(row.e_q25_nemd),
            left=float(row.e_q25_nemd),
            height=0.52,
            linewidth=1.0,
            zorder=2.5,
            **shade,
        )
        axis.plot(
            float(row.e_median_nemd),
            position,
            marker="|",
            markersize=13,
            markeredgewidth=1.8,
            color=shade["edgecolor"],
            linestyle="none",
            zorder=3,
        )
        axis.plot(
            float(row.e_mean_nemd),
            position,
            marker="D",
            markersize=7,
            color=INK,
            linestyle="none",
            zorder=3.5,
        )
    drawn = _reference_rule(axis, ordered, "e_mean_nemd", "x")
    _row_labels(axis, ordered)
    axis.set_xlim(left=0.0)
    axis.set_xlabel("Subpopulation nEMD (lower is closer to the WVS)")
    # axis.set_title(_title(view, mode, POPULATION, "Every subpopulation behind the mean"))
    handles: list[Any] = [
        Patch(facecolor=GRID, edgecolor=INK, label="q25 to q75 of the run's cells"),
        Line2D([], [], color=MUTED_INK, lw=1.2, label="q10 to q90"),
        Line2D([], [], linestyle="none", marker="|", markersize=13, color=INK, label="median cell"),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="D",
            markersize=7,
            color=INK,
            label="mean cell, the accuracy term",
        ),
    ]
    if drawn:
        handles.append(_reference_handle("reference: Mixtral archived"))
    legend_below(figure, handles, 2)
    return save_plate(figure, destination / f"fig_fidelity_cells_{view}_{mode}")


def _bold(text: str) -> str:
    escaped = text.replace(" ", r"\ ")
    return rf"$\bf{{{escaped}}}$"


def level_rows(frame: pd.DataFrame) -> list[LevelRow]:
    """The y-order of the subgroup plate.

    The pooled cells come first, then every family in FAMILIES order with its levels in
    LEVEL_ORDER, a blank spacer row between families and the family name in bold, the
    way predictor_rows labels the regression plates.
    """
    rows: list[LevelRow] = []
    if frame["group"].eq(POPULATION).any():
        rows.append((POPULATION, EVERY_CELL, _bold(POOLED_TICK)))
    for family in FAMILIES:
        names = levels(family, frame.loc[frame["group"].eq(family), "level"])
        if not names:
            continue
        if rows:
            rows.append((None, None, ""))
        label = _bold(FAMILY_LABELS[family])
        rows.extend((family, level, f"{label}: {TICK_NAMES.get(level, level)}") for level in names)
    return rows


def subgroup_frame(
    per_question: pd.DataFrame,
    overall: pd.DataFrame,
    view: str,
    mode: str,
) -> pd.DataFrame:
    """Every group and level of one view and mode, served models only.

    The Mixtral references are the reproduction's yardstick, not conditions of the sweep,
    so they stay out of the boxes the way they stay out of the paper's group table.
    """
    question = VIEWS[view]
    table = overall if question is None else per_question
    if table.empty:
        return table
    frame = table if question is None else table[table["question"].eq(question)]
    return frame[frame["mode"].eq(mode) & frame["model_key"].notna()].copy()


def _boxes(
    axis: Axes,
    frame: pd.DataFrame,
    rows: Sequence[LevelRow],
    orientation: Literal["vertical", "horizontal"] = "horizontal",
) -> None:
    offsets = (-SUBGROUP_DODGE, SUBGROUP_DODGE)
    for position, (family, level, _) in enumerate(rows):
        if family is None or level is None:
            continue
        subset = frame[frame["group"].eq(family) & frame["level"].eq(level)]
        for (column, ink, fill, _), offset in zip(SUBGROUP_BOXES, offsets, strict=True):
            values = subset[column].dropna().to_numpy(dtype=np.float64)
            if values.size == 0:
                continue
            axis.boxplot(
                [values],
                orientation=orientation,
                positions=[position + offset],
                widths=SUBGROUP_BOX,
                patch_artist=True,
                manage_ticks=False,
                boxprops={"facecolor": fill, "edgecolor": ink, "linewidth": 1.0},
                medianprops={"color": ink, "linewidth": 1.7},
                whiskerprops={"color": ink, "linewidth": 0.9},
                capprops={"color": ink, "linewidth": 0.9},
                flierprops={
                    "marker": "o",
                    "markersize": 3.4,
                    "markerfacecolor": SURFACE,
                    "markeredgecolor": ink,
                    "markeredgewidth": 0.8,
                },
                zorder=3,
            )


def subgroups_plate(
    per_question: pd.DataFrame,
    overall: pd.DataFrame,
    view: str,
    destination: Path,
) -> list[Path]:
    """PFS beside center alignment for every level of every family, NTP | FA.

    One row per level and two boxes per row, each over the sweep's model conditions: the
    filled box is PFS, the hollow one the center alignment reported beside it and never
    inside it. The rows read as the standardized-coefficient plates do, the pooled cells
    on top and each family in its own block.
    """
    panels = [(mode, subgroup_frame(per_question, overall, view, mode)) for mode in MODES]
    panels = [(mode, frame) for mode, frame in panels if not frame.empty]
    if not panels:
        return []
    rows = level_rows(pd.concat([frame for _, frame in panels], ignore_index=True))
    counts = ", ".join(f"{mode.upper()} {int(frame['series'].nunique())}" for mode, frame in panels)
    handles = _box_handles(counts)
    figure = Figure(
        figsize=(SUBGROUP_WIDTH, SUBGROUP_ROW * len(rows) + legend_height(handles, 2) + 1.0),
        layout="constrained",
    )
    axes = figure.subplots(1, len(panels), sharey=True, squeeze=False)[0]
    for axis, (mode, frame) in zip(axes, panels, strict=True):
        _score_axis(axis)
        _boxes(axis, frame, rows)
        axis.set_title(mode.upper())
    _level_ticks(axes, rows)
    legend_below(figure, handles, 2)
    return save_plate(figure, destination / f"fig_fidelity_subgroups_{view}")


def _box_handles(counts: str) -> list[Any]:
    handles: list[Any] = [
        Patch(facecolor=fill, edgecolor=ink, label=label) for _, ink, fill, label in SUBGROUP_BOXES
    ]
    handles.append(
        Line2D(
            [],
            [],
            linestyle="none",
            label=f"boxes span the model conditions ({counts}); Mixtral references excluded",
        )
    )
    return handles


def _score_axis(axis: Axes) -> None:
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.set_xlim(-SUBGROUP_MARGIN, 1.0 + SUBGROUP_MARGIN)
    axis.set_xlabel(SCORE_XLABEL)


def _level_ticks(axes: Sequence[Axes], rows: Sequence[LevelRow]) -> None:
    axes[0].set_yticks(range(len(rows)), [label for _, _, label in rows])
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    axes[0].tick_params(axis="y", pad=ROW_LABEL_PAD)
    for axis in axes[1:]:
        axis.tick_params(labelleft=False)


def family_blocks(rows: Sequence[LevelRow]) -> list[tuple[str, float, float]]:
    """Every family's block as (family, first position, last position), pooled row aside."""
    blocks: list[tuple[str, float, float]] = []
    for position, (family, level, _) in enumerate(rows):
        if family is None or level is None or family == POPULATION:
            continue
        if blocks and blocks[-1][0] == family:
            blocks[-1] = (family, blocks[-1][1], float(position))
        else:
            blocks.append((family, float(position), float(position)))
    return blocks


def _tick_names(rows: Sequence[LevelRow]) -> list[str]:
    """One name per column: the pooled tick, the abbreviated level, or nothing for a spacer."""
    return [
        POOLED_TICK if family == POPULATION else TICK_NAMES.get(level or "", level or "")
        for family, level, _ in rows
    ]


def _wide_tick_labels(axis: Axes, rows: Sequence[LevelRow]) -> None:
    """Level names as leaning ticks below the axis of a stacked plate."""
    axis.set_xticks(
        range(len(rows)),
        _tick_names(rows),
        rotation=WIDE_TICK_ROTATION,
        ha="right",
        rotation_mode="anchor",
    )
    axis.set_xlim(-0.7, len(rows) - 0.3)


def _wide_blocks(axis: Axes, rows: Sequence[LevelRow], named: bool) -> None:
    """A rule between family blocks and, when asked, the family names above them."""
    for position, (family, _, _) in enumerate(rows):
        if family is None:
            axis.axvline(position, color=GRID, linewidth=0.8, zorder=1)
    if not named:
        return
    for family, first, last in family_blocks(rows):
        axis.text(
            (first + last) / 2,
            1.01,
            FAMILY_LABELS[family],
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontweight="bold",
        )


def _wide_ticks(axis: Axes, rows: Sequence[LevelRow]) -> None:
    """Level names upright below their columns, family names above, a rule between blocks.

    The one-panel plate is the main text's, so each level name stands upright and centred
    under its own column rather than leaning away from the tick as the stacked plates'
    do.
    """
    axis.set_xticks(
        range(len(rows)), _tick_names(rows), rotation=BOX_TICK_ROTATION, ha="center", va="top"
    )
    axis.set_xlim(-0.7, len(rows) - 0.3)
    _wide_blocks(axis, rows, named=True)


def _stack(
    rows: Sequence[LevelRow],
    count: int,
    height: float,
    handles: Sequence[Any],
    columns: int,
) -> tuple[Figure, list[Axes]]:
    """`count` panels sharing the level axis, each on the 0 to 1 scale, legend below.

    The family names sit above the top panel, the level names below the bottom one,
    and every panel carries the rules between family blocks.
    """
    figure = Figure(
        figsize=(SUBGROUP_WIDE_WIDTH, count * height + legend_height(handles, columns) + 1.6),
        layout="constrained",
    )
    axes = list(figure.subplots(count, 1, sharex=True, squeeze=False)[:, 0])
    for axis in axes:
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.set_ylim(-SUBGROUP_MARGIN, 1.0 + SUBGROUP_MARGIN)
        _wide_blocks(axis, rows, named=axis is axes[0])
    _wide_tick_labels(axes[-1], rows)
    legend_below(figure, handles, columns)
    return figure, axes


def _wide_panels(
    rows: Sequence[LevelRow],
    handles: Sequence[Any],
    columns: int,
) -> tuple[Figure, list[Axes]]:
    """Two panels sharing the level axis, PFS above center, each on the 0 to 1 scale."""
    figure, axes = _stack(rows, len(SUBGROUP_SCORES), SUBGROUP_WIDE_PANEL, handles, columns)
    for axis, (column, _) in zip(axes, SUBGROUP_SCORES, strict=True):
        axis.set_ylabel(SUBGROUP_WIDE_YLABELS[column])
    return figure, axes


def _question_panels(
    per_question: pd.DataFrame,
    overall: pd.DataFrame,
    mode: str,
) -> list[tuple[str, pd.DataFrame]]:
    """(question label, served frame) for every topic view of one mode that has rows."""
    panels: list[tuple[str, pd.DataFrame]] = []
    for view in QUESTION_VIEWS:
        frame = subgroup_frame(per_question, overall, view, mode)
        if not frame.empty:
            panels.append((str(frame["question_label"].iat[0]), frame))
    return panels


def subgroups_questions_plate(
    per_question: pd.DataFrame,
    overall: pd.DataFrame,
    mode: str,
    destination: Path,
) -> list[Path]:
    """The box plates of every question under one mode, stacked in one figure.

    One row per question with the two boxes per level of `subgroups_wide_plate`, the
    level axis and the legend shared, so the appendix shows the questions in the
    space of one plate.
    """
    panels = _question_panels(per_question, overall, mode)
    if not panels:
        return []
    joined = pd.concat([frame for _, frame in panels], ignore_index=True)
    rows = level_rows(joined)
    handles = _box_handles(f"{mode.upper()} {int(joined['series'].nunique())}")
    figure, axes = _stack(rows, len(panels), QUESTION_STACK_ROW, handles, 3)
    for axis, (label, frame) in zip(axes, panels, strict=True):
        _boxes(axis, frame, rows, "vertical")
        axis.set_ylabel(label)
    figure.supylabel(SCORE_XLABEL)
    return save_plate(figure, destination / f"fig_fidelity_subgroups_questions_{mode}")


def _question_pairs(
    axes: Sequence[Axes],
    panels: Sequence[tuple[str, pd.DataFrame]],
    rows: Sequence[LevelRow],
    place: Callable[[Any], float],
    style: Callable[[Any], dict[str, Any]],
    size: float,
) -> None:
    """PFS above center for every question, each pair of panels labelled by question."""
    scores = len(SUBGROUP_SCORES)
    for index, (label, frame) in enumerate(panels):
        pair = axes[scores * index : scores * (index + 1)]
        for axis, (column, _) in zip(pair, SUBGROUP_SCORES, strict=True):
            _level_markers(axis, frame, rows, column, place, style, vertical=True, size=size)
            axis.set_ylabel(f"{label}\n{STACK_YLABELS[column]}")


def subgroup_models_questions_plate(
    per_question: pd.DataFrame,
    overall: pd.DataFrame,
    mode: str,
    destination: Path,
) -> list[Path]:
    """The per-model plates of every question under one mode, stacked in one figure."""
    panels = _question_panels(per_question, overall, mode)
    if not panels:
        return []
    joined = pd.concat([frame for _, frame in panels], ignore_index=True)
    rows = level_rows(joined)
    offsets = variant_offsets(joined)
    handles: list[Any] = [*compact_handles(joined), _mode_note(mode, joined)]
    figure, axes = _stack(
        rows,
        len(SUBGROUP_SCORES) * len(panels),
        QUESTION_STACK_PAIR_ROW,
        handles,
        SUBGROUP_MODEL_COLUMNS,
    )
    _question_pairs(
        axes,
        panels,
        rows,
        lambda row: offsets.get(str(row.arm), 0.0),
        lambda row: series_style(str(row.series), row.model_key, row.arm),
        SUBGROUP_WIDE_MARKER,
    )
    return save_plate(figure, destination / f"fig_fidelity_subgroups_models_questions_{mode}")


def subgroups_wide_plate(
    per_question: pd.DataFrame,
    overall: pd.DataFrame,
    view: str,
    mode: str,
    destination: Path,
) -> list[Path]:
    """The box plate of one mode laid along the page: levels across, score up.

    The same two boxes per level as `subgroups_plate`, the family names above their
    blocks and the level names as ticks below, so one mode fits a text-width figure.
    """
    frame = subgroup_frame(per_question, overall, view, mode)
    if frame.empty:
        return []
    rows = level_rows(frame)
    handles = _box_handles(f"{mode.upper()} {int(frame['series'].nunique())}")
    with rc_context(cast(Any, BOX_PLATE_TEXT)):
        figure = Figure(
            figsize=(
                SUBGROUP_WIDE_WIDTH,
                SUBGROUP_WIDE_HEIGHT + legend_height(handles, 3) + SUBGROUP_WIDE_MARGINS,
            ),
            layout="constrained",
        )
        axis = figure.subplots(1, 1)
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        _boxes(axis, frame, rows, "vertical")
        _wide_ticks(axis, rows)
        axis.set_ylim(-BOX_MARGIN, 1.0 + BOX_MARGIN)
        axis.set_ylabel(BOX_YLABEL)
        legend_below(figure, handles, 3)
        return save_plate(figure, destination / f"fig_fidelity_subgroups_wide_{view}_{mode}")


def variant_offsets(frame: pd.DataFrame) -> dict[str, float]:
    """One sub-row per variant, as released on top.

    A row's markers dodge by variant rather than by model, so the fill and the vertical
    position say the same thing and the six or seven models of one variant share a line.
    """
    arms = arm_order(set(frame["arm"].dropna()))
    _, offsets = bar_layout(len(arms))
    return {arm: float(offset) for arm, offset in zip(arms, offsets, strict=True)}


def _level_markers(
    axis: Axes,
    frame: pd.DataFrame,
    rows: Sequence[LevelRow],
    column: str,
    place: Callable[[Any], float],
    style: Callable[[Any], dict[str, Any]],
    vertical: bool = False,
    size: float = SUBGROUP_MODEL_MARKER,
) -> None:
    """One marker per row of the frame on the level it belongs to, dodged by `place`.

    Levels run down the plate by default and across it when `vertical`, with the score
    then read on the y axis.
    """
    for position, (family, level, _) in enumerate(rows):
        if family is None or level is None:
            continue
        subset = frame[frame["group"].eq(family) & frame["level"].eq(level)]
        for row in _rows(subset.dropna(subset=[column])):
            score = float(getattr(row, column))
            slot = position + place(row)
            axis.plot(
                slot if vertical else score,
                score if vertical else slot,
                linestyle="none",
                markersize=size,
                markeredgewidth=1.1,
                **style(row),
            )


def subgroup_models_plate(
    per_question: pd.DataFrame,
    overall: pd.DataFrame,
    view: str,
    mode: str,
    destination: Path,
) -> list[Path]:
    """Every model condition on every level, PFS | center, one mode.

    The box plate summarises the conditions; this one names them. The rows are the
    same, colour and marker carry the model, and the fill and the sub-row carry the
    variant, so the spread of a row is read model by model.
    """
    frame = subgroup_frame(per_question, overall, view, mode)
    if frame.empty:
        return []
    rows = level_rows(frame)
    offsets = variant_offsets(frame)
    handles = compact_handles(frame)
    figure = Figure(
        figsize=(
            SUBGROUP_MODEL_WIDTH,
            SUBGROUP_MODEL_ROW * len(rows) + legend_height(handles, SUBGROUP_MODEL_COLUMNS) + 1.0,
        ),
        layout="constrained",
    )
    axes = figure.subplots(1, len(SUBGROUP_SCORES), sharey=True, squeeze=False)[0]
    for axis, (column, title) in zip(axes, SUBGROUP_SCORES, strict=True):
        _score_axis(axis)
        _level_markers(
            axis,
            frame,
            rows,
            column,
            lambda row: offsets.get(str(row.arm), 0.0),
            lambda row: series_style(str(row.series), row.model_key, row.arm),
        )
        axis.set_title(f"{title}, {mode.upper()}")
    _level_ticks(axes, rows)
    legend_below(figure, handles, SUBGROUP_MODEL_COLUMNS)
    return save_plate(figure, destination / f"fig_fidelity_subgroups_models_{view}_{mode}")


def _mode_note(mode: str, frame: pd.DataFrame) -> Line2D:
    conditions = int(frame["series"].nunique())
    return Line2D(
        [],
        [],
        linestyle="none",
        label=f"{mode.upper()} elicitation, {conditions} model conditions",
    )


def subgroup_models_wide_plate(
    per_question: pd.DataFrame,
    overall: pd.DataFrame,
    view: str,
    mode: str,
    destination: Path,
) -> list[Path]:
    """The per-model plate of one mode laid along the page: levels across, PFS above center.

    Every condition keeps its marker and fill, and the variants dodge into sub-columns
    inside each level as they dodge into sub-rows on the vertical plate.
    """
    frame = subgroup_frame(per_question, overall, view, mode)
    if frame.empty:
        return []
    rows = level_rows(frame)
    offsets = variant_offsets(frame)
    handles: list[Any] = [*compact_handles(frame), _mode_note(mode, frame)]
    figure, axes = _wide_panels(rows, handles, SUBGROUP_MODEL_COLUMNS)
    for axis, (column, _) in zip(axes, SUBGROUP_SCORES, strict=True):
        _level_markers(
            axis,
            frame,
            rows,
            column,
            lambda row: offsets.get(str(row.arm), 0.0),
            lambda row: series_style(str(row.series), row.model_key, row.arm),
            vertical=True,
            size=SUBGROUP_WIDE_MARKER,
        )
    return save_plate(figure, destination / f"fig_fidelity_subgroups_models_wide_{view}_{mode}")


def proprietary_frame(
    per_question: pd.DataFrame,
    overall: pd.DataFrame,
    archived: pd.DataFrame,
    archived_overall: pd.DataFrame,
    view: str,
) -> pd.DataFrame:
    """The served proprietary conditions under FA beside each archived proprietary series
    under the one mode the archive holds it in."""
    question = VIEWS[view]
    sweep = overall if question is None else per_question
    published = archived_overall if question is None else archived
    served = sweep[sweep["model_key"].isin(PROPRIETARY_SERVED) & sweep["mode"].eq(PROPRIETARY_MODE)]
    held = published["model_label"].map(PROPRIETARY_ARCHIVED)
    parts = [served, published[published["mode"].eq(held)]]
    frame = pd.concat([part for part in parts if not part.empty], ignore_index=True)
    if question is not None and not frame.empty:
        frame = frame[frame["question"].eq(question)]
    return frame.copy()


def _proprietary_label(row: Any) -> str:
    mode = str(row.mode)
    flag = "" if mode == PROPRIETARY_MODE else "*"
    return f"{row.series}, {mode.upper()}{flag}"


def _proprietary_style(row: Any) -> dict[str, Any]:
    """A served model keeps the style it wears everywhere; an archived one a solid disc."""
    if isinstance(row.model_key, str):
        return series_style(str(row.series), row.model_key, row.arm)
    ink = PROPRIETARY_INKS[str(row.model_label)]
    return {
        "marker": "o",
        "color": ink,
        "markerfacecolor": ink,
        "markeredgecolor": ink,
        "zorder": 3.1,
    }


def _proprietary_handles(frame: pd.DataFrame) -> list[Line2D]:
    handles: list[Line2D] = []
    for label in PROPRIETARY_ORDER:
        rows = frame[frame["model_label"].eq(label)]
        if rows.empty:
            continue
        row = next(_rows(rows))
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                markersize=SUBGROUP_MODEL_MARKER,
                markeredgewidth=MARKER_EDGE,
                label=_proprietary_label(row),
                **_proprietary_style(row),
            )
        )
    return handles


def proprietary_plate(
    per_question: pd.DataFrame,
    overall: pd.DataFrame,
    archived: pd.DataFrame,
    archived_overall: pd.DataFrame,
    view: str,
    destination: Path,
) -> list[Path]:
    """The proprietary lineage on every level, PFS | center, under one elicitation mode.

    Each model keeps its own sub-row, the archived series above the served one, so the
    original study's GPT-3 and the served GPT-5.6-Terra are read level by level under
    the mode both were evaluated in.
    """
    frame = proprietary_frame(per_question, overall, archived, archived_overall, view)
    if frame.empty:
        return []
    rows = level_rows(frame)
    labels = [label for label in PROPRIETARY_ORDER if frame["model_label"].eq(label).any()]
    _, dodge = bar_layout(len(labels))
    offsets = {label: float(offset) for label, offset in zip(labels, dodge, strict=True)}
    handles = _proprietary_handles(frame)
    figure = Figure(
        figsize=(
            SUBGROUP_MODEL_WIDTH,
            SUBGROUP_MODEL_ROW * len(rows) + legend_height(handles, PROPRIETARY_COLUMNS) + 1.0,
        ),
        layout="constrained",
    )
    axes = figure.subplots(1, len(SUBGROUP_SCORES), sharey=True, squeeze=False)[0]
    for axis, (column, title) in zip(axes, SUBGROUP_SCORES, strict=True):
        _score_axis(axis)
        _level_markers(
            axis,
            frame,
            rows,
            column,
            lambda row: offsets[str(row.model_label)],
            _proprietary_style,
        )
        axis.set_title(f"{title}, {PROPRIETARY_MODE.upper()}")
    _level_ticks(axes, rows)
    legend_below(figure, handles, PROPRIETARY_COLUMNS)
    return save_plate(figure, destination / f"fig_fidelity_subgroups_proprietary_{view}")


def proprietary_wide_plate(
    per_question: pd.DataFrame,
    overall: pd.DataFrame,
    archived: pd.DataFrame,
    archived_overall: pd.DataFrame,
    view: str,
    destination: Path,
) -> list[Path]:
    """The proprietary lineage laid along the page: levels across, PFS above center."""
    frame = proprietary_frame(per_question, overall, archived, archived_overall, view)
    if frame.empty:
        return []
    rows = level_rows(frame)
    labels = [label for label in PROPRIETARY_ORDER if frame["model_label"].eq(label).any()]
    _, dodge = bar_layout(len(labels))
    offsets = {label: float(offset) for label, offset in zip(labels, dodge, strict=True)}
    handles: list[Any] = [*_proprietary_handles(frame), _mode_note(PROPRIETARY_MODE, frame)]
    figure, axes = _wide_panels(rows, handles, PROPRIETARY_COLUMNS + 1)
    for axis, (column, _) in zip(axes, SUBGROUP_SCORES, strict=True):
        _level_markers(
            axis,
            frame,
            rows,
            column,
            lambda row: offsets[str(row.model_label)],
            _proprietary_style,
            vertical=True,
        )
    return save_plate(figure, destination / f"fig_fidelity_subgroups_proprietary_wide_{view}")


def proprietary_questions_plate(
    per_question: pd.DataFrame,
    overall: pd.DataFrame,
    archived: pd.DataFrame,
    archived_overall: pd.DataFrame,
    destination: Path,
) -> list[Path]:
    """The proprietary lineage of every question, stacked in one figure."""
    panels: list[tuple[str, pd.DataFrame]] = []
    for view in QUESTION_VIEWS:
        frame = proprietary_frame(per_question, overall, archived, archived_overall, view)
        if not frame.empty:
            panels.append((str(frame["question_label"].iat[0]), frame))
    if not panels:
        return []
    joined = pd.concat([frame for _, frame in panels], ignore_index=True)
    rows = level_rows(joined)
    labels = [label for label in PROPRIETARY_ORDER if joined["model_label"].eq(label).any()]
    _, dodge = bar_layout(len(labels))
    offsets = {label: float(offset) for label, offset in zip(labels, dodge, strict=True)}
    handles: list[Any] = [*_proprietary_handles(joined), _mode_note(PROPRIETARY_MODE, joined)]
    figure, axes = _stack(
        rows,
        len(SUBGROUP_SCORES) * len(panels),
        QUESTION_STACK_PAIR_ROW,
        handles,
        PROPRIETARY_COLUMNS + 1,
    )
    _question_pairs(
        axes,
        panels,
        rows,
        lambda row: offsets[str(row.model_label)],
        _proprietary_style,
        SUBGROUP_MODEL_MARKER,
    )
    return save_plate(figure, destination / "fig_fidelity_subgroups_proprietary_questions")


def _matrix(frame: pd.DataFrame, family: str, column: str) -> pd.DataFrame:
    columns = levels(family, frame["level"])
    rows = series_order(frame)
    table = frame.pivot_table(index="series", columns="level", values=column, aggfunc="first")
    return table.reindex(index=rows, columns=columns)


def _annotate(axis: Axes, matrix: pd.DataFrame, light_below: float | None) -> None:
    values = matrix.to_numpy(dtype=np.float64)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if not np.isfinite(value):
                continue
            pale = light_below is not None and value < light_below
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=CELL_VALUE_SIZE,
                color=SEPARATOR if pale else INK,
            )


def _matrix_ticks(axis: Axes, matrix: pd.DataFrame) -> None:
    names = [TICK_NAMES.get(str(level), str(level)) for level in matrix.columns]
    axis.set_xticks(range(matrix.shape[1]), names, rotation=30, ha="right")
    axis.set_yticks(range(matrix.shape[0]), [stacked(series) for series in matrix.index])
    axis.tick_params(axis="y", pad=ROW_LABEL_PAD)
    axis.grid(visible=False)


def _heatmap(
    matrix: pd.DataFrame,
    title: str,
    bar_label: str,
    stem: Path,
    *,
    shift: bool,
) -> list[Path]:
    figure = Figure(
        figsize=(1.1 * matrix.shape[1] + 3.8, 0.5 * matrix.shape[0] + 2.8),
        layout="constrained",
    )
    axis = figure.subplots(1, 1)
    values = matrix.to_numpy(dtype=np.float64)
    if shift:
        reach = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
        reach = reach if reach > 0 else 1.0
        image = axis.imshow(
            values,
            cmap=SHIFT_COLORMAP,
            norm=TwoSlopeNorm(vmin=-reach, vcenter=0.0, vmax=reach),
            aspect="auto",
        )
        _annotate(axis, matrix, None)
    else:
        image = axis.imshow(values, cmap=FIDELITY_COLORMAP, vmin=0.0, vmax=1.0, aspect="auto")
        _annotate(axis, matrix, 0.5)
    _matrix_ticks(axis, matrix)
    # axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02, label=bar_label)
    return save_plate(figure, stem)


def group_plate(
    frame: pd.DataFrame,
    family: str,
    mode: str,
    view: str,
    destination: Path,
) -> list[Path]:
    matrix = _matrix(frame, family, "pfs")
    return _heatmap(
        matrix,
        _title(view, mode, family, "Population fidelity"),
        PFS_LABEL,
        destination / f"fig_fidelity_{family}_{view}_{mode}",
        shift=False,
    )


def group_components_plate(
    frame: pd.DataFrame,
    family: str,
    mode: str,
    view: str,
    destination: Path,
) -> list[Path]:
    shape = _matrix(frame, family, "pfs").shape
    figure = Figure(
        figsize=(1.6 * shape[1] + 4.4, 0.8 * shape[0] + 3.2),
        layout="constrained",
    )
    axes = figure.subplots(2, 2, sharey=True, squeeze=False).ravel()
    images: list[Any] = []
    for axis, (column, label) in zip(axes, GROUP_COMPONENTS, strict=True):
        matrix = _matrix(frame, family, column)
        images.append(
            axis.imshow(
                matrix.to_numpy(dtype=np.float64),
                cmap=FIDELITY_COLORMAP,
                vmin=0.0,
                vmax=1.0,
                aspect="auto",
            )
        )
        _annotate(axis, matrix, 0.5)
        _matrix_ticks(axis, matrix)
        axis.set_title(label, fontsize=15)
    # figure.suptitle(_title(view, mode, family, "What each PFS is made of"))
    figure.colorbar(
        images[0],
        ax=axes.tolist(),
        fraction=0.02,
        pad=0.02,
        label=SCORE_XLABEL,
    )
    return save_plate(figure, destination / f"fig_fidelity_{family}_components_{view}_{mode}")


def group_shift_plate(
    frame: pd.DataFrame,
    family: str,
    mode: str,
    view: str,
    destination: Path,
) -> list[Path]:
    matrix = _matrix(frame.dropna(subset=[DELTA_PFS]), family, DELTA_PFS)
    return _heatmap(
        matrix,
        _title(view, mode, family, "Finetuning change in fidelity"),
        "Change in PFS against the as-released variant",
        destination / f"fig_fidelity_{family}_shift_{view}_{mode}",
        shift=True,
    )


def fidelity_plates(
    groups: pd.DataFrame,
    overall: pd.DataFrame,
    paired: pd.DataFrame,
    paired_overall: pd.DataFrame,
    destination: Path = FIDELITY_FIGURES,
    archived: pd.DataFrame | None = None,
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    groups, overall, paired, paired_overall = (
        across_replicates(frame, spread=SPREAD)
        for frame in (groups, overall, paired, paired_overall)
    )
    published: tuple[pd.DataFrame, pd.DataFrame] | None = None
    if archived is not None and not archived.empty:
        published = (
            across_replicates(archived, spread=SPREAD),
            across_replicates(overall_fidelity(archived), spread=SPREAD),
        )
    produced: list[Path] = []
    with rc_context(cast(Any, PLATE_TEXT)):
        for mode in MODES:
            for view in VIEWS:
                pooled = view_frame(groups, overall, view, mode, POPULATION)
                if pooled.empty:
                    continue
                produced += ranking_plate(pooled, mode, view, destination)
                produced += components_plate(pooled, mode, view, destination)
                produced += home_components_plate(groups, overall, view, mode, destination)
                produced += center_plate(pooled, mode, view, destination)
                produced += cells_plate(pooled, mode, view, destination)
                produced += subgroup_models_plate(groups, overall, view, mode, destination)
                produced += subgroup_models_wide_plate(groups, overall, view, mode, destination)
                produced += subgroups_wide_plate(groups, overall, view, mode, destination)
                if view == "all":
                    produced += subgroups_questions_plate(groups, overall, mode, destination)
                    produced += subgroup_models_questions_plate(groups, overall, mode, destination)
                shifts = view_frame(paired, paired_overall, view, mode, POPULATION)
                if not shifts.empty:
                    produced += shift_plate(shifts, mode, view, destination)
                homed = home_shifts(paired, paired_overall, view, mode)
                if not homed.empty:
                    produced += shift_plate(homed, mode, view, destination, HOME_SCOPE)
                for family in FAMILIES:
                    frame = view_frame(groups, overall, view, mode, family)
                    if frame.empty:
                        continue
                    folder = destination / family
                    folder.mkdir(parents=True, exist_ok=True)
                    produced += group_plate(frame, family, mode, view, folder)
                    produced += group_components_plate(frame, family, mode, view, folder)
                    shifts = view_frame(paired, paired_overall, view, mode, family)
                    if not shifts.empty:
                        produced += group_shift_plate(shifts, family, mode, view, folder)
        for view in VIEWS:
            produced += subgroups_plate(groups, overall, view, destination)
            if published is not None:
                produced += proprietary_plate(groups, overall, *published, view, destination)
                produced += proprietary_wide_plate(groups, overall, *published, view, destination)
        if published is not None:
            produced += proprietary_questions_plate(groups, overall, *published, destination)
    return produced
