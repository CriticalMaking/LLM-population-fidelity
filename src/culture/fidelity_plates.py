from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, cast

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
    GRID,
    INK,
    MUTED_INK,
    QUALITY_RAMP,
    SEPARATOR,
    SURFACE,
    magnitude_steps,
)

from .fidelity import (
    FAMILIES,
    FAMILY_LABELS,
    POPULATION,
    SCORE_COLUMNS,
    delta_column,
    levels,
)
from .palette import MIXTRAL_ARCHIVED, REFERENCE_TONES, arm_hatch, arm_order, model_tone
from .population import MODEL_INDEX, MODES, across_replicates
from .population_plates import (
    MARKER_EDGE,
    MARKER_SIZE,
    NOTE_SIZE,
    PLATE_TEXT,
    SPREAD_BAR,
    VIEWS,
    compact_handles,
    legend_below,
    legend_height,
    replicated,
    series_style,
    spread_handle,
    spread_of,
    stacked,
)
from .registry import is_base

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

ROW_LABEL_PAD = 10.0

SCATTER_WIDTH = 5.8

DELTA_PFS = delta_column("pfs")

DELTA_CENTER = delta_column("score_center")

SPREAD: tuple[str, ...] = (*SCORE_COLUMNS, "e_mean_nemd", DELTA_PFS, DELTA_CENTER)

BAR_SPREAD: dict[str, Any] = {"ecolor": MUTED_INK, "elinewidth": 1.1, "capsize": 2.6}

COMPONENTS: tuple[tuple[str, str, str], ...] = (
    ("score_accuracy", "o", "Accuracy: 1 - mean nEMD"),
    ("score_dispersion", "s", "Adaptability: min(A, 1/A)"),
    ("score_structure", "^", "Structure: max(0, rho)"),
)

COMPONENT_INKS = magnitude_steps(len(COMPONENTS))

CENTER_LABEL = "Cultural center alignment (1 - C)"

PFS_LABEL = "Population Fidelity Score"

CENTER_LEGEND = "Center: 1 - C (not in PFS)"

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
    (0.02, 0.97, "left", "top", "Center gained,\nfidelity lost"),
    (0.98, 0.97, "right", "top", "Both improved"),
    (0.02, 0.03, "left", "bottom", "Both worse"),
    (0.98, 0.03, "right", "bottom", "Fidelity gained,\ncenter drifted"),
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


def _row_figure(count: int, width: float) -> tuple[Figure, Axes]:
    figure = Figure(figsize=(width, ROW_HEIGHT * count + 3.0), layout="constrained")
    axis = figure.subplots(1, 1)
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    return figure, axis


def _row_labels(axis: Axes, frame: pd.DataFrame) -> None:
    axis.set_yticks(range(len(frame)), [stacked(series) for series in frame["series"]])
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
    figure, axis = _row_figure(len(ordered), ROW_WIDTH)
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
    axis.legend(handles=handles, loc="lower right")
    return save_plate(figure, destination / f"fig_fidelity_ranking_{view}_{mode}")


def components_plate(frame: pd.DataFrame, mode: str, view: str, destination: Path) -> list[Path]:
    ordered = _ordered(frame, "pfs")
    figure, axis = _row_figure(len(ordered), ROW_WIDTH)
    for position, row in enumerate(_rows(ordered)):
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
    _row_labels(axis, ordered)
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Score (0 to 1, higher is better)")
    # axis.set_title(_title(view, mode, POPULATION, "What each PFS is made of"))
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
    if replicated(ordered):
        handles.append(spread_handle())
    legend_below(figure, handles, 2)
    return save_plate(figure, destination / f"fig_fidelity_components_{view}_{mode}")


def _scatter_figure(handles: Sequence[Line2D]) -> tuple[Figure, Axes]:
    figure = Figure(figsize=(SCATTER_WIDTH, 5.6 + legend_height(handles, 2)), layout="constrained")
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
        if f"{column}_sd" in frame.columns:
            values = values + frame[f"{column}_sd"].fillna(0.0)
        if not values.empty:
            extent = max(extent, float(values.max()))
    return extent or 0.1


def shift_plate(frame: pd.DataFrame, mode: str, view: str, destination: Path) -> list[Path]:
    drawable = frame.dropna(subset=[DELTA_PFS, DELTA_CENTER])
    handles = [
        *compact_handles(drawable),
        Line2D([], [], linestyle=REFERENCE_DASH, color=MUTED_INK, label="no change"),
    ]
    if replicated(drawable):
        handles.append(spread_handle())
    figure, axis = _scatter_figure(handles)
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
    reach = _reach(drawable, (DELTA_PFS, DELTA_CENTER))
    axis.set_xlim(-reach * 1.25, reach * 1.25)
    axis.set_ylim(-reach * 1.25, reach * 1.25)
    _corner_notes(axis, SHIFT_CORNERS)
    axis.set_xlabel("Change in PFS against the as-released variant")
    axis.set_ylabel("Change in center alignment\nagainst the as-released variant")
    # axis.set_title(_title(view, mode, POPULATION, "What the finetuning bought"))
    return save_plate(figure, destination / f"fig_fidelity_shift_{view}_{mode}")


def cells_plate(frame: pd.DataFrame, mode: str, view: str, destination: Path) -> list[Path]:
    ordered = _ordered(frame, "e_mean_nemd").iloc[::-1]
    figure, axis = _row_figure(len(ordered), ROW_WIDTH)
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
    axis.set_xticks(range(matrix.shape[1]), list(matrix.columns), rotation=30, ha="right")
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
        label="Score (0 to 1, higher is better)",
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
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    groups, overall, paired, paired_overall = (
        across_replicates(frame, spread=SPREAD)
        for frame in (groups, overall, paired, paired_overall)
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
                produced += center_plate(pooled, mode, view, destination)
                produced += cells_plate(pooled, mode, view, destination)
                shifts = view_frame(paired, paired_overall, view, mode, POPULATION)
                if not shifts.empty:
                    produced += shift_plate(shifts, mode, view, destination)
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
    return produced
