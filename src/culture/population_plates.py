from __future__ import annotations

from collections.abc import Sequence
from math import ceil
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D

from machine_bias_reproduction.config import FIGURES_ROOT
from machine_bias_reproduction.figures import save_plate
from machine_bias_reproduction.plates import GRID, MUTED_INK, SURFACE

from .matching import CULTURE_COUNTRIES
from .palette import (
    MIXTRAL_ARCHIVED,
    REFERENCE_MARKERS,
    REFERENCE_TONES,
    arm_fill,
    arm_order,
    model_marker,
    model_tone,
)
from .population import MODEL_INDEX, REFERENCE_COUNTRIES, across_replicates
from .registry import BASE_ARM, arm_display, is_base

VIEWS: dict[str, str | None] = {
    "all": None,
    "happiness": "d_happy",
    "politics": "d_polpos",
    "religious": "d_religiousp",
    "trust": "d_trust",
}

PLATE_TEXT: dict[str, Any] = {
    "font.size": 16.0,
    "axes.labelsize": 16.0,
    "axes.titlesize": 17.0,
    "figure.titlesize": 18.0,
    "figure.labelsize": 16.0,
    "xtick.labelsize": 15.0,
    "ytick.labelsize": 15.0,
    "legend.fontsize": 14.0,
}

NOTE_SIZE = 11.5

PANEL_NOTE_SIZE = 9.5

MARKER_SIZE = 9.0

MARKER_EDGE = 1.4

LEGEND_ROW = 0.36

REACH_RULE: dict[str, Any] = {"linestyle": (0, (1, 1.2)), "linewidth": 1.1}

XTICK: dict[str, Any] = {"marker": "|", "markersize": 10, "markeredgewidth": 1.5}

YTICK: dict[str, Any] = {"marker": "_", "markersize": 10, "markeredgewidth": 1.5}

SLASH_TICK: dict[str, Any] = {"marker": (2, 0, 45), "markersize": 10, "markeredgewidth": 1.5}

REFERENCE_ADJECTIVES: dict[str, str] = {"german": "German", "mexican": "Mexican"}

# Each reference country reaches from the marker to its own cells, capped by a tick
# no series wears: an upright bar for Germany, a slash for Mexico. The bar keeps its
# axis-dependent orientation (vertical on the x reaches, horizontal on the y ones);
# the slash is the same either way, so only the angle says which country is which.
REFERENCE_TICKS: dict[str, dict[str, dict[str, Any]]] = {
    "german": {"x": XTICK, "y": YTICK},
    "mexican": {"x": SLASH_TICK, "y": SLASH_TICK},
}

# Germany keeps the unqualified stem it has always written, so the figure the paper
# already includes keeps its name; every further reference country is named in its own.
CENTER_STEMS: dict[str, str] = {
    "german": "fig_population_center",
    "mexican": "fig_population_center_mexican",
}

POPULATION_SPREAD: tuple[str, ...] = (
    "e_mean_nemd",
    "adaptability_ratio",
    "rho_structure",
    "pearson_structure",
    "c_center_pop_nemd",
    *(f"c_center_{key}_nemd" for key in REFERENCE_COUNTRIES),
)

SPREAD_BAR: dict[str, Any] = {
    "fmt": "none",
    "elinewidth": 1.1,
    "capsize": 2.6,
    "capthick": 1.1,
    "alpha": 0.75,
    "zorder": 2.8,
}

SPREAD_LABEL = "95% CI across runs"


class Whisker(Line2D):
    pass


LEGEND_HANDLERS: dict[Any, Any] = {Whisker: HandlerLine2D(numpoints=2, marker_pad=0.1)}

FILL_NAMES: dict[str, str] = {
    "none": "hollow",
    "full": "solid",
    "bottom": "bottom half",
    "top": "top half",
    "left": "left half",
    "right": "right half",
}

ADAPTABILITY_XLABEL = "Mean nEMD to WVS cells (E)"

ADAPTABILITY_YLABEL = "Adaptability ratio A\n(model / survey dispersion)"

TICK_LABEL = "same run, {} cells alone"

ADAPTABILITY_CORNERS = (
    (0.02, 0.97, "left", "top", "Accurate but\nover-dispersed"),
    (0.02, 0.03, "left", "bottom", "Accurate but\nunder-dispersed"),
    (0.98, 0.03, "right", "bottom", "Inaccurate and\nunder-dispersed"),
    (0.98, 0.97, "right", "top", "Inaccurate and\nover-dispersed"),
)

STRUCTURE_CORNERS = (
    (0.02, 0.97, "left", "top", "Compressed,\ncorrectly directed"),
    (0.98, 0.97, "right", "top", "Expansive,\ncorrectly directed"),
    (0.02, 0.03, "left", "bottom", "Compressed and\nunstructured"),
    (0.98, 0.03, "right", "bottom", "Over-steerability\nwithout social fidelity"),
)


def series_style(label: str, model_key: Any, arm: Any) -> dict[str, Any]:
    if label in REFERENCE_TONES:
        ink = REFERENCE_TONES[label].ink
        return {
            "marker": REFERENCE_MARKERS[label],
            "color": ink,
            "markerfacecolor": ink,
            "markeredgecolor": ink if label == MIXTRAL_ARCHIVED else MUTED_INK,
            "zorder": 3.2 if label == MIXTRAL_ARCHIVED else 3.0,
        }
    index = MODEL_INDEX[model_key]
    ink = model_tone(model_key, index).ink
    return {
        "marker": model_marker(model_key, index),
        "color": ink,
        "fillstyle": arm_fill(arm),
        "markerfacecolor": ink,
        "markerfacecoloralt": SURFACE,
        "markeredgecolor": ink,
        "zorder": 3.1,
    }


def is_served_model(row: pd.Series) -> bool:
    return isinstance(row["model_key"], str)


# Plot-only names for the two cultural arms. They stay out of ARM_DISPLAY, which also
# builds the series label written to every table, so the tables keep the arm identifier.
ARM_PLOT_NAME: dict[str, str] = {
    "german": "German",
    "spanish-mx": "Mexican",
}

SERIES_PLOT_NAME: dict[str, str] = {
    f"({arm_display(arm)})": f"({name})" for arm, name in ARM_PLOT_NAME.items()
}


def arm_plot_name(arm: str) -> str:
    return ARM_PLOT_NAME.get(arm, arm_display(arm))


def series_plot_name(series: Any) -> str:
    text = str(series)
    for written, shown in SERIES_PLOT_NAME.items():
        if text.endswith(written):
            return text[: -len(written)] + shown
    return text


def stacked(series: Any) -> str:
    return series_plot_name(series).replace(" (", "\n(", 1)


def compact_handles(frame: pd.DataFrame) -> list[Line2D]:
    handles: list[Line2D] = []
    models = frame.dropna(subset=["model_key"]).drop_duplicates("model_key")
    for row in models.itertuples():
        style = series_style(str(row.series), row.model_key, BASE_ARM)
        handles.append(
            Line2D(
                [], [],
                linestyle="none",
                markersize=MARKER_SIZE,
                label=str(row.model_label),
                **style,
            )
        )
    for arm in arm_order(set(frame["arm"].dropna())):
        fill = arm_fill(arm)
        handles.append(
            Line2D(
                [], [],
                linestyle="none",
                marker="o",
                markersize=MARKER_SIZE,
                color=MUTED_INK,
                markerfacecolor=MUTED_INK,
                markerfacecoloralt=SURFACE,
                markeredgecolor=MUTED_INK,
                fillstyle=cast(Any, fill),
                label=f"{FILL_NAMES.get(fill, fill)}: {arm_plot_name(arm)}",
            )
        )
    references = frame[frame["model_key"].isna()].drop_duplicates("series")
    for row in references.itertuples():
        style = series_style(str(row.series), None, None)
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                markersize=MARKER_SIZE,
                label=series_plot_name(row.series),
                **style,
            )
        )
    return handles


def legend_height(handles: Sequence[Line2D], columns: int) -> float:
    return LEGEND_ROW * ceil(len(handles) / columns) + 0.6


def legend_below(figure: Figure, handles: Sequence[Line2D], columns: int) -> None:
    figure.legend(
        handles=list(handles),
        loc="outside lower center",
        ncols=columns,
        handler_map=LEGEND_HANDLERS,
    )


def spread_of(row: Any, column: str) -> float | None:
    value = getattr(row, f"{column}_ci", None)
    if value is None or not np.isfinite(value) or float(value) <= 0.0:
        return None
    return float(value)


def spread_bars(axis: Axes, row: pd.Series, x_column: str, y_column: str, ink: str) -> None:
    xerr = spread_of(row, x_column)
    yerr = spread_of(row, y_column)
    if xerr is None and yerr is None:
        return
    axis.errorbar(row["_x"], row["_y"], xerr=xerr, yerr=yerr, ecolor=ink, **SPREAD_BAR)


def spread_handle() -> Line2D:
    return Whisker(
        [], [],
        color=MUTED_INK,
        linewidth=SPREAD_BAR["elinewidth"],
        alpha=SPREAD_BAR["alpha"],
        marker="|",
        markersize=2.0 * SPREAD_BAR["capsize"],
        markeredgewidth=SPREAD_BAR["capthick"],
        label=SPREAD_LABEL,
    )


def country_reach(
    axis: Axes,
    row: pd.Series,
    ink: str,
    column: str,
    along: str,
    tick: dict[str, Any],
) -> None:
    if column not in row or pd.isna(row[column]):
        return
    if along == "x":
        axis.plot(
            [row[column], row["_x"]],
            [row["_y"], row["_y"]],
            color=ink,
            alpha=0.85,
            zorder=2,
            **REACH_RULE,
        )
        axis.plot(row[column], row["_y"], linestyle="none", color=ink, zorder=2.5, **tick)
        return
    axis.plot(
        [row["_x"], row["_x"]],
        [row[column], row["_y"]],
        color=ink,
        alpha=0.85,
        zorder=2,
        **REACH_RULE,
    )
    axis.plot(row["_x"], row[column], linestyle="none", color=ink, zorder=2.5, **tick)


def reference_keys(arm: Any) -> tuple[str, ...]:
    """The reference countries a row reaches to.

    A culture-finetuned variant reaches only to the country it was fitted for, so a
    german marker carries the German tick alone and a spanish-mx marker the Mexican
    one. The as-released variant reaches to every reference country, because it is
    the baseline each tuned tick is read against.
    """
    if not isinstance(arm, str) or is_base(arm):
        return tuple(REFERENCE_COUNTRIES)
    targets = set(CULTURE_COUNTRIES.get(arm, ()))
    return tuple(key for key, country in REFERENCE_COUNTRIES.items() if country in targets)


def reference_reaches(
    axis: Axes,
    row: pd.Series,
    ink: str,
    template: str,
    along: str,
) -> None:
    for key in reference_keys(row["arm"]):
        country_reach(axis, row, ink, template.format(key), along, REFERENCE_TICKS[key][along])


def reference_handles(along: str) -> list[Line2D]:
    return [
        Line2D(
            [], [],
            color=MUTED_INK,
            label=TICK_LABEL.format(REFERENCE_ADJECTIVES[key]),
            **REACH_RULE,
            **REFERENCE_TICKS[key][along],
        )
        for key in REFERENCE_COUNTRIES
    ]


def _view_frame(table: pd.DataFrame, mode: str, view: str) -> pd.DataFrame:
    frame = table[table["mode"] == mode]
    question = VIEWS[view]
    if question is not None:
        frame = frame[frame["question"] == question]
    return across_replicates(frame, spread=POPULATION_SPREAD)


def replicated(frame: pd.DataFrame) -> bool:
    return "n_replicates" in frame.columns and bool(frame["n_replicates"].gt(1).any())


def _corner_notes(axis: Axes, corners: tuple[tuple[float, float, str, str, str], ...]) -> None:
    for x, y, ha, va, text in corners:
        axis.text(
            x, y, text,
            transform=axis.transAxes,
            ha=ha, va=va,
            fontsize=PANEL_NOTE_SIZE, style="italic", color=MUTED_INK,
        )


def _label_axes(axes: Sequence[Axes], xlabel: str, ylabel: str) -> None:
    for axis in axes:
        axis.set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)


def _dashed(label: str) -> Line2D:
    return Line2D([], [], linestyle=(0, (4, 2)), color=MUTED_INK, label=label)


def _plate(
    frame: pd.DataFrame,
    mode: str,
    handles: Sequence[Line2D],
    title_suffix: str = "",
) -> tuple[Figure, list[Axes], list[str]]:
    questions = list(dict.fromkeys(frame["question"]))
    count = len(questions)
    columns = 4 if count >= 3 else 2
    width = 3.4 * count if count >= 3 else 6.8
    figure = Figure(
        figsize=(width, 4.8 + legend_height(handles, columns)), layout="constrained"
    )
    axes = list(figure.subplots(1, count, sharey=True, squeeze=False)[0])
    # figure.suptitle(f"{mode.upper()}{title_suffix}")
    legend_below(figure, handles, columns)
    return figure, axes, questions


def _marker(axis: Axes, row: pd.Series, style: dict[str, Any]) -> None:
    axis.plot(
        row["_x"], row["_y"],
        linestyle="none",
        markersize=MARKER_SIZE,
        markeredgewidth=MARKER_EDGE,
        **style,
    )


def adaptability_plate(
    table: pd.DataFrame,
    mode: str,
    view: str,
    destination: Path = FIGURES_ROOT,
) -> list[Path]:
    frame = _view_frame(table, mode, view)
    handles = [
        *compact_handles(frame),
        *reference_handles("x"),
        _dashed("survey dispersion (A = 1)"),
        Line2D([], [], linestyle="none", label="ideal: E → 0 with A = 1"),
    ]
    if replicated(frame):
        handles.append(spread_handle())
    with rc_context(cast(Any, PLATE_TEXT)):
        figure, axes, questions = _plate(frame, mode, handles)
        top = max(1.08, float(frame["adaptability_ratio"].max()) * 1.12)
        for axis, question in zip(axes, questions, strict=True):
            subset = frame[frame["question"] == question].copy()
            subset["_x"] = subset["e_mean_nemd"]
            subset["_y"] = subset["adaptability_ratio"]
            axis.axhline(1.0, linestyle=(0, (4, 2)), color=MUTED_INK, linewidth=1.1)
            axis.grid(color=GRID, linewidth=0.6)
            axis.set_axisbelow(True)
            for _, row in subset.iterrows():
                style = series_style(row["series"], row["model_key"], row["arm"])
                if is_served_model(row):
                    reference_reaches(axis, row, style["color"], "e_mean_nemd_{}", "x")
                spread_bars(axis, row, "e_mean_nemd", "adaptability_ratio", style["color"])
                _marker(axis, row, style)
            axis.set_title(str(subset["question_label"].iat[0]))
            axis.set_xlim(left=0)
            axis.set_ylim(-0.03 * top, top)
        _label_axes(axes, ADAPTABILITY_XLABEL, ADAPTABILITY_YLABEL)
        _corner_notes(axes[0], ADAPTABILITY_CORNERS)
        return save_plate(figure, destination / f"fig_population_adaptability_{view}_{mode}")


def center_plate(
    table: pd.DataFrame,
    mode: str,
    view: str,
    destination: Path = FIGURES_ROOT,
    country: str = "german",
) -> list[Path]:
    frame = _view_frame(table, mode, view)
    adjective = REFERENCE_ADJECTIVES[country]
    column = f"c_center_{country}_nemd"
    handles = [
        *compact_handles(frame),
        _dashed(f"indifference: C_{country} = C_pop"),
    ]
    if replicated(frame):
        handles.append(spread_handle())
    with rc_context(cast(Any, PLATE_TEXT)):
        figure, axes, questions = _plate(frame, mode, handles)
        limit = float(max(frame[column].max(), frame["c_center_pop_nemd"].max())) * 1.12
        for axis, question in zip(axes, questions, strict=True):
            subset = frame[frame["question"] == question].copy()
            subset["_x"] = subset[column]
            subset["_y"] = subset["c_center_pop_nemd"]
            axis.plot(
                [0, limit], [0, limit], linestyle=(0, (4, 2)), color=MUTED_INK, linewidth=1.1
            )
            axis.grid(color=GRID, linewidth=0.6)
            axis.set_axisbelow(True)
            for _, row in subset.iterrows():
                style = series_style(row["series"], row["model_key"], row["arm"])
                spread_bars(axis, row, column, "c_center_pop_nemd", style["color"])
                _marker(axis, row, style)
            axis.set_title(str(subset["question_label"].iat[0]))
            axis.set_xlim(-0.02 * limit, limit)
            axis.set_ylim(-0.02 * limit, limit)
        _label_axes(
            axes,
            f"C_{country}: nEMD to the\npooled {adjective} cells",
            "C_pop: nEMD to the\npooled cells",
        )
        _corner_notes(
            axes[0],
            (
                (0.03, 0.97, "left", "top", f"closer to the\n{adjective} pool"),
                (0.97, 0.03, "right", "bottom", "closer to the\npooled world"),
            ),
        )
        return save_plate(figure, destination / f"{CENTER_STEMS[country]}_{view}_{mode}")


def structure_plate(
    joined: pd.DataFrame,
    mode: str,
    view: str,
    column: str,
    ylabel: str,
    stem: str,
    destination: Path = FIGURES_ROOT,
) -> list[Path]:
    frame = _view_frame(joined, mode, view)
    handles = [
        *compact_handles(frame),
        *reference_handles("y"),
        _dashed("A = 1 and correlation = 0"),
        Line2D([], [], linestyle="none", label="ideal: A = 1, high correlation"),
    ]
    if replicated(frame):
        handles.append(spread_handle())
    with rc_context(cast(Any, PLATE_TEXT)):
        figure, axes, questions = _plate(
            frame, mode, handles, "" if column == "rho_structure" else " — Pearson"
        )
        right = max(1.08, float(frame["adaptability_ratio"].max()) * 1.12)
        for axis, question in zip(axes, questions, strict=True):
            subset = frame[frame["question"] == question].copy()
            subset["_x"] = subset["adaptability_ratio"]
            subset["_y"] = subset[column]
            axis.axvline(1.0, linestyle=(0, (4, 2)), color=MUTED_INK, linewidth=1.1)
            axis.axhline(0.0, linestyle=(0, (4, 2)), color=MUTED_INK, linewidth=1.1)
            axis.grid(color=GRID, linewidth=0.6)
            axis.set_axisbelow(True)
            for _, row in subset.iterrows():
                style = series_style(row["series"], row["model_key"], row["arm"])
                if is_served_model(row):
                    reference_reaches(axis, row, style["color"], column + "_{}", "y")
                spread_bars(axis, row, "adaptability_ratio", column, style["color"])
                _marker(axis, row, style)
            axis.set_title(str(subset["question_label"].iat[0]))
            axis.set_xlim(-0.02 * right, right)
            axis.set_ylim(-1.05, 1.05)
        _label_axes(axes, ADAPTABILITY_YLABEL, ylabel)
        _corner_notes(axes[0], STRUCTURE_CORNERS)
        return save_plate(figure, destination / f"{stem}_{view}_{mode}")
