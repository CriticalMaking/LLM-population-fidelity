from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from machine_bias_reproduction.config import FIGURES_ROOT
from machine_bias_reproduction.figures import save_plate
from machine_bias_reproduction.plates import GRID, MUTED_INK

from .palette import (
    MIXTRAL_ARCHIVED,
    REFERENCE_MARKERS,
    REFERENCE_TONES,
    model_marker,
    model_tone,
)
from .population import MODEL_INDEX
from .registry import is_base

VIEWS: dict[str, str | None] = {
    "all": None,
    "happiness": "d_happy",
    "politics": "d_polpos",
    "religious": "d_religiousp",
    "trust": "d_trust",
}

REACH_RULE: dict[str, Any] = {"linestyle": (0, (1, 1.2)), "linewidth": 0.9}

XTICK: dict[str, Any] = {"marker": "|", "markersize": 7, "markeredgewidth": 1.2}

YTICK: dict[str, Any] = {"marker": "_", "markersize": 7, "markeredgewidth": 1.2}

ADAPTABILITY_XLABEL = "Mean nEMD to the WVS cells (E)"

ADAPTABILITY_YLABEL = "Model dispersion / survey dispersion (AR)"

GERMAN_TICK_LABEL = "same run over the German cells alone"

ADAPTABILITY_CORNERS = (
    (0.02, 0.97, "left", "top", "Accurate but over-dispersed"),
    (0.02, 0.03, "left", "bottom", "Accurate but under-dispersed"),
    (0.98, 0.03, "right", "bottom", "Inaccurate and under-dispersed"),
    (0.98, 0.97, "right", "top", "Inaccurate and over-dispersed"),
)

STRUCTURE_CORNERS = (
    (0.02, 0.97, "left", "top", "Compressed, correctly directed"),
    (0.98, 0.97, "right", "top", "Expansive, correctly directed"),
    (0.02, 0.03, "left", "bottom", "Compressed and unstructured"),
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
        "markerfacecolor": "none" if is_base(arm) else ink,
        "markeredgecolor": ink,
        "zorder": 3.1,
    }


def is_served_model(row: pd.Series) -> bool:
    return isinstance(row["model_key"], str)


def series_handles(frame: pd.DataFrame) -> list[Line2D]:
    handles = []
    for series in dict.fromkeys(frame["series"]):
        subset = frame[frame["series"] == series]
        style = series_style(series, subset["model_key"].iat[0], subset["arm"].iat[0])
        handles.append(Line2D([], [], linestyle="none", markersize=6, label=series, **style))
    return handles


def spacer() -> Line2D:
    return Line2D([], [], linestyle="none", label=" ")


def german_reach(
    axis: Axes,
    row: pd.Series,
    ink: str,
    column: str,
    along: str,
) -> None:
    if pd.isna(row[column]):
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
        axis.plot(row[column], row["_y"], linestyle="none", color=ink, zorder=2.5, **XTICK)
        return
    axis.plot(
        [row["_x"], row["_x"]],
        [row[column], row["_y"]],
        color=ink,
        alpha=0.85,
        zorder=2,
        **REACH_RULE,
    )
    axis.plot(row["_x"], row[column], linestyle="none", color=ink, zorder=2.5, **YTICK)


def _view_frame(table: pd.DataFrame, mode: str, view: str) -> pd.DataFrame:
    frame = table[table["mode"] == mode]
    question = VIEWS[view]
    if question is not None:
        frame = frame[frame["question"] == question]
    return frame


def _corner_notes(axis: Axes, corners: tuple[tuple[float, float, str, str, str], ...]) -> None:
    for x, y, ha, va, text in corners:
        axis.text(
            x, y, text,
            transform=axis.transAxes,
            ha=ha, va=va,
            fontsize=7, style="italic", color=MUTED_INK,
        )


def _plate(
    frame: pd.DataFrame,
    mode: str,
    title_suffix: str = "",
) -> tuple[Figure, list[Axes], list[str]]:
    questions = list(dict.fromkeys(frame["question"]))
    count = len(questions)
    width = 3.6 * count + (0.0 if count >= 3 else 4.0)
    figure = Figure(figsize=(width, 4.2), layout="constrained")
    axes = list(figure.subplots(1, count, sharey=True, squeeze=False)[0])
    figure.suptitle(f"{mode.upper()}{title_suffix}")
    return figure, axes, questions


def adaptability_plate(
    table: pd.DataFrame,
    mode: str,
    view: str,
    destination: Path = FIGURES_ROOT,
) -> list[Path]:
    frame = _view_frame(table, mode, view)
    figure, axes, questions = _plate(frame, mode)
    top = max(1.08, float(frame["adaptability_ratio"].max()) * 1.12)
    for axis, question in zip(axes, questions, strict=True):
        subset = frame[frame["question"] == question].copy()
        subset["_x"] = subset["e_mean_nemd"]
        subset["_y"] = subset["adaptability_ratio"]
        axis.axhline(1.0, linestyle=(0, (4, 2)), color=MUTED_INK, linewidth=0.9)
        axis.grid(color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        for _, row in subset.iterrows():
            style = series_style(row["series"], row["model_key"], row["arm"])
            if is_served_model(row):
                german_reach(axis, row, style["color"], "e_mean_nemd_german", "x")
            axis.plot(
                row["_x"], row["_y"],
                linestyle="none",
                markersize=6.5,
                markeredgewidth=1.1,
                **style,
            )
        axis.set_title(str(subset["question_label"].iat[0]))
        axis.set_xlim(left=0)
        axis.set_ylim(-0.03 * top, top)
    axes[0].set_ylabel(ADAPTABILITY_YLABEL)
    _corner_notes(axes[0], ADAPTABILITY_CORNERS)
    figure.supxlabel(ADAPTABILITY_XLABEL, fontsize=9)
    handles = series_handles(frame)
    handles.append(spacer())
    handles.append(
        Line2D([], [], color=MUTED_INK, label=GERMAN_TICK_LABEL, **REACH_RULE, **XTICK)
    )
    handles.append(
        Line2D(
            [], [],
            linestyle=(0, (4, 2)),
            color=MUTED_INK,
            label="reference: survey dispersion (AR = 1)",
        )
    )
    handles.append(Line2D([], [], linestyle="none", label="ideal: E → 0 with AR = 1"))
    figure.legend(handles=handles, loc="outside right upper")
    return save_plate(figure, destination / f"fig_population_adaptability_{view}_{mode}")


def center_plate(
    table: pd.DataFrame,
    mode: str,
    view: str,
    destination: Path = FIGURES_ROOT,
) -> list[Path]:
    frame = _view_frame(table, mode, view)
    figure, axes, questions = _plate(frame, mode)
    limit = (
        float(max(frame["c_center_german_nemd"].max(), frame["c_center_pop_nemd"].max())) * 1.12
    )
    for axis, question in zip(axes, questions, strict=True):
        subset = frame[frame["question"] == question]
        axis.plot([0, limit], [0, limit], linestyle=(0, (4, 2)), color=MUTED_INK, linewidth=0.9)
        axis.grid(color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        for _, row in subset.iterrows():
            style = series_style(row["series"], row["model_key"], row["arm"])
            axis.plot(
                row["c_center_german_nemd"],
                row["c_center_pop_nemd"],
                linestyle="none",
                markersize=6.5,
                markeredgewidth=1.1,
                **style,
            )
        axis.set_title(str(subset["question_label"].iat[0]))
        axis.set_xlim(-0.02 * limit, limit)
        axis.set_ylim(-0.02 * limit, limit)
    axes[0].set_ylabel("C_pop, nEMD to every subpopulation pooled")
    axes[0].text(
        0.03, 0.97, "closer to the German pool",
        transform=axes[0].transAxes, ha="left", va="top",
        fontsize=7, style="italic", color=MUTED_INK,
    )
    axes[0].text(
        0.97, 0.03, "closer to the pooled world",
        transform=axes[0].transAxes, ha="right", va="bottom",
        fontsize=7, style="italic", color=MUTED_INK,
    )
    figure.supxlabel("C_german, nEMD to the pooled German cells", fontsize=9)
    handles = series_handles(frame)
    handles.append(spacer())
    handles.append(
        Line2D(
            [], [],
            linestyle=(0, (4, 2)),
            color=MUTED_INK,
            label="reference: indifference (C_german = C_pop)",
        )
    )
    handles.append(
        Line2D([], [], linestyle="none", label="x carries the German reference here")
    )
    figure.legend(handles=handles, loc="outside right upper")
    return save_plate(figure, destination / f"fig_population_center_{view}_{mode}")


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
    figure, axes, questions = _plate(
        frame, mode, "" if column == "rho_structure" else " — Pearson"
    )
    right = max(1.08, float(frame["adaptability_ratio"].max()) * 1.12)
    for axis, question in zip(axes, questions, strict=True):
        subset = frame[frame["question"] == question].copy()
        subset["_x"] = subset["adaptability_ratio"]
        subset["_y"] = subset[column]
        axis.axvline(1.0, linestyle=(0, (4, 2)), color=MUTED_INK, linewidth=0.9)
        axis.axhline(0.0, linestyle=(0, (4, 2)), color=MUTED_INK, linewidth=0.9)
        axis.grid(color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        for _, row in subset.iterrows():
            style = series_style(row["series"], row["model_key"], row["arm"])
            if is_served_model(row):
                german_reach(axis, row, style["color"], f"{column}_german", "y")
            axis.plot(
                row["_x"], row["_y"],
                linestyle="none",
                markersize=6.5,
                markeredgewidth=1.1,
                **style,
            )
        axis.set_title(str(subset["question_label"].iat[0]))
        axis.set_xlim(-0.02 * right, right)
        axis.set_ylim(-1.05, 1.05)
    axes[0].set_ylabel(ylabel)
    _corner_notes(axes[0], STRUCTURE_CORNERS)
    figure.supxlabel(ADAPTABILITY_YLABEL, fontsize=9)
    handles = series_handles(frame)
    handles.append(spacer())
    handles.append(
        Line2D([], [], color=MUTED_INK, label=GERMAN_TICK_LABEL, **REACH_RULE, **YTICK)
    )
    handles.append(
        Line2D(
            [], [],
            linestyle=(0, (4, 2)),
            color=MUTED_INK,
            label="reference: AR = 1 / correlation = 0",
        )
    )
    handles.append(
        Line2D([], [], linestyle="none", label="ideal: AR = 1 with the correlation high")
    )
    figure.legend(handles=handles, loc="outside right upper")
    return save_plate(figure, destination / f"{stem}_{view}_{mode}")
