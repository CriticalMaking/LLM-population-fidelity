from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from machine_bias_reproduction.figures import (
    NEMD_BREAKS,
    PREDICTOR_GROUPS,
    annotate_quality_bands,
    classical_mds,
    draw_coefficients,
    label_fit,
    log_nemd_density,
    mds_block,
    predictor_rows,
    save_plate,
)
from machine_bias_reproduction.metrics import pairwise_nemd_matrix
from machine_bias_reproduction.plates import (
    METHOD_TONES,
    MODEL_TONE,
    MUTED_INK,
    SURVEY_TONE,
    Tone,
    slot_linestyle,
    slot_tone,
)
from machine_bias_reproduction.questions import QUESTIONS

from .load import QuestionData
from .series import SERIES_NAMES

SERIES_TONES: dict[str, Tone] = {
    name: slot_tone(index) for index, name in enumerate(SERIES_NAMES)
} | {"Linear": METHOD_TONES["Linear"], "Random": METHOD_TONES["Random"]}

SERIES_LINESTYLES = {name: slot_linestyle(index) for index, name in enumerate(SERIES_NAMES)} | {
    "Linear": "solid",
    "Random": (0, (1, 1.6)),
}


def _method_order(distances: pd.DataFrame) -> list[str]:
    present = set(distances["series"].unique())
    order = ["Linear", "Random", *SERIES_NAMES]
    return [name for name in order if name in present]


def _pooled(distances: pd.DataFrame) -> pd.DataFrame:
    frame = distances.copy()
    frame["series"] = frame["method"].str.replace(r"^Random_\d+$", "Random", regex=True)
    return frame


def figure2(distances: pd.DataFrame, destination: Path) -> list[Path]:
    frame = _pooled(distances)
    questions = [var for var in QUESTIONS if var in set(frame["question"])]
    figure, axes = plt.subplots(
        1, len(questions), figsize=(5.0 * len(questions), 5.2), sharey=True, squeeze=False
    )
    for axis, var in zip(axes[0], questions, strict=True):
        subset = frame[frame["question"] == var]
        for name in _method_order(subset):
            values = subset.loc[subset["series"] == name, "nEMD"].to_numpy(dtype=np.float64)
            if not np.any(values > 0):
                continue
            grid, density = log_nemd_density(values)
            axis.plot(
                grid,
                density,
                label=name,
                color=SERIES_TONES[name].ink if name in SERIES_TONES else MUTED_INK,
                linestyle=SERIES_LINESTYLES.get(name, "solid"),
                linewidth=1.4,
            )
        axis.set_xscale("log")
        axis.set_xlim(0.01, 1.0)
        axis.set_xticks(NEMD_BREAKS, [f"{threshold:.2f}" for threshold in NEMD_BREAKS])
        axis.set_ylim(bottom=0)
        annotate_quality_bands(axis)
        axis.set_xlabel(f"nEMD (log scale) — {QUESTIONS[var].label}")
    axes[0][0].set_ylabel("Density (per natural-log unit)")
    axes[0][0].legend(loc="upper left")
    figure.tight_layout()
    return save_plate(figure, destination / "Figure-2-nEMD-density")


def _coefficient_plate(
    coefficients: pd.DataFrame,
    fit: pd.DataFrame,
    series: str,
    model: str,
    stem: Path,
    xlabel: str,
) -> list[Path]:
    frame = coefficients[
        (coefficients["series"] == series)
        & (coefficients["model"] == model)
        & (coefficients["predictor"] != "const")
    ]
    if frame.empty:
        return []
    questions = [var for var in QUESTIONS if var in set(frame["question"])]
    rows = predictor_rows({name for name, _, _ in PREDICTOR_GROUPS})
    r_squared = (
        fit[(fit["series"] == series) & (fit["model"] == model.replace("_standardized", ""))]
        .set_index("question")["adjusted_r_squared"]
        .to_dict()
    )
    height = max(8.0, 0.34 * len(rows))
    figure, axes = plt.subplots(
        1, len(questions), figsize=(4.6 * len(questions), height), sharex=True, squeeze=False
    )
    ink = SERIES_TONES[series].ink if series in SERIES_TONES else MUTED_INK
    for axis, var in zip(axes[0], questions, strict=True):
        draw_coefficients(axis, frame[frame["question"] == var], rows, ink)
        adjusted = r_squared.get(var)
        suffix = f" · adj. $R^2$ = {adjusted:.3f}" if adjusted is not None else ""
        label_fit(axis, f"{QUESTIONS[var].label}{suffix}")
        axis.set_xlabel(xlabel)
    for axis in axes[0][1:]:
        axis.tick_params(labelleft=False)
    figure.tight_layout()
    return save_plate(figure, stem)


def figure3(
    coefficients: pd.DataFrame, fit: pd.DataFrame, series: str, destination: Path
) -> list[Path]:
    return _coefficient_plate(
        coefficients,
        fit,
        series,
        "social",
        destination / f"Figure-3-social-{series}",
        "Regression coefficient — outcome log1p(nEMD)",
    )


def figure6(
    coefficients: pd.DataFrame, fit: pd.DataFrame, series: str, destination: Path
) -> list[Path]:
    return _coefficient_plate(
        coefficients,
        fit,
        series,
        "full_standardized",
        destination / f"Figure-6-{series}",
        "Coefficient per 1 SD of predictor — outcome log1p(nEMD)",
    )


def mds_plate(
    loaded: dict[str, QuestionData],
    series: list[str],
    stem: Path,
    *,
    include_baselines: bool = False,
) -> list[Path]:
    questions = [var for var in QUESTIONS if var in loaded]
    columns = [*(["Linear", "Random"] if include_baselines else []), *series]
    figure, axes = plt.subplots(
        len(questions),
        len(columns),
        figsize=(3.4 * len(columns), 3.4 * len(questions)),
        squeeze=False,
    )
    for row, var in enumerate(questions):
        data = loaded[var]
        available = [
            name
            for name in columns
            if (data.props(name) is not None or (name == "Random" and data.random))
        ]
        blocks: list[np.ndarray] = []
        wvs = data.wvs.to_numpy(dtype=np.float64)
        for name in available:
            props = data.random[0] if name == "Random" else data.props(name)
            if props is None:
                continue
            blocks.append(props.to_numpy(dtype=np.float64))
        if not blocks:
            for axis in axes[row]:
                axis.set_axis_off()
            continue
        stacked = np.vstack([wvs] * len(blocks) + blocks)
        coordinates = classical_mds(pairwise_nemd_matrix(stacked))
        count = len(data.names)
        for column, name in enumerate(columns):
            axis = axes[row][column]
            if name not in available:
                axis.set_axis_off()
                continue
            index = available.index(name)
            wvs_points = mds_block(coordinates, index, count)
            model_points = mds_block(coordinates, len(available) + index, count)
            axis.scatter(
                wvs_points[:, 0],
                wvs_points[:, 1],
                s=10,
                alpha=0.6,
                marker="^",
                label="WVS",
                color=SURVEY_TONE.fill,
                edgecolors=SURVEY_TONE.ink,
                linewidths=0.3,
            )
            axis.scatter(
                model_points[:, 0],
                model_points[:, 1],
                s=10,
                alpha=0.6,
                marker="o",
                label="Model",
                color=MODEL_TONE.fill,
                edgecolors=MODEL_TONE.ink,
                linewidths=0.3,
            )
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_xlabel(name, fontsize=9, labelpad=6)
                axis.xaxis.set_label_position("top")
            if column == 0:
                axis.set_ylabel(QUESTIONS[var].label, fontsize=9)
    axes[0][0].legend(loc="upper left", fontsize=7, markerscale=1.6)
    figure.tight_layout()
    return save_plate(figure, stem)


def figure4(loaded: dict[str, QuestionData], series: list[str], destination: Path) -> list[Path]:
    return mds_plate(loaded, series, destination / "Figure-4-MDS-NTP")
