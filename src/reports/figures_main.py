"""The paper's main-text figures: 2, 3, 4 and 6.

Bare plates — no titles, no captions. Panel titles and series legends stay.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from machine_bias_reproduction.figures import (
    NEMD_BREAKS,
    PREDICTOR_GROUPS,
    _annotate_quality_bands,
    _classical_mds,
    _draw_coefficients,
    _log_nemd_density,
    _predictor_rows,
    _save,
)
from machine_bias_reproduction.metrics import pairwise_nemd_matrix
from machine_bias_reproduction.questions import QUESTIONS

from .load import QuestionData
from .series import SERIES_NAMES

matplotlib.use("Agg")
from matplotlib import pyplot as plt

SERIES_COLORS = {
    "NTP-GPT-4T": "#1b9e77",
    "NTP-Llama-3-70B": "#d95f02",
    "NTP-Mixtral-8x7B": "#7570b3",
    "FA-GPT-3": "#e7298a",
    "FA-Llama-3-70B": "#66a61e",
    "FA-Mixtral-8x7B": "#e6ab02",
    "Linear": "#000000",
    "Random": "#808080",
}
MDS_COLORS = {"WVS": "#fde725", "Model": "#440154"}


def _method_order(distances: pd.DataFrame) -> list[str]:
    present = set(distances["series"].unique())
    order = ["Linear", "Random", *SERIES_NAMES]
    return [name for name in order if name in present]


def _pooled(distances: pd.DataFrame) -> pd.DataFrame:
    frame = distances.copy()
    frame["series"] = frame["method"].str.replace(r"^Random_\d+$", "Random", regex=True)
    return frame


def figure2(distances: pd.DataFrame, destination: Path) -> list[Path]:
    """Figure 2 — density of nEMD, one facet per question, all series overlaid."""
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
            grid, density = _log_nemd_density(values)
            axis.plot(
                grid,
                density,
                label=name,
                color=SERIES_COLORS.get(name, "#444444"),
                linewidth=1.5,
            )
        axis.set_xscale("log")
        axis.set_xlim(0.01, 1.0)
        axis.set_xticks(NEMD_BREAKS, [f"{break_:.2f}" for break_ in NEMD_BREAKS])
        axis.set_ylim(bottom=0)
        _annotate_quality_bands(axis)
        axis.set(xlabel="nEMD (log scale)", title=QUESTIONS[var].label)
    axes[0][0].set_ylabel("Density (per natural-log unit)")
    axes[0][0].legend(frameon=False, loc="upper left", fontsize=8)
    figure.tight_layout()
    return _save(figure, destination / "Figure-2-nEMD-density")


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
    rows = _predictor_rows({name for name, _, _ in PREDICTOR_GROUPS})
    r_squared = (
        fit[(fit["series"] == series) & (fit["model"] == model.replace("_standardized", ""))]
        .set_index("question")["adjusted_r_squared"]
        .to_dict()
    )
    height = max(8.0, 0.34 * len(rows))
    figure, axes = plt.subplots(
        1, len(questions), figsize=(4.6 * len(questions), height), sharex=True, squeeze=False
    )
    for axis, var in zip(axes[0], questions, strict=True):
        _draw_coefficients(
            axis, frame[frame["question"] == var], rows, SERIES_COLORS.get(series, "#444444")
        )
        adjusted = r_squared.get(var)
        suffix = f"  (adj. $R^2$ = {adjusted:.3f})" if adjusted is not None else ""
        axis.set_title(f"{QUESTIONS[var].label}{suffix}")
        axis.set_xlabel(xlabel)
    for axis in axes[0][1:]:
        axis.tick_params(labelleft=False)
    figure.tight_layout()
    return _save(figure, stem)


def figure3(
    coefficients: pd.DataFrame, fit: pd.DataFrame, series: str, destination: Path
) -> list[Path]:
    """Figure 3 — social-model coefficients for one series, faceted by question."""
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
    """Figure 6 — standardized full-model coefficients for one series."""
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
    """Draw an MDS panel per question and series, on one shared embedding.

    The upstream embedding repeats the WVS block once per series before
    computing the distance matrix (``6-results.R:861-863``), so every panel of a
    question shares one configuration and one scale.
    """
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
        coordinates = _classical_mds(pairwise_nemd_matrix(stacked))
        count = len(data.names)
        for column, name in enumerate(columns):
            axis = axes[row][column]
            if name not in available:
                axis.set_axis_off()
                continue
            index = available.index(name)
            wvs_points = coordinates[index * count : (index + 1) * count]
            model_points = coordinates[
                (len(available) + index) * count : (len(available) + index + 1) * count
            ]
            axis.scatter(
                wvs_points[:, 0],
                wvs_points[:, 1],
                s=10,
                alpha=0.35,
                marker="^",
                label="WVS",
                color=MDS_COLORS["WVS"],
                edgecolors="#8a8a2a",
                linewidths=0.3,
            )
            axis.scatter(
                model_points[:, 0],
                model_points[:, 1],
                s=10,
                alpha=0.35,
                marker="o",
                label="Model",
                color=MDS_COLORS["Model"],
                edgecolors="none",
            )
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(name, fontsize=9)
            if column == 0:
                axis.set_ylabel(QUESTIONS[var].label, fontsize=9)
    axes[0][0].legend(frameon=False, loc="upper left", fontsize=7, markerscale=1.6)
    figure.tight_layout()
    return _save(figure, stem)


def figure4(loaded: dict[str, QuestionData], series: list[str], destination: Path) -> list[Path]:
    """Figure 4 — MDS of the three main NTP series against the WVS."""
    return mds_plate(loaded, series, destination / "Figure-4-MDS-NTP")
