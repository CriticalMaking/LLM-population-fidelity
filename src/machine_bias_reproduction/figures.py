from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.linalg
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from .config import RunPaths
from .data import PreparedData
from .metrics import DISTANCES, QUALITY_LABELS, pairwise_nemd_matrix

matplotlib.use("Agg")
from matplotlib import pyplot as plt

COLORS = {
    "Linear": "#000000",
    "Random": "#808080",
    "NTP": "#d95f02",
    "FA": "#7570b3",
}
LINESTYLES = {
    "Linear": "solid",
    "Random": "solid",
    "NTP": (0, (6, 3)),
    "FA": (0, (7, 2, 1, 2)),
}
MDS_COLORS = {"WVS": "#fde725", "Model": "#440154"}
QUALITY_PALETTE = ("#1b9e77", "#66c2a5", "#ffd92f", "#fc8d62", "#d73027")

QUALITY_THRESHOLDS = (0.0, 0.05, 0.10, 0.15, 0.30, 1.0)

NEMD_BREAKS = (0.01, 0.05, 0.10, 0.15, 0.30, 0.50, 1.00)

METHOD_ORDER = ("Linear", "Random", "NTP", "FA")

PREDICTOR_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("Age_<25", "Age", "15-24"),
    ("Age_25-34", "Age", "25-34"),
    ("Age_45-54", "Age", "45-54"),
    ("Age_55-64", "Age", "55-64"),
    ("Age_65-74", "Age", "65-74"),
    ("Age_75+", "Age", "75 and more"),
    ("countryAustralia", "Country", "Australia"),
    ("countryGermany", "Country", "Germany"),
    ("countryMexico", "Country", "Mexico"),
    ("countryRussia", "Country", "Russia"),
    ("decade199X", "Decade", "1990s"),
    ("decade200X", "Decade", "2000s"),
    ("Education_Low", "Education", "Low"),
    ("Education_Middle", "Education", "Middle"),
    ("Employment_Homemaker", "Employment", "Homemaker"),
    ("Employment_Retired", "Employment", "Retired"),
    ("Employment_Student", "Employment", "Student"),
    ("Employment_Unemployed", "Employment", "Unemployed"),
    ("Marstat_Cohabiting", "Marital status", "Cohabiting"),
    ("Marstat_Divorced or separated", "Marital status", "Divorced or separated"),
    ("Marstat_Single", "Marital status", "Single"),
    ("Marstat_Widowed", "Marital status", "Widowed"),
    ("Sex_Female", "Sex", "Female"),
)

CENTER_PREDICTOR = "nEMD_center"

FloatArray = npt.NDArray[np.float64]


PLATE_METADATA: tuple[tuple[str, dict[str, None]], ...] = (
    (".png", {"Software": None}),
    (".pdf", {"CreationDate": None}),
)


def _save(figure: Figure, stem: Path, *, pdf_dpi: int = 180) -> list[Path]:
    paths = []
    for suffix, metadata in PLATE_METADATA:
        path = stem.with_suffix(suffix)
        dpi = pdf_dpi if suffix == ".pdf" else 180
        figure.savefig(path, dpi=dpi, bbox_inches="tight", metadata=metadata)
        paths.append(path)
    plt.close(figure)
    return paths


def _bw_nrd0(sample: FloatArray) -> float:
    deviation = float(np.std(sample, ddof=1))
    low, high = np.percentile(sample, (25, 75))
    spread = min(deviation, float(high - low) / 1.349)
    if spread <= 0:
        spread = deviation or abs(float(sample[0])) or 1.0
    return float(0.9 * spread * sample.size ** (-0.2))


def _log_nemd_density(values: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    sample = np.asarray(values, dtype=np.float64)
    sample = np.log(100.0 * sample[sample > 0])
    bandwidth = _bw_nrd0(sample)
    grid = np.linspace(sample.min() - 3 * bandwidth, sample.max() + 3 * bandwidth, 512)
    weights = np.exp(-0.5 * ((grid[:, None] - sample[None, :]) / bandwidth) ** 2)
    density = weights.sum(axis=1) / (sample.size * bandwidth * math.sqrt(2 * math.pi))
    return np.exp(grid) / 100.0, np.asarray(density, dtype=np.float64)


def _annotate_quality_bands(axis: Axes) -> None:
    for threshold in QUALITY_THRESHOLDS[1:-1]:
        axis.axvline(threshold, color="#333333", linestyle=":", linewidth=0.9)
    lower = np.asarray(QUALITY_THRESHOLDS[:-1], dtype=np.float64)
    upper = np.minimum(np.asarray(QUALITY_THRESHOLDS[1:], dtype=np.float64), 0.45)
    for label, left, right in zip(QUALITY_LABELS, lower, upper, strict=True):
        axis.annotate(
            label,
            xy=((left + right) / 2, 0.0),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#666666"},
        )


def _pooled_methods(distances: pd.DataFrame) -> pd.DataFrame:
    pooled = distances.copy()
    pooled.loc[pooled["method"].str.startswith("Random_"), "method"] = "Random"
    return pooled


def _density_quality(
    paths: RunPaths,
    distances: pd.DataFrame,
    quality: pd.DataFrame,
) -> list[Path]:
    plot_data = _pooled_methods(distances)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))

    for method in METHOD_ORDER:
        values = plot_data.loc[plot_data["method"] == method, "nEMD"].to_numpy(dtype=np.float64)
        grid, density = _log_nemd_density(values)
        axes[0].plot(
            grid,
            density,
            label=method,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.6,
        )
    axes[0].set_xscale("log")
    axes[0].set_xlim(0.01, 1.0)
    axes[0].set_xticks(NEMD_BREAKS, [f"{break_:.2f}" for break_ in NEMD_BREAKS])
    axes[0].set_ylim(bottom=0)
    _annotate_quality_bands(axes[0])
    axes[0].set(
        xlabel="nEMD (log scale)",
        ylabel="Density (per natural-log unit)",
        title="A. Density of nEMD",
    )
    axes[0].legend(frameon=False, loc="upper left")

    pivot = quality.pivot(index="method", columns="quality", values="percent").reindex(
        list(METHOD_ORDER)
    )
    pivot = pivot.loc[:, list(QUALITY_LABELS)]
    pivot.plot(kind="bar", stacked=True, ax=axes[1], color=list(QUALITY_PALETTE), width=0.75)
    bottoms = np.zeros(len(pivot))
    for quality_label in QUALITY_LABELS:
        values = pivot[quality_label].to_numpy(dtype=np.float64)
        for position, (value, bottom) in enumerate(zip(values, bottoms, strict=True)):
            if value >= 4.0:
                axes[1].text(
                    position,
                    bottom + value / 2,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        bottoms = bottoms + values
    axes[1].set(
        xlabel="",
        ylabel="Subpopulations (%)",
        title="B. Prediction-quality bands",
        ylim=(0, 100),
    )
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")

    figure.tight_layout()
    return _save(figure, paths.figures / "fig_nemd_density_quality")


def _distance_comparison(paths: RunPaths, distances: pd.DataFrame) -> list[Path]:
    plot_data = _pooled_methods(distances)
    names = [name for name in DISTANCES if name in plot_data.columns]
    figure, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 4.6))
    for axis, name in zip(np.atleast_1d(axes), names, strict=True):
        for method in METHOD_ORDER:
            values = plot_data.loc[plot_data["method"] == method, name].to_numpy(dtype=np.float64)
            if not np.any(values > 0):
                continue
            grid, density = _log_nemd_density(values)
            axis.plot(
                grid,
                density,
                label=method,
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=1.5,
            )
        axis.set_xscale("log")
        axis.set_ylim(bottom=0)
        axis.set(xlabel=f"{name} (log scale)", title=name)
        if name == "nEMD":
            axis.set_xlim(0.01, 1.0)
            _annotate_quality_bands(axis)
    np.atleast_1d(axes)[0].set_ylabel("Density (per natural-log unit)")
    np.atleast_1d(axes)[0].legend(frameon=False, loc="upper left", fontsize=8)
    figure.tight_layout()
    return _save(figure, paths.figures / "fig_distance_comparison")


def _classical_mds(distance: FloatArray) -> FloatArray:
    count = distance.shape[0]
    centering = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * centering @ np.square(distance) @ centering
    values, vectors = scipy.linalg.eigh(
        gram,
        subset_by_index=(max(0, count - 2), count - 1),
        check_finite=False,
    )
    order = np.argsort(values)[::-1]
    positive = np.maximum(values[order], 0)
    coordinates = np.asarray(vectors[:, order] * np.sqrt(positive), dtype=np.float64)
    span = float(np.ptp(coordinates[:, 0]))
    coordinates = 2 * coordinates / span
    coordinates[:, 0] -= coordinates[:, 0].min() + 1.0
    return coordinates


def _mds(paths: RunPaths, data: PreparedData, distances: pd.DataFrame) -> list[Path]:
    columns = list(data.question.answer_columns)
    modes = ("NTP", "FA")
    model_props = {"NTP": data.ntp_props, "FA": data.fa_props}
    wvs = data.wvs_props.loc[:, columns].to_numpy(dtype=np.float64)
    blocks = [wvs] * len(modes)
    blocks.extend(model_props[mode].loc[:, columns].to_numpy(dtype=np.float64) for mode in modes)
    coordinates = _classical_mds(pairwise_nemd_matrix(np.vstack(blocks)))

    count = len(data.names)
    wvs_coordinates = coordinates[:count]
    model_coordinates = {
        mode: coordinates[(len(modes) + index) * count : (len(modes) + index + 1) * count]
        for index, mode in enumerate(modes)
    }
    errors = distances.groupby("method")["nEMD"].mean()

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True, sharey=True)
    for axis, mode in zip(axes, modes, strict=True):
        axis.scatter(
            wvs_coordinates[:, 0],
            wvs_coordinates[:, 1],
            s=16,
            alpha=0.35,
            marker="^",
            label="WVS",
            color=MDS_COLORS["WVS"],
            edgecolors="#8a8a2a",
            linewidths=0.3,
        )
        axis.scatter(
            model_coordinates[mode][:, 0],
            model_coordinates[mode][:, 1],
            s=16,
            alpha=0.35,
            marker="o",
            label="Model",
            color=MDS_COLORS["Model"],
            edgecolors="none",
        )
        axis.set_aspect("equal")
        axis.set_title(mode)
        axis.set_xlabel("Classical MDS dimension 1")
        axis.annotate(
            f"Err= {errors[mode]:.3f}",
            xy=(0.03, 0.03),
            xycoords="axes fraction",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#666666"},
        )
    axes[0].set_ylabel("Classical MDS dimension 2")
    axes[0].legend(frameon=False, loc="upper left", markerscale=1.8)
    figure.tight_layout()
    return _save(figure, paths.figures / "fig_mds")


def _predictor_rows(predictors: set[str]) -> list[tuple[str | None, str]]:
    rows: list[tuple[str | None, str]] = []
    previous_group: str | None = None
    for predictor, group, label in PREDICTOR_GROUPS:
        if predictor not in predictors:
            continue
        if previous_group is not None and group != previous_group:
            rows.append((None, ""))
        bold = group.replace(" ", r"\ ")
        rows.append((predictor, rf"$\bf{{{bold}}}$: {label}"))
        previous_group = group
    return rows


def _draw_coefficients(
    axis: Axes,
    frame: pd.DataFrame,
    rows: list[tuple[str | None, str]],
    color: str,
) -> None:
    indexed = frame.set_index("predictor")
    positions = [
        position
        for position, (predictor, _) in enumerate(rows)
        if predictor is not None and predictor in indexed.index
    ]
    present = [rows[position][0] for position in positions]
    selected = indexed.reindex(present)
    estimates = selected["estimate"].to_numpy(dtype=np.float64)
    lower = estimates - selected["ci_low"].to_numpy(dtype=np.float64)
    upper = selected["ci_high"].to_numpy(dtype=np.float64) - estimates
    axis.errorbar(
        estimates,
        positions,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        markersize=4,
        color=color,
        ecolor=color,
        capsize=2,
        linewidth=1.2,
    )
    axis.axvline(0, color="#333333", linewidth=1)
    axis.set_yticks(range(len(rows)), [label for _, label in rows])
    axis.set_ylim(len(rows) - 0.5, -0.5)
    axis.grid(axis="x", alpha=0.2)


def _coefficient_figure(
    paths: RunPaths,
    coefficients: pd.DataFrame,
    fit: pd.DataFrame,
    *,
    fit_model: str,
    stem: str,
    xlabel: str,
) -> list[Path]:
    frame = coefficients[coefficients["predictor"] != "const"].copy()
    modes = ("ntp", "fa")
    social_rows = _predictor_rows(set(frame["predictor"]))
    has_center = CENTER_PREDICTOR in set(frame["predictor"])
    r_squared = fit[fit["model"] == fit_model].set_index("mode")["adjusted_r_squared"]

    height = max(8.0, 0.34 * len(social_rows) + (1.4 if has_center else 0.0))
    figure = Figure(figsize=(15, height))
    if has_center:
        grid = GridSpec(
            2,
            2,
            figure=figure,
            height_ratios=[1, len(social_rows)],
            hspace=0.12,
            wspace=0.08,
        )
    else:
        grid = GridSpec(1, 2, figure=figure, wspace=0.08)

    center_axes: list[Axes] = []
    social_axes: list[Axes] = []
    for column, mode in enumerate(modes):
        subset = frame[frame["mode"] == mode]
        color = COLORS[mode.upper()]
        if has_center:
            center_axis = figure.add_subplot(
                grid[0, column],
                sharex=center_axes[0] if center_axes else None,
                sharey=center_axes[0] if center_axes else None,
            )
            _draw_coefficients(
                center_axis,
                subset,
                [(CENTER_PREDICTOR, r"$\bf{nEMD\ center}$")],
                color,
            )
            center_axis.set_title(f"{mode.upper()}  (adj. $R^2$ = {r_squared[mode]:.3f})")
            center_axes.append(center_axis)

        social_axis = figure.add_subplot(
            grid[1 if has_center else 0, column],
            sharex=social_axes[0] if social_axes else None,
            sharey=social_axes[0] if social_axes else None,
        )
        _draw_coefficients(social_axis, subset, social_rows, color)
        social_axis.set_xlabel(xlabel)
        if not has_center:
            social_axis.set_title(f"{mode.upper()}  (adj. $R^2$ = {r_squared[mode]:.3f})")
        social_axes.append(social_axis)

    for axis in center_axes[1:] + social_axes[1:]:
        axis.tick_params(labelleft=False)
    if has_center:
        for axis in center_axes:
            axis.tick_params(labelbottom=True)
            axis.set_xlabel(xlabel)

    return _save(figure, paths.figures / stem)


def generate_figures(
    paths: RunPaths,
    data: PreparedData,
    distances: pd.DataFrame,
    quality: pd.DataFrame,
    regression_fit: pd.DataFrame,
    social_coefficients: pd.DataFrame,
    full_coefficients: pd.DataFrame,
    standardized_coefficients: pd.DataFrame,
) -> list[Path]:
    paths.figures.mkdir(parents=True, exist_ok=True)
    generated = _density_quality(paths, distances, quality)
    generated.extend(_distance_comparison(paths, distances))
    generated.extend(_mds(paths, data, distances))
    generated.extend(
        _coefficient_figure(
            paths,
            social_coefficients,
            regression_fit,
            fit_model="social",
            stem="fig_social_coefficients",
            xlabel="Regression coefficient — outcome log1p(nEMD)",
        )
    )
    generated.extend(
        _coefficient_figure(
            paths,
            standardized_coefficients,
            regression_fit,
            fit_model="full",
            stem="fig_full_standardized_coefficients",
            xlabel="Coefficient per 1 SD of predictor — outcome log1p(nEMD)",
        )
    )
    generated.extend(
        _coefficient_figure(
            paths,
            full_coefficients,
            regression_fit,
            fit_model="full",
            stem="fig_full_coefficients",
            xlabel="Regression coefficient — outcome log1p(nEMD)",
        )
    )
    return generated
