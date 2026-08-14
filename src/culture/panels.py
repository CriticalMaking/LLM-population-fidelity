from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from machine_bias_reproduction.figures import (
    NEMD_BREAKS,
    _annotate_quality_bands,
    _log_nemd_density,
    _save,
)
from machine_bias_reproduction.metrics import DISTANCES, QUALITY_LABELS
from machine_bias_reproduction.questions import Question

from .matching import WVS_COUNTRIES, countries_for, country_of
from .palette import (
    MODES,
    OUTCOME_PALETTE,
    QUALITY_PALETTE,
    REFERENCE_COLOR,
    arm_order,
    culture_color,
)
from .registry import CULTURE_MODELS, PREFLIGHT_MIN_VALID_ANSWER_MASS, CultureModel
from .tables import (
    has_distances,
    is_flagged,
    matched_frames,
    matched_table,
    reference_distances,
    series,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def _draw_culture_densities(
    axis: Any,
    frames: dict[str, dict[str, Any]],
    values_by_culture: dict[str, Any],
    *,
    column: str = "nEMD",
    reference: np.ndarray | None = None,
) -> None:
    for culture, values in values_by_culture.items():
        if not np.any(values > 0):
            continue
        grid, density = _log_nemd_density(values)
        flagged = is_flagged(frames[culture])
        axis.plot(
            grid,
            density,
            label=f"{culture} (low mass)" if flagged else culture,
            color=culture_color(culture),
            linewidth=1.5,
            linestyle=":" if flagged else "solid",
            alpha=0.45 if flagged else 1.0,
        )
    if reference is not None and reference.size:
        grid, density = _log_nemd_density(reference)
        axis.plot(
            grid,
            density,
            label="Mixtral-8x7B (paper)",
            color=REFERENCE_COLOR,
            linestyle=(0, (6, 3)),
            linewidth=1.8,
        )
    axis.set_xscale("log")
    axis.set_ylim(bottom=0)
    if column == "nEMD":
        axis.set_xlim(0.01, 1.0)
        axis.set_xticks(NEMD_BREAKS, [f"{break_:.2f}" for break_ in NEMD_BREAKS])
        _annotate_quality_bands(axis)


def density_figure(
    frames: dict[str, dict[str, Any]],
    question: Question,
    destination: Path,
) -> list[Path]:
    reference = reference_distances(question)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for axis, mode in zip(axes, MODES, strict=True):
        baseline = None
        if reference is not None:
            baseline = reference.loc[reference["method"] == mode, "nEMD"].to_numpy(np.float64)
        _draw_culture_densities(axis, frames, series(frames, mode), reference=baseline)
        axis.set(xlabel="nEMD (log scale)", title=mode)
    axes[0].set_ylabel("Density (per natural-log unit)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8, ncol=2)
    figure.tight_layout()
    return _save(figure, destination / "fig_culture_nemd_density")


def distance_density_figure(
    frames: dict[str, dict[str, Any]],
    destination: Path,
) -> list[Path]:
    names = list(DISTANCES)
    figure, axes = plt.subplots(
        len(MODES),
        len(names),
        figsize=(4.2 * len(names), 4.0 * len(MODES)),
        squeeze=False,
    )
    for row, mode in enumerate(MODES):
        for column, name in enumerate(names):
            axis = axes[row][column]
            _draw_culture_densities(axis, frames, series(frames, mode, name), column=name)
            axis.set(xlabel=f"{name} (log scale)", title=f"{mode} — {name}")
        axes[row][0].set_ylabel("Density (per natural-log unit)")
    axes[0][0].legend(frameon=False, loc="upper left", fontsize=7, ncol=2)
    figure.tight_layout()
    return _save(figure, destination / "fig_culture_distance_density")


def capacity_figure(
    frames: dict[str, dict[str, Any]],
    destination: Path,
) -> list[Path]:
    rows: list[dict[str, Any]] = []
    for culture, tables in frames.items():
        capacity = tables.get("capacity")
        if capacity is None or capacity.empty:
            continue
        rows.extend(
            {
                "culture": culture,
                "mode": str(row["mode"]),
                "valid": float(row["valid"]),
                "invalid": float(row["invalid"]),
                "failed": float(row["failed"]),
                "prompts": float(row["prompts"]),
                "mean_valid_mass": row.get("mean_valid_mass"),
            }
            for _, row in capacity.iterrows()
        )
    frame = pd.DataFrame(rows)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))

    if not frame.empty:
        order = arm_order(frame["culture"].unique())
        positions = np.arange(len(order))
        height = 0.38
        for index, mode in enumerate(MODES):
            subset = frame[frame["mode"] == mode].set_index("culture").reindex(order)
            offset = (index - 0.5) * height
            left = np.zeros(len(order))
            for outcome, color in OUTCOME_PALETTE.items():
                share = (
                    100.0
                    * subset[outcome].to_numpy(dtype=np.float64)
                    / subset["prompts"].to_numpy(dtype=np.float64)
                )
                share = np.nan_to_num(share)
                axes[0].barh(
                    positions + offset,
                    share,
                    height=height,
                    left=left,
                    color=color,
                    edgecolor="white",
                    linewidth=0.4,
                    label=f"{outcome} ({mode})" if index == 0 else None,
                    alpha=1.0 if index == 0 else 0.65,
                )
                left = left + share
        axes[0].set_yticks(positions, order)
        axes[0].set_ylim(len(order) - 0.5, -0.5)
        axes[0].set_xlim(0, 100)
        axes[0].set(xlabel="Prompts (%)", title="A. Answers in the paper's format")
        axes[0].legend(frameon=False, fontsize=8, loc="lower right")

        ntp = frame[frame["mode"] == "NTP"].set_index("culture").reindex(order)
        masses = pd.to_numeric(ntp["mean_valid_mass"], errors="coerce").to_numpy(dtype=np.float64)
        axes[1].barh(
            positions,
            np.nan_to_num(masses),
            color=[culture_color(culture) for culture in order],
            height=0.6,
        )
        axes[1].axvline(
            PREFLIGHT_MIN_VALID_ANSWER_MASS,
            color="#333333",
            linestyle=":",
            linewidth=1.2,
            label=f"informative threshold ({PREFLIGHT_MIN_VALID_ANSWER_MASS:.2f})",
        )
        axes[1].set_xscale("log")
        axes[1].set_yticks(positions, order)
        axes[1].set_ylim(len(order) - 0.5, -0.5)
        axes[1].set(xlabel="Mean valid-answer mass (log scale)", title="B. NTP valid-answer mass")
        axes[1].legend(frameon=False, fontsize=8, loc="lower right")

    figure.tight_layout()
    return _save(figure, destination / "fig_culture_capacity")


def quality_figure(
    frames: dict[str, dict[str, Any]],
    destination: Path,
    stem: str = "fig_culture_quality_bands",
) -> list[Path]:
    rows: list[dict[str, Any]] = []
    for culture, tables in frames.items():
        quality = tables.get("quality")
        if quality is None:
            continue
        rows.extend(
            {
                "culture": culture,
                "mode": mode,
                "quality": row["quality"],
                "percent": float(row["percent"]),
            }
            for mode in MODES
            for _, row in quality[quality["method"] == mode].iterrows()
        )
    frame = pd.DataFrame(rows)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for axis, mode in zip(axes, MODES, strict=True):
        subset = frame[frame["mode"] == mode] if not frame.empty else frame
        if subset.empty:
            axis.set_axis_off()
            continue
        pivot = subset.pivot(index="culture", columns="quality", values="percent")
        pivot = pivot.reindex(columns=list(QUALITY_LABELS)).sort_index()
        pivot.plot(kind="bar", stacked=True, ax=axis, color=list(QUALITY_PALETTE), width=0.78)
        axis.set(xlabel="", ylabel="Subpopulations (%)", title=mode, ylim=(0, 100))
        axis.tick_params(axis="x", rotation=45)
        axis.get_legend().remove()
    axes[1].legend(
        list(QUALITY_LABELS),
        title="",
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    figure.tight_layout()
    return _save(figure, destination / stem)


def ranking_figure(
    frames: dict[str, dict[str, Any]],
    question: Question,
    destination: Path,
    stem: str = "fig_culture_ranking",
) -> list[Path]:
    rows: list[dict[str, Any]] = []
    for culture, tables in frames.items():
        if not has_distances(tables):
            continue
        distances = tables["distances"]
        for mode in MODES:
            values = distances.loc[distances["method"] == mode, "nEMD"].to_numpy(np.float64)
            if values.size:
                rows.append(
                    {
                        "culture": culture,
                        "mode": mode,
                        "mean": float(values.mean()),
                        "median": float(np.median(values)),
                    }
                )
    frame = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(9, 6))
    if frame.empty:
        axis.set_axis_off()
        figure.tight_layout()
        return _save(figure, destination / stem)

    ntp = frame[frame["mode"] == "NTP"]
    order = (
        ntp.sort_values("mean")["culture"].tolist()
        if not ntp.empty
        else sorted(frame["culture"].unique())
    )
    positions = {culture: index for index, culture in enumerate(order)}
    offsets = {"NTP": -0.16, "FA": 0.16}
    markers = {"NTP": "o", "FA": "s"}
    colors = {"NTP": "#d95f02", "FA": "#7570b3"}
    for mode in MODES:
        subset = frame[frame["mode"] == mode]
        if subset.empty:
            continue
        y = [positions[culture] + offsets[mode] for culture in subset["culture"]]
        axis.scatter(
            subset["mean"], y, label=f"{mode} mean", marker=markers[mode], color=colors[mode], s=46
        )
        axis.scatter(
            subset["median"], y, label=f"{mode} median", marker="|", color=colors[mode], s=140
        )
    reference = reference_distances(question)
    if reference is not None:
        for mode in MODES:
            baseline = reference.loc[reference["method"] == mode, "nEMD"]
            if not baseline.empty:
                axis.axvline(
                    float(baseline.mean()),
                    color=colors[mode],
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.7,
                )
    axis.set_yticks(range(len(order)), order)
    axis.set_ylim(len(order) - 0.5, -0.5)
    axis.set_xlabel("Subpopulation nEMD (lower is closer to the WVS)")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    return _save(figure, destination / stem)


def _country_matrix(frames: dict[str, dict[str, Any]], mode: str) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for culture, tables in frames.items():
        if not has_distances(tables):
            continue
        subset = tables["distances"]
        subset = subset[subset["method"] == mode].copy()
        if subset.empty:
            continue
        subset["country"] = country_of(subset["subpopulation"])
        means = subset.groupby("country")["nEMD"].mean()
        rows[culture] = {country: float(means.get(country, np.nan)) for country in WVS_COUNTRIES}
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=list(WVS_COUNTRIES))


def country_heatmap(frames: dict[str, dict[str, Any]], destination: Path) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, mode in zip(axes, MODES, strict=True):
        matrix = _country_matrix(frames, mode).sort_index()
        if matrix.empty:
            axis.set_axis_off()
            continue
        values = matrix.to_numpy(dtype=np.float64)
        image = axis.imshow(values, cmap="viridis_r", aspect="auto")
        axis.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=45, ha="right")
        axis.set_yticks(range(matrix.shape[0]), matrix.index)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = values[row, column]
                if np.isnan(value):
                    continue
                matched = matrix.columns[column] in countries_for(str(matrix.index[row]))
                axis.text(
                    column,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > np.nanmedian(values) else "black",
                    fontweight="bold" if matched else "normal",
                )
        axis.set_title(mode)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="mean nEMD")
    figure.tight_layout()
    return _save(figure, destination / "fig_culture_country_heatmap")


def response_shift(
    frames: dict[str, dict[str, Any]],
    question: Question,
    destination: Path,
) -> list[Path]:
    columns = list(question.answer_columns)
    labels = [f"{code}. {name}" for code, name in zip(columns, question.wvs_labels, strict=True)]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    wvs: np.ndarray | None = None
    for axis, mode in zip(axes, MODES, strict=True):
        cultures = arm_order(
            culture for culture, tables in frames.items() if tables["responses"] is not None
        )
        width = 0.8 / max(len(cultures), 1)
        positions = np.arange(len(columns))
        for index, culture in enumerate(cultures):
            responses = frames[culture]["responses"]
            selected = responses[responses["method"] == mode]
            if selected.empty:
                continue
            axis.bar(
                positions + index * width - 0.4 + width / 2,
                selected.iloc[0][columns].to_numpy(dtype=np.float64),
                width=width,
                label=culture,
                color=culture_color(culture),
            )
            if wvs is None:
                wvs_row = responses[responses["method"] == "WVS"]
                if not wvs_row.empty:
                    wvs = wvs_row.iloc[0][columns].to_numpy(dtype=np.float64)
        if wvs is not None:
            for position, value in zip(positions, wvs, strict=True):
                axis.hlines(
                    value, position - 0.44, position + 0.44, color=REFERENCE_COLOR, linewidth=1.6
                )
        axis.set_xticks(positions, labels)
        axis.tick_params(axis="x", rotation=20)
        axis.set_title(mode)
    axes[0].set_ylabel("Share of answers")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    figure.tight_layout()
    return _save(figure, destination / "fig_culture_response_shift")


def matched_figures(
    model: CultureModel,
    frames: dict[str, dict[str, Any]],
    question: Question,
    destination: Path,
) -> tuple[list[Path], pd.DataFrame]:
    matched = matched_frames(frames)
    if not matched:
        return [], pd.DataFrame()
    destination.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for axis, mode in zip(axes, MODES, strict=True):
        _draw_culture_densities(axis, matched, series(matched, mode))
        axis.set(xlabel="nEMD (log scale)", title=mode)
    axes[0].set_ylabel("Density (per natural-log unit)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    figure.tight_layout()
    figures.extend(_save(figure, destination / "fig_culture_matched_density"))
    figures.extend(distance_density_figure(matched, destination))
    figures.extend(ranking_figure(matched, question, destination, "fig_culture_matched_ranking"))
    return figures, matched_table(matched, model)


def cross_model_figure(tables: dict[str, pd.DataFrame], destination: Path) -> list[Path]:
    combined = pd.concat(tables.values(), ignore_index=True)
    combined = combined[combined["mode"].notna()]
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    order = arm_order(combined["culture"].unique())
    positions = {culture: index for index, culture in enumerate(order)}
    markers = {key: marker for key, marker in zip(tables, ("o", "s", "^", "D"), strict=False)}
    for axis, mode in zip(axes, MODES, strict=True):
        for model_key, frame in tables.items():
            subset = frame[frame["mode"] == mode]
            if subset.empty:
                continue
            axis.scatter(
                subset["mean_nEMD"],
                [positions[culture] for culture in subset["culture"]],
                label=CULTURE_MODELS[model_key].label,
                marker=markers.get(model_key, "o"),
                s=52,
                alpha=0.85,
            )
        axis.set_yticks(range(len(order)), order)
        axis.set_ylim(len(order) - 0.5, -0.5)
        axis.set_xlabel("Mean subpopulation nEMD")
        axis.grid(axis="x", alpha=0.2)
        axis.set_title(mode)
    axes[0].legend(frameon=False, fontsize=9)
    figure.tight_layout()
    return _save(figure, destination / "fig_model_culture_comparison")


__all__ = [
    "capacity_figure",
    "country_heatmap",
    "cross_model_figure",
    "density_figure",
    "distance_density_figure",
    "matched_figures",
    "quality_figure",
    "ranking_figure",
    "response_shift",
]
