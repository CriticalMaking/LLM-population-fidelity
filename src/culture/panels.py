from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from machine_bias_reproduction.figures import (
    NEMD_BREAKS,
    _annotate_quality_bands,
    _log_nemd_density,
    _save,
)
from machine_bias_reproduction.metrics import DISTANCES, QUALITY_LABELS
from machine_bias_reproduction.plates import (
    GRID,
    MUTED_INK,
    QUALITY_RAMP,
    SEPARATOR,
    STATUS_TONES,
    magnitude_colormap,
)
from machine_bias_reproduction.questions import Question

from .matching import WVS_COUNTRIES, countries_for, country_of
from .palette import (
    MODE_MARKERS,
    MODE_TONES,
    MODES,
    REFERENCE_TONES,
    arm_linestyle,
    arm_order,
    arm_tone,
    model_marker,
    model_tone,
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

MIXTRAL_INK = REFERENCE_TONES["Mixtral archived"].ink

ModeFigure = tuple[Any, list[Any]]


def _empty_panel(axis: Any, message: str) -> None:
    axis.set_axis_off()
    axis.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=10,
        color=MUTED_INK,
        transform=axis.transAxes,
    )


def _align(built: dict[str, ModeFigure], which: str | None) -> None:
    if which not in ("x", "y"):
        return
    groups = [axes for _, axes in built.values()]
    if not groups:
        return
    getter, setter = ("get_xlim", "set_xlim") if which == "x" else ("get_ylim", "set_ylim")
    for index in range(min(len(group) for group in groups)):
        axes = [group[index] for group in groups]
        limits = [getattr(axis, getter)() for axis in axes]
        low = min(limit[0] for limit in limits)
        high = max(limit[1] for limit in limits)
        for axis in axes:
            getattr(axis, setter)(low, high)


def _per_mode(
    build: Callable[[str], ModeFigure],
    stem: str,
    destination: Path,
    *,
    align: str | None = "x",
) -> list[Path]:
    built = {mode: build(mode) for mode in MODES}
    _align(built, align)
    produced: list[Path] = []
    for mode, (figure, _) in built.items():
        figure.tight_layout()
        produced.extend(_save(figure, destination / f"{stem}_{mode.lower()}"))
    return produced


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
            label=culture,
            color=arm_tone(culture).ink,
            linewidth=1.4,
            linestyle=(0, (1, 1.6)) if flagged else arm_linestyle(culture),
            alpha=0.45 if flagged else 1.0,
        )
    if reference is not None and reference.size:
        grid, density = _log_nemd_density(reference)
        axis.plot(
            grid,
            density,
            label="Mixtral archived",
            color=MIXTRAL_INK,
            linestyle="solid",
            linewidth=1.6,
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

    def build(mode: str) -> ModeFigure:
        figure, axis = plt.subplots(figsize=(9, 6))
        values = series(frames, mode)
        if not values:
            _empty_panel(axis, f"no {mode} distances")
            return figure, [axis]
        baseline = None
        if reference is not None:
            baseline = reference.loc[reference["method"] == mode, "nEMD"].to_numpy(np.float64)
        _draw_culture_densities(axis, frames, values, reference=baseline)
        axis.set_xlabel("nEMD (log scale)")
        axis.set_ylabel("Density (per natural-log unit)")
        axis.legend(loc="upper left", ncol=2)
        return figure, [axis]

    return _per_mode(build, "fig_culture_nemd_density", destination, align="y")


def distance_density_figure(
    frames: dict[str, dict[str, Any]],
    destination: Path,
) -> list[Path]:
    names = list(DISTANCES)

    def build(mode: str) -> ModeFigure:
        figure, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.0), squeeze=False)
        row = list(axes[0])
        for axis, name in zip(row, names, strict=True):
            values = series(frames, mode, name)
            if not values:
                _empty_panel(axis, f"no {mode} {name}")
                continue
            _draw_culture_densities(axis, frames, values, column=name)
            axis.set_xlabel(f"{name} (log scale)")
        row[0].set_ylabel("Density (per natural-log unit)")
        if series(frames, mode):
            row[0].legend(loc="upper left", ncol=2)
        return figure, row

    return _per_mode(build, "fig_culture_distance_density", destination, align="y")


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
            for outcome, status in STATUS_TONES.items():
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
                    color=status.ink if index == 0 else status.fill,
                    edgecolor=SEPARATOR,
                    linewidth=0.8,
                    label=f"{outcome} ({mode})",
                )
                left = left + share
        axes[0].set_yticks(positions, order)
        axes[0].set_ylim(len(order) - 0.5, -0.5)
        axes[0].set_xlim(0, 100)
        axes[0].set_xlabel("Prompts (%)")
        axes[0].legend(loc="lower right", ncol=2)

        ntp = frame[frame["mode"] == "NTP"].set_index("culture").reindex(order)
        masses = pd.to_numeric(ntp["mean_valid_mass"], errors="coerce").to_numpy(dtype=np.float64)
        axes[1].barh(
            positions,
            np.nan_to_num(masses),
            color=[arm_tone(culture).fill for culture in order],
            edgecolor=[arm_tone(culture).ink for culture in order],
            linewidth=0.7,
            height=0.6,
        )
        axes[1].axvline(
            PREFLIGHT_MIN_VALID_ANSWER_MASS,
            color=MUTED_INK,
            linestyle=(0, (4, 2)),
            linewidth=1.0,
            label=f"informative threshold ({PREFLIGHT_MIN_VALID_ANSWER_MASS:.2f})",
        )
        axes[1].set_xscale("log")
        axes[1].set_yticks(positions, order)
        axes[1].set_ylim(len(order) - 0.5, -0.5)
        axes[1].set_xlabel("Mean valid-answer mass (log scale)")
        axes[1].legend(loc="lower right")

    figure.tight_layout()
    return _save(figure, destination / "fig_culture_capacity")


def run_capacity_figure(
    capacity: pd.DataFrame,
    coverage: dict[str, Any],
    destination: Path,
) -> list[Path]:
    if capacity.empty:
        return []
    figure, axis = plt.subplots(figsize=(9, 4.0))
    modes = [str(mode) for mode in capacity["mode"]]
    positions = np.arange(len(modes))
    left = np.zeros(len(modes))
    prompts = capacity["prompts"].to_numpy(dtype=np.float64)
    for outcome, status in STATUS_TONES.items():
        share = 100.0 * capacity[outcome].to_numpy(dtype=np.float64) / prompts
        share = np.nan_to_num(share)
        axis.barh(
            positions,
            share,
            height=0.5,
            left=left,
            color=status.ink,
            edgecolor=SEPARATOR,
            linewidth=0.8,
            label=outcome,
        )
        left = left + share
    axis.set_yticks(positions, modes)
    axis.set_ylim(len(modes) - 0.5, -0.5)
    axis.set_xlim(0, 100)
    axis.set_xlabel("Prompts (%)")
    axis.legend(loc="lower right")

    retained = coverage.get("subpopulations_retained")
    total = coverage.get("subpopulations_total")
    parts = [f"{retained} of {total} subpopulations placed"]
    ntp = capacity[capacity["mode"] == "NTP"]
    if not ntp.empty:
        row = ntp.iloc[0]
        parts.append(f"NTP valid-answer mass {float(row['mean_valid_mass']):.3g}")
        distinct = row.get("distinct_distributions")
        if not pd.isna(distinct):
            parts.append(f"{int(distinct)} distinct answers over {int(row['prompts'])} prompts")
        ties = row.get("tied_answers")
        if not pd.isna(ties):
            parts.append(f"{100.0 * float(ties) / float(row['prompts']):.0f}% tied")
    axis.annotate(
        "  ·  ".join(parts),
        xy=(0.0, 1.02),
        xycoords="axes fraction",
        fontsize=9,
        color=MUTED_INK,
        va="bottom",
    )
    figure.tight_layout()
    return _save(figure, destination / "fig_run_capacity")


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

    def build(mode: str) -> ModeFigure:
        figure, axis = plt.subplots(figsize=(9, 6))
        subset = frame[frame["mode"] == mode] if not frame.empty else frame
        if subset.empty:
            _empty_panel(axis, f"no {mode} quality bands")
            return figure, [axis]
        pivot = subset.pivot(index="culture", columns="quality", values="percent")
        pivot = pivot.reindex(columns=list(QUALITY_LABELS)).sort_index()
        pivot.plot(
            kind="bar",
            stacked=True,
            ax=axis,
            color=list(QUALITY_RAMP),
            width=0.72,
            edgecolor=SEPARATOR,
            linewidth=0.8,
        )
        axis.set(xlabel="", ylabel="Subpopulations (%)", ylim=(0, 100))
        axis.tick_params(axis="x", rotation=45)
        axis.legend(list(QUALITY_LABELS), title="", bbox_to_anchor=(1.02, 1), loc="upper left")
        return figure, [axis]

    return _per_mode(build, stem, destination, align=None)


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
    reference = reference_distances(question)
    if frame.empty:
        order: list[str] = []
    else:
        ntp = frame[frame["mode"] == "NTP"]
        order = (
            ntp.sort_values("mean")["culture"].tolist()
            if not ntp.empty
            else sorted(frame["culture"].unique())
        )
    positions = {culture: index for index, culture in enumerate(order)}

    def build(mode: str) -> ModeFigure:
        figure, axis = plt.subplots(figsize=(9, 6))
        subset = frame[frame["mode"] == mode] if not frame.empty else frame
        if subset.empty:
            _empty_panel(axis, f"no {mode} distances")
            return figure, [axis]
        tone = MODE_TONES[mode]
        y = [positions[culture] for culture in subset["culture"]]
        axis.scatter(
            subset["mean"],
            y,
            label=f"{mode} mean",
            marker=MODE_MARKERS[mode],
            color=tone.fill,
            edgecolors=tone.ink,
            linewidths=0.8,
            s=46,
        )
        axis.scatter(subset["median"], y, label=f"{mode} median", marker="|", color=tone.ink, s=140)
        if reference is not None:
            baseline = reference.loc[reference["method"] == mode, "nEMD"]
            if not baseline.empty:
                axis.axvline(
                    float(baseline.mean()),
                    color=MIXTRAL_INK,
                    linestyle=(0, (4, 2)),
                    linewidth=1.0,
                    label="Mixtral archived",
                )
        axis.set_yticks(range(len(order)), order)
        axis.set_ylim(len(order) - 0.5, -0.5)
        axis.set_xlabel("Subpopulation nEMD (lower is closer to the WVS)")
        axis.grid(axis="x", color=GRID, linewidth=0.6)
        axis.legend()
        return figure, [axis]

    return _per_mode(build, stem, destination)


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
    matrices = {mode: _country_matrix(frames, mode).sort_index() for mode in MODES}
    finite = np.concatenate(
        [
            matrix.to_numpy(dtype=np.float64).ravel()
            for matrix in matrices.values()
            if not matrix.empty
        ]
        or [np.array([np.nan])]
    )
    measurable = bool(np.any(np.isfinite(finite)))
    low = float(np.nanmin(finite)) if measurable else None
    high = float(np.nanmax(finite)) if measurable else None
    midpoint = float(np.nanmedian(finite)) if measurable else 0.0

    def build(mode: str) -> ModeFigure:
        figure, axis = plt.subplots(figsize=(8, 6))
        matrix = matrices[mode]
        if matrix.empty:
            _empty_panel(axis, f"no {mode} distances")
            return figure, [axis]
        values = matrix.to_numpy(dtype=np.float64)
        image = axis.imshow(values, cmap=magnitude_colormap(), aspect="auto", vmin=low, vmax=high)
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
                    color=SEPARATOR if value > midpoint else MUTED_INK,
                    fontweight="bold" if matched else "normal",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="mean nEMD")
        return figure, [axis]

    return _per_mode(build, "fig_culture_country_heatmap", destination, align=None)


def response_shift(
    frames: dict[str, dict[str, Any]],
    question: Question,
    destination: Path,
) -> list[Path]:
    columns = list(question.answer_columns)
    labels = [f"{code}. {name}" for code, name in zip(columns, question.wvs_labels, strict=True)]
    cultures = arm_order(
        culture for culture, tables in frames.items() if tables["responses"] is not None
    )
    positions = np.arange(len(columns))
    width = 0.8 / max(len(cultures), 1)

    def build(mode: str) -> ModeFigure:
        figure, axis = plt.subplots(figsize=(9, 5.5))
        wvs: np.ndarray | None = None
        drawn = False
        for index, culture in enumerate(cultures):
            responses = frames[culture]["responses"]
            selected = responses[responses["method"] == mode]
            if selected.empty:
                continue
            drawn = True
            axis.bar(
                positions + index * width - 0.4 + width / 2,
                selected.iloc[0][columns].to_numpy(dtype=np.float64),
                width=width,
                label=culture,
                color=arm_tone(culture).fill,
                edgecolor=arm_tone(culture).ink,
                linewidth=0.7,
            )
            if wvs is None:
                wvs_row = responses[responses["method"] == "WVS"]
                if not wvs_row.empty:
                    wvs = wvs_row.iloc[0][columns].to_numpy(dtype=np.float64)
        if not drawn:
            _empty_panel(axis, f"no {mode} answers")
            return figure, [axis]
        if wvs is not None:
            for index, (position, value) in enumerate(zip(positions, wvs, strict=True)):
                axis.hlines(
                    value,
                    position - 0.44,
                    position + 0.44,
                    color=MIXTRAL_INK,
                    linewidth=1.4,
                    label="WVS" if index == 0 else "",
                )
        axis.set_xticks(positions, labels)
        axis.tick_params(axis="x", rotation=20)
        axis.set_ylabel("Share of answers")
        axis.legend(ncol=2)
        return figure, [axis]

    return _per_mode(build, "fig_culture_response_shift", destination, align="y")


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

    def build(mode: str) -> ModeFigure:
        figure, axis = plt.subplots(figsize=(9, 6))
        values = series(matched, mode)
        if not values:
            _empty_panel(axis, f"no {mode} distances")
            return figure, [axis]
        _draw_culture_densities(axis, matched, values)
        axis.set_xlabel("nEMD (log scale)")
        axis.set_ylabel("Density (per natural-log unit)")
        axis.legend(loc="upper left")
        return figure, [axis]

    figures = list(_per_mode(build, "fig_culture_matched_density", destination, align="y"))
    figures.extend(distance_density_figure(matched, destination))
    figures.extend(ranking_figure(matched, question, destination, "fig_culture_matched_ranking"))
    return figures, matched_table(matched, model)


def _is_low_capacity(row: pd.Series) -> bool:
    informative = row.get("informative")
    return informative is False or informative == 0


def cross_model_figure(
    tables: dict[str, pd.DataFrame],
    question: Question,
    destination: Path,
) -> list[Path]:
    combined = pd.concat(tables.values(), ignore_index=True)
    combined = combined[combined["mode"].notna()]
    order = arm_order(combined["culture"].unique())
    positions = {culture: index for index, culture in enumerate(order)}
    reference = reference_distances(question)

    def build(mode: str) -> ModeFigure:
        figure, axis = plt.subplots(figsize=(9, 6))
        drawn = False
        for index, (model_key, frame) in enumerate(tables.items()):
            subset = frame[frame["mode"] == mode]
            if subset.empty:
                continue
            drawn = True
            tone = model_tone(model_key, index)
            marker = model_marker(model_key, index)
            label = CULTURE_MODELS[model_key].label
            flagged = subset.apply(_is_low_capacity, axis=1)
            for low_capacity, group in subset.groupby(flagged):
                axis.scatter(
                    group["mean_nEMD"],
                    [positions[culture] for culture in group["culture"]],
                    label=f"{label} (low mass)" if low_capacity else label,
                    marker=marker,
                    s=52,
                    color="none" if low_capacity else tone.fill,
                    edgecolors=tone.ink,
                    linewidths=1.2 if low_capacity else 0.8,
                )
        if not drawn:
            _empty_panel(axis, f"no {mode} distances")
            return figure, [axis]
        if reference is not None:
            baseline = reference.loc[reference["method"] == mode, "nEMD"]
            if not baseline.empty:
                axis.axvline(
                    float(baseline.mean()),
                    color=MIXTRAL_INK,
                    linestyle=(0, (4, 2)),
                    linewidth=1.0,
                    label="Mixtral archived",
                )
        axis.set_yticks(range(len(order)), order)
        axis.set_ylim(len(order) - 0.5, -0.5)
        axis.set_xlabel("Subpopulation nEMD (lower is closer to the WVS)")
        axis.grid(axis="x", color=GRID, linewidth=0.6)
        axis.legend()
        return figure, [axis]

    return _per_mode(build, "fig_model_culture_comparison", destination)


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
