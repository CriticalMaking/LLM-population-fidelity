"""Cross-culture and cross-model comparison figures and reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd

from machine_bias_reproduction.analysis import _load_source
from machine_bias_reproduction.config import FIGURES_ROOT, OUTPUTS_ROOT, paths_for
from machine_bias_reproduction.data import PreparedData
from machine_bias_reproduction.figures import (
    MDS_COLORS,
    NEMD_BREAKS,
    _annotate_quality_bands,
    _classical_mds,
    _log_nemd_density,
    _save,
)
from machine_bias_reproduction.metrics import (
    DISTANCES,
    QUALITY_LABELS,
    nemd,
    pairwise_nemd_matrix,
)
from machine_bias_reproduction.questions import Question, resolve_question

from . import capacity as capacity_module
from .matching import MATCHED_CULTURES, WVS_COUNTRIES, countries_for, country_of, restrict
from .registry import CULTURE_MODELS, PREFLIGHT_MIN_MASS, CultureModel, csv_stems

matplotlib.use("Agg")
from matplotlib import pyplot as plt

FloatArray = npt.NDArray[np.float64]

CULTURE_ROOT = OUTPUTS_ROOT / "culture"
CULTURE_FIGURES = FIGURES_ROOT / "culture"
MODES = ("NTP", "FA")
QUALITY_PALETTE = ("#1b9e77", "#66c2a5", "#ffd92f", "#fc8d62", "#d73027")
OUTCOME_PALETTE = {"valid": "#1b9e77", "invalid": "#fc8d62", "failed": "#d73027"}

CULTURE_COLORS = {
    "arabic": "#1b9e77",
    "bengali": "#d95f02",
    "chinese": "#7570b3",
    "english": "#e7298a",
    "german": "#66a61e",
    "korean": "#e6ab02",
    "portuguese": "#a6761d",
    "spanish": "#666666",
    "turkish": "#1f78b4",
}
"""Nine qualitative hues, one per culture, stable across every panel."""

REFERENCE_COLOR = "#000000"

MDS_REFERENCES: tuple[tuple[str, str], ...] = (
    ("Mixtral archived", "archived"),
    ("Mixtral fresh", "fresh"),
)
"""The two Mixtral series every matched MDS panel draws beside the adapter.

Both are references rather than controls: Mixtral is a different base model, so a
gap between it and an adapter confounds the base model with the culture tuning.
Drawing the archived and the freshly re-run series together shows how much of any
gap is merely run-to-run noise.
"""

MDS_SERIES_COLORS = {"Mixtral archived": REFERENCE_COLOR, "Mixtral fresh": "#c8c8c8"}
MDS_SERIES_MARKERS = {"Mixtral archived": "o", "Mixtral fresh": "s"}

MDS_ADAPTER_COLORS = {"spanish": "#1f78b4"}
"""Adapter hues that override ``CULTURE_COLORS`` in the matched MDS plate only.

This plate spends black and grey on the two Mixtral references, so an adapter
drawn in a grey cannot be told from the references it is being compared against.
``CULTURE_COLORS["spanish"]`` is ``#666666``, so Spanish takes a blue here. Every
other culture keeps its usual hue, and ``CULTURE_COLORS`` itself is untouched --
the nine-hue convention still holds everywhere else.
"""


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.is_file() else None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as stream:
        payload: dict[str, Any] = json.load(stream)
    return payload


def _load_model_frames(
    model: CultureModel,
    cultures: list[str],
    question: Question,
) -> dict[str, dict[str, Any]]:
    """Load every started run's tables for one model, keyed by culture.

    A run with no distances is still loaded when it has a capacity table: an
    adapter that answered nothing has no nEMD to plot but is exactly the case
    the capacity panel exists to report.
    """
    loaded: dict[str, dict[str, Any]] = {}
    for culture in cultures:
        paths = paths_for(f"culture/{model.key}/{culture}", question.var)
        distances = _read(paths.outputs / "subpopulation_distances.csv")
        capacity = capacity_module.read(paths)
        if distances is None and capacity is None:
            continue
        manifest = _read_json(paths.outputs / "run_manifest.json")
        loaded[culture] = {
            "distances": distances,
            "capacity": capacity,
            "quality": _read(paths.outputs / "quality_bands.csv"),
            "responses": _read(paths.outputs / "response_distributions.csv"),
            "summary": _read(paths.outputs / "summary_metrics.csv"),
            "fit": _read(paths.outputs / "regression_fit.csv"),
            "coverage": manifest.get("coverage", {}),
            "preflight": _preflight(paths.outputs / "inference_manifest.json"),
        }
    return loaded


def _preflight(path: Path) -> dict[str, Any]:
    """Return one run's pre-flight probe, defaulting to 'not measured'."""
    probe = _read_json(path).get("preflight") or {}
    return {"mean_mass": probe.get("mean_mass"), "informative": probe.get("informative")}


def _valid_rate(tables: dict[str, Any]) -> float | None:
    """Return the share of prompts answered in the paper's format, over both modes."""
    capacity = tables.get("capacity")
    if capacity is None or capacity.empty:
        return None
    prompts = float(capacity["prompts"].sum())
    return float(capacity["valid"].sum()) / prompts if prompts else 0.0


def _is_flagged(tables: dict[str, Any]) -> bool:
    """Return whether a run answered too rarely for its nEMD to be a measurement."""
    rate = _valid_rate(tables)
    if rate is not None:
        return rate < PREFLIGHT_MIN_MASS
    return tables.get("preflight", {}).get("informative") is False


def _has_distances(tables: dict[str, Any]) -> bool:
    return tables.get("distances") is not None


def _reference_distances(question: Question) -> pd.DataFrame | None:
    """Return the archived Mixtral subpopulation distances, if analysed."""
    return _read(paths_for("archived", question.var).outputs / "subpopulation_distances.csv")


def _series(frames: dict[str, dict[str, Any]], mode: str, column: str = "nEMD") -> dict[str, Any]:
    values: dict[str, Any] = {}
    for culture, tables in frames.items():
        if not _has_distances(tables):
            continue
        distances = tables["distances"]
        selected = distances.loc[distances["method"] == mode, column]
        if not selected.empty:
            values[culture] = selected.to_numpy(dtype=np.float64)
    return values


def _draw_culture_densities(
    axis: Any,
    frames: dict[str, dict[str, Any]],
    series: dict[str, Any],
    *,
    column: str = "nEMD",
    reference: np.ndarray | None = None,
) -> None:
    for culture, values in series.items():
        if not np.any(values > 0):
            continue
        grid, density = _log_nemd_density(values)
        flagged = _is_flagged(frames[culture])
        axis.plot(
            grid,
            density,
            label=f"{culture} (low mass)" if flagged else culture,
            color=CULTURE_COLORS[culture],
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


def _density_figure(
    frames: dict[str, dict[str, Any]],
    question: Question,
    destination: Path,
) -> list[Path]:
    """Overlay each culture's nEMD density against the paper's Mixtral baseline."""
    reference = _reference_distances(question)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for axis, mode in zip(axes, MODES, strict=True):
        baseline = None
        if reference is not None:
            baseline = reference.loc[reference["method"] == mode, "nEMD"].to_numpy(np.float64)
        _draw_culture_densities(axis, frames, _series(frames, mode), reference=baseline)
        axis.set(xlabel="nEMD (log scale)", title=mode)
    axes[0].set_ylabel("Density (per natural-log unit)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8, ncol=2)
    figure.tight_layout()
    return _save(figure, destination / "fig_culture_nemd_density")


def _distance_density_figure(
    frames: dict[str, dict[str, Any]],
    destination: Path,
) -> list[Path]:
    """Draw one density panel per distance measure, per generation strategy."""
    names = [name for name in DISTANCES]
    figure, axes = plt.subplots(
        len(MODES),
        len(names),
        figsize=(4.2 * len(names), 4.0 * len(MODES)),
        squeeze=False,
    )
    for row, mode in enumerate(MODES):
        for column, name in enumerate(names):
            axis = axes[row][column]
            _draw_culture_densities(axis, frames, _series(frames, mode, name), column=name)
            axis.set(xlabel=f"{name} (log scale)", title=f"{mode} — {name}")
        axes[row][0].set_ylabel("Density (per natural-log unit)")
    axes[0][0].legend(frameon=False, loc="upper left", fontsize=7, ncol=2)
    figure.tight_layout()
    return _save(figure, destination / "fig_culture_distance_density")


def _capacity_figure(
    frames: dict[str, dict[str, Any]],
    destination: Path,
) -> list[Path]:
    """Chart how often each adapter answered the paper's prompt at all.

    Left: the share of generations the paper's acceptance rules would take.
    Right: the mass placed on any valid answer token — the same question asked
    of the distribution rather than of the sampled text.
    """
    rows: list[dict[str, Any]] = []
    for culture, tables in frames.items():
        capacity = tables.get("capacity")
        if capacity is None or capacity.empty:
            continue
        for _, row in capacity.iterrows():
            rows.append(
                {
                    "culture": culture,
                    "mode": str(row["mode"]),
                    "valid": float(row["valid"]),
                    "invalid": float(row["invalid"]),
                    "failed": float(row["failed"]),
                    "prompts": float(row["prompts"]),
                    "mean_valid_mass": row.get("mean_valid_mass"),
                }
            )
    frame = pd.DataFrame(rows)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))

    if not frame.empty:
        order = sorted(frame["culture"].unique())
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
            color=[CULTURE_COLORS[culture] for culture in order],
            height=0.6,
        )
        axes[1].axvline(
            PREFLIGHT_MIN_MASS,
            color="#333333",
            linestyle=":",
            linewidth=1.2,
            label=f"informative threshold ({PREFLIGHT_MIN_MASS:.2f})",
        )
        axes[1].set_xscale("log")
        axes[1].set_yticks(positions, order)
        axes[1].set_ylim(len(order) - 0.5, -0.5)
        axes[1].set(xlabel="Mean valid-answer mass (log scale)", title="B. NTP valid-answer mass")
        axes[1].legend(frameon=False, fontsize=8, loc="lower right")

    figure.tight_layout()
    return _save(figure, destination / "fig_culture_capacity")


def _quality_figure(
    frames: dict[str, dict[str, Any]],
    destination: Path,
    stem: str = "fig_culture_quality_bands",
) -> list[Path]:
    rows: list[dict[str, Any]] = []
    for culture, tables in frames.items():
        quality = tables.get("quality")
        if quality is None:
            continue
        for mode in MODES:
            for _, row in quality[quality["method"] == mode].iterrows():
                rows.append(
                    {
                        "culture": culture,
                        "mode": mode,
                        "quality": row["quality"],
                        "percent": float(row["percent"]),
                    }
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


def _ranking_figure(
    frames: dict[str, dict[str, Any]],
    question: Question,
    destination: Path,
    stem: str = "fig_culture_ranking",
) -> list[Path]:
    rows: list[dict[str, Any]] = []
    for culture, tables in frames.items():
        if not _has_distances(tables):
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
    reference = _reference_distances(question)
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
        if not _has_distances(tables):
            continue
        subset = tables["distances"]
        subset = subset[subset["method"] == mode].copy()
        if subset.empty:
            continue
        subset["country"] = country_of(subset["subpopulation"])
        means = subset.groupby("country")["nEMD"].mean()
        rows[culture] = {country: float(means.get(country, np.nan)) for country in WVS_COUNTRIES}
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=list(WVS_COUNTRIES))


def _country_heatmap(frames: dict[str, dict[str, Any]], destination: Path) -> list[Path]:
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


def _response_shift(
    frames: dict[str, dict[str, Any]],
    question: Question,
    destination: Path,
) -> list[Path]:
    columns = list(question.answer_columns)
    labels = [f"{code}. {name}" for code, name in zip(columns, question.wvs_labels, strict=True)]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    wvs: np.ndarray | None = None
    for axis, mode in zip(axes, MODES, strict=True):
        cultures = sorted(
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
                color=CULTURE_COLORS[culture],
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


def _matched_frames(frames: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Restrict each matched culture to the subpopulations of its own countries.

    Only ``english``, ``german`` and ``spanish`` survive: the other six cultures
    have no WVS respondent block, so there is nothing to restrict them to.
    """
    matched: dict[str, dict[str, Any]] = {}
    for culture in MATCHED_CULTURES:
        tables = frames.get(culture)
        if tables is None or not _has_distances(tables):
            continue
        restricted = restrict(tables["distances"], culture)
        if restricted.empty:
            continue
        matched[culture] = {**tables, "distances": restricted}
    return matched


def _matched_table(matched: dict[str, dict[str, Any]], model: CultureModel) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for culture, tables in matched.items():
        distances = tables["distances"]
        for mode in MODES:
            values = distances.loc[distances["method"] == mode, "nEMD"].to_numpy(np.float64)
            if not values.size:
                continue
            record: dict[str, Any] = {
                "model_key": model.key,
                "model_label": model.label,
                "culture": culture,
                "countries": "; ".join(countries_for(culture)),
                "mode": mode,
                "subpopulations": int(values.size),
                "mean_nEMD": float(values.mean()),
                "median_nEMD": float(np.median(values)),
            }
            for name in DISTANCES:
                if name == "nEMD" or name not in distances.columns:
                    continue
                column = distances.loc[distances["method"] == mode, name].to_numpy(np.float64)
                record[f"mean_{name}"] = float(column.mean())
            rows.append(record)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["mode", "mean_nEMD"]).reset_index(drop=True)


def _matched_figures(
    model: CultureModel,
    frames: dict[str, dict[str, Any]],
    question: Question,
    destination: Path,
) -> tuple[list[Path], pd.DataFrame]:
    matched = _matched_frames(frames)
    if not matched:
        return [], pd.DataFrame()
    destination.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for axis, mode in zip(axes, MODES, strict=True):
        _draw_culture_densities(axis, matched, _series(matched, mode))
        axis.set(xlabel="nEMD (log scale)", title=mode)
    axes[0].set_ylabel("Density (per natural-log unit)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    figure.tight_layout()
    figures.extend(_save(figure, destination / "fig_culture_matched_density"))
    figures.extend(_distance_density_figure(matched, destination))
    figures.extend(_ranking_figure(matched, question, destination, "fig_culture_matched_ranking"))
    return figures, _matched_table(matched, model)


def _comparison_table(model: CultureModel, frames: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for culture, tables in frames.items():
        coverage = tables.get("coverage") or {}
        rate = _valid_rate(tables)
        base: dict[str, Any] = {
            "model_key": model.key,
            "model_label": model.label,
            "culture": culture,
            "valid_answer_rate": rate,
            "valid_answer_mass": tables.get("preflight", {}).get("mean_mass"),
            "informative": (not _is_flagged(tables)) if rate is not None else None,
            "subpopulations_retained": coverage.get("subpopulations_retained"),
            "subpopulations_dropped": coverage.get("subpopulations_dropped"),
        }
        if not _has_distances(tables):
            rows.append({**base, "mode": None})
            continue
        distances = tables["distances"]
        for mode in MODES:
            selected = distances[distances["method"] == mode]
            values = selected["nEMD"].to_numpy(np.float64)
            if not values.size:
                continue
            record: dict[str, Any] = {
                **base,
                "mode": mode,
                "subpopulations": int(values.size),
                "mean_nEMD": float(values.mean()),
                "median_nEMD": float(np.median(values)),
                "p90_nEMD": float(np.percentile(values, 90)),
            }
            for name in DISTANCES:
                if name != "nEMD" and name in selected.columns:
                    record[f"mean_{name}"] = float(selected[name].to_numpy(np.float64).mean())
            summary = tables.get("summary")
            if summary is not None:
                overall = summary[
                    (summary["metric"] == "overall_nEMD") & (summary["method"] == mode)
                ]
                if not overall.empty:
                    record["overall_nEMD"] = float(overall.iloc[0]["value"])
            fit = tables.get("fit")
            if fit is not None:
                selected_fit = fit[(fit["mode"] == mode.lower()) & (fit["model"] == "social")]
                if not selected_fit.empty:
                    record["social_adj_r_squared"] = float(
                        selected_fit.iloc[0]["adjusted_r_squared"]
                    )
            rows.append(record)
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["mode", "mean_nEMD"], na_position="last")
        .reset_index(drop=True)
    )


def _cross_model_figure(tables: dict[str, pd.DataFrame], destination: Path) -> list[Path]:
    combined = pd.concat(tables.values(), ignore_index=True)
    combined = combined[combined["mode"].notna()]
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    order = sorted(combined["culture"].unique())
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


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No completed runs._\n"
    display = frame.copy()
    for column in display.columns:
        if display[column].dtype.kind == "f":
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )
    header = "| " + " | ".join(display.columns) + " |"
    rule = "| " + " | ".join("---" for _ in display.columns) + " |"
    body = "\n".join(
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        for row in display.itertuples(index=False)
    )
    return f"{header}\n{rule}\n{body}\n"


def _capacity_section(frames: dict[str, dict[str, Any]]) -> str:
    """Report answer-format compliance before any distance is quoted."""
    rows: list[dict[str, Any]] = []
    for culture, tables in sorted(frames.items()):
        capacity = tables.get("capacity")
        if capacity is None or capacity.empty:
            continue
        for _, row in capacity.iterrows():
            rows.append(
                {
                    "culture": culture,
                    "mode": row["mode"],
                    "prompts": int(row["prompts"]),
                    "valid": int(row["valid"]),
                    "invalid": int(row["invalid"]),
                    "failed": int(row["failed"]),
                    "valid_rate": float(row["valid_rate"]),
                    "mean_valid_mass": row.get("mean_valid_mass"),
                }
            )
    if not rows:
        return "_Capacity has not been measured for any run._\n"
    frame = pd.DataFrame(rows)
    degenerate = sorted(set(frame.loc[frame["valid_rate"] < PREFLIGHT_MIN_MASS, "culture"]))
    note = ""
    if degenerate:
        note = (
            "\n> **Could not answer the paper's prompt:** "
            + ", ".join(f"`{name}`" for name in degenerate)
            + f".\n> Under {PREFLIGHT_MIN_MASS:.0%} of prompts produced an answer the paper's "
            "own acceptance rules would take. Any nEMD reported for these runs describes the "
            "few answers that survived, not the adapter's alignment. Rejected generations are "
            "in `capacity_examples.jsonl`.\n"
        )
    return _markdown_table(frame) + note


def _model_report(
    model: CultureModel,
    question: Question,
    table: pd.DataFrame,
    frames: dict[str, dict[str, Any]],
    matched: pd.DataFrame,
    figures: list[Path],
    destination: Path,
) -> Path:
    lines = [
        f"# {model.label} — culture-finetuned {question.label.lower()} reproduction",
        "",
        f"Base model: `{model.base_model_id}`  ",
        f"Quantization: `{model.quantization or 'none'}`  ",
        f"Question: `{question.var}` ({question.levels} ordered answers)  ",
        f"Cultures analysed: {len(frames)}",
        "",
        "Prompts are the paper's, verbatim. The adapters carry LoRA on the text attention",
        "projections only; the vision and audio towers keep their pretrained weights and",
        "remain part of the loaded model.",
        "",
        "## Answer-format capacity",
        "",
        _capacity_section(frames),
        "## Culture comparison",
        "",
        _markdown_table(table),
        "## Culture-matched subsets",
        "",
        "Restricted to subpopulations whose respondents live in a country where the",
        "fine-tuning language is dominant. Only english, german and spanish have one;",
        "Russia has respondents but no russian adapter.",
        "",
        _markdown_table(matched),
        "## Figures",
        "",
    ]
    lines.extend(
        f"- `{path.relative_to(destination.parent.parent.parent)}`"
        for path in figures
        if path.suffix == ".png"
    )
    lines.extend(
        [
            "",
            "## Reading these numbers",
            "",
            "`mean_nEMD` is the average normalized Earth-Mover's distance between a model's",
            "answer distribution and the WVS distribution, over the subpopulations that",
            "retained enough valid answers to score; lower is closer. Read the capacity table",
            "first: a distance computed over a handful of surviving answers is not comparable",
            "with one computed over all of them. The paper's Mixtral checkpoints are not",
            "applied here — they are claims about the paper's own model.",
            "",
        ]
    )
    report = destination / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def _combined_report(
    question: Question,
    tables: dict[str, pd.DataFrame],
    figures: list[Path],
    destination: Path,
) -> Path:
    combined = pd.concat(tables.values(), ignore_index=True) if tables else pd.DataFrame()
    lines = [
        f"# Culture-finetuned MLLM {question.label.lower()} reproduction",
        "",
        f"Nine culture LoRA adapters per base model, question `{question.var}`, scored",
        "against the same WVS subpopulations the Mixtral reproduction uses and prompted",
        "with the paper's own prompt.",
        "",
        "## All runs",
        "",
        _markdown_table(
            combined.sort_values(["mode", "mean_nEMD"], na_position="last").reset_index(drop=True)
            if not combined.empty
            else combined
        ),
        "## Figures",
        "",
    ]
    lines.extend(
        f"- `{path.relative_to(CULTURE_FIGURES.parent.parent)}`"
        for path in figures
        if path.suffix == ".png"
    )
    lines.extend(["", "## Per-model reports", ""])
    lines.extend(f"- `outputs/culture/{key}/{question.var}/REPORT.md`" for key in tables)
    lines.append("")
    report = destination / "REPORT.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def _mds_series(
    model: CultureModel,
    culture: str,
    question: Question,
) -> dict[str, PreparedData]:
    """Load the adapter and both Mixtral references for one culture.

    A series whose consolidated CSVs are missing is dropped rather than raising,
    so an unfinished cell of the sweep costs its own panel and nothing else.
    """
    wanted: list[tuple[str, str, tuple[str, str] | None]] = [
        (label, source, None) for label, source in MDS_REFERENCES
    ]
    wanted.append(
        (culture, f"culture/{model.key}/{culture}", csv_stems(model.key, culture, question))
    )
    loaded: dict[str, PreparedData] = {}
    for label, source, stems in wanted:
        try:
            loaded[label] = _load_source(source, paths_for(source, question.var), question, stems)
        except (FileNotFoundError, ValueError):
            continue
    return loaded


def _matched_mds_embedding(
    series: dict[str, PreparedData],
    culture: str,
    question: Question,
) -> tuple[pd.Index, FloatArray, dict[tuple[str, str], FloatArray]] | None:
    """Embed every series of one culture in a single shared MDS space.

    The subpopulations are the intersection of what all three series retained,
    restricted to the countries where this culture's language is dominant, so
    the adapter and both Mixtral series are scored on *identical* respondents
    and a difference between the clouds is the model rather than the sample.

    Following ``6-results.R:861-863`` -- as ``figures._mds`` does -- the WVS
    block is repeated once per model block before the distance matrix is
    computed, so the survey carries the same weight as the models it is being
    compared against. One embedding covers both generation modes, which is what
    makes the two panels of a row comparable.
    """
    shared: pd.Index | None = None
    for prepared in series.values():
        shared = prepared.names if shared is None else shared.intersection(prepared.names)
    if shared is None or shared.empty:
        return None
    frame = pd.DataFrame({"subpopulation": shared})
    names = pd.Index(restrict(frame, culture)["subpopulation"])
    if names.empty:
        return None

    columns = list(question.answer_columns)
    order = [(label, mode) for label in series for mode in MODES]
    props = {
        (label, mode): (series[label].ntp_props if mode == "NTP" else series[label].fa_props).loc[
            names, columns
        ]
        for label, mode in order
    }
    wvs = next(iter(series.values())).wvs_props.loc[names, columns].to_numpy(dtype=np.float64)

    blocks = [wvs] * len(order)
    blocks.extend(props[key].to_numpy(dtype=np.float64) for key in order)
    coordinates = _classical_mds(pairwise_nemd_matrix(np.vstack(blocks)))

    count = len(names)
    model_coordinates = {
        key: coordinates[(len(order) + index) * count : (len(order) + index + 1) * count]
        for index, key in enumerate(order)
    }
    return names, coordinates[:count], model_coordinates


def _matched_mds_plate(
    model: CultureModel,
    culture: str,
    question: Question,
) -> tuple[list[Path], pd.DataFrame]:
    """Draw one culture's matched MDS plate, one panel per generation mode.

    The plate lands beside that run's other figures, in
    ``figures/culture/<model>/<culture>/<question>/``, so a culture's evidence is
    read in one place instead of being cropped out of a grid.
    """
    if culture not in MATCHED_CULTURES:
        return [], pd.DataFrame()
    series = _mds_series(model, culture, question)
    # The adapter itself has to be there. Without this guard an unfinished cell
    # still draws a plate titled with the culture but holding only the two
    # Mixtral references, which reads as a result about that culture.
    if culture not in series or len(series) < 2:
        return [], pd.DataFrame()
    embedded = _matched_mds_embedding(series, culture, question)
    if embedded is None:
        return [], pd.DataFrame()
    names, wvs_coordinates, model_coordinates = embedded

    destination = CULTURE_FIGURES / model.key / culture / question.var
    destination.mkdir(parents=True, exist_ok=True)
    columns = list(question.answer_columns)
    wvs_props = next(iter(series.values())).wvs_props.loc[names, columns]
    rows: list[dict[str, Any]] = []

    figure, axes = plt.subplots(1, len(MODES), figsize=(13, 5.4))
    for axis, mode in zip(axes, MODES, strict=True):
        axis.scatter(
            wvs_coordinates[:, 0],
            wvs_coordinates[:, 1],
            s=22,
            alpha=0.35,
            marker="^",
            label="WVS respondents",
            color=MDS_COLORS["WVS"],
            edgecolors="#8a8a2a",
            linewidths=0.3,
        )
        annotations: list[str] = []
        for label in series:
            coordinates = model_coordinates.get((label, mode))
            if coordinates is None:
                continue
            model_props = (
                series[label].ntp_props if mode == "NTP" else series[label].fa_props
            ).loc[names, columns]
            error = float(np.mean(nemd(wvs_props.to_numpy(), model_props.to_numpy())))
            adapter = label not in MDS_SERIES_COLORS
            axis.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                s=28 if adapter else 22,
                alpha=0.6 if adapter else 0.42,
                marker=MDS_SERIES_MARKERS.get(label, "o"),
                label=f"{label} adapter" if adapter else label,
                color=MDS_SERIES_COLORS.get(
                    label,
                    MDS_ADAPTER_COLORS.get(label, CULTURE_COLORS.get(label, "#e7298a")),
                ),
                edgecolors="white" if adapter else "none",
                linewidths=0.4 if adapter else 0.0,
                zorder=3 if adapter else 2,
            )
            annotations.append(f"{label} {error:.3f}")
            rows.append(
                {
                    "culture": culture,
                    "series": label,
                    "mode": mode,
                    "subpopulations": len(names),
                    "countries": "; ".join(countries_for(culture)),
                    "mean_nEMD": error,
                }
            )
        axis.set_aspect("equal")
        axis.set_title("Next-token probabilities" if mode == "NTP" else "Full answer", fontsize=12)
        axis.set_xlabel("Classical MDS dimension 1")
        # Say the sample out loud. Every plate here is restricted to this
        # culture's own countries, so these means are over the panel's own
        # subpopulations and are not the run's overall nEMD.
        axis.annotate(
            f"Mean nEMD over these {len(names)} subpopulations\n" + "  ·  ".join(annotations),
            xy=(0.03, 0.03),
            xycoords="axes fraction",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#666666"},
        )

    # Both panels come from one embedding, so they must share a scale: a cloud
    # that looks tighter has to be tighter, not zoomed.
    spread = np.vstack([wvs_coordinates, *model_coordinates.values()])
    margin = 0.05 * float(np.ptp(spread[:, 0]))
    for axis in axes:
        axis.set_xlim(spread[:, 0].min() - margin, spread[:, 0].max() + margin)
        axis.set_ylim(spread[:, 1].min() - margin, spread[:, 1].max() + margin)
    axes[0].set_ylabel("Classical MDS dimension 2")
    axes[0].legend(frameon=False, loc="upper left", fontsize=9, markerscale=1.6)
    # Bare plate: no figure title. Which culture, which countries and which
    # question this is belongs to the caption, as it does for every other
    # figure in this repository.
    figure.tight_layout()
    return _save(figure, destination / "fig_culture_mds_matched"), pd.DataFrame(rows)


def culture_mds(
    models: list[CultureModel],
    cultures: list[str],
    question: str | Question = "d_happy",
) -> dict[str, Any]:
    """Build one matched MDS plate per culture, plus the shared coordinate export.

    Each plate is written beside that run's other figures rather than into a
    single per-question grid, so a culture can be read on its own. Cultures whose
    run has not finished are skipped, which makes this safe to re-run while a
    sweep is still going.
    """
    outcome = resolve_question(question)
    produced: list[Path] = []
    tables: list[pd.DataFrame] = []
    for model in models:
        for culture in cultures:
            figures, table = _matched_mds_plate(model, culture, outcome)
            produced.extend(figures)
            if not table.empty:
                table.insert(0, "model_key", model.key)
                table.insert(1, "model_label", model.label)
                tables.append(table)

    export: Path | None = None
    if tables:
        root = CULTURE_ROOT / outcome.var
        root.mkdir(parents=True, exist_ok=True)
        export = root / "culture_mds_matched.csv"
        pd.concat(tables, ignore_index=True).to_csv(export, index=False)

    return {
        "question": outcome.var,
        "figures": [str(path) for path in produced],
        "table": str(export) if export else None,
    }


def compare_cultures(
    models: list[CultureModel],
    cultures: list[str],
    question: str | Question = "d_happy",
) -> dict[str, Any]:
    """Build every comparison figure, table and report from existing outputs."""
    outcome = resolve_question(question)
    tables: dict[str, pd.DataFrame] = {}
    produced: list[Path] = []
    per_model: dict[str, list[str]] = {}

    for model in models:
        frames = _load_model_frames(model, cultures, outcome)
        per_model[model.key] = sorted(frames)
        if not frames:
            continue
        destination = CULTURE_FIGURES / model.key / outcome.var
        destination.mkdir(parents=True, exist_ok=True)
        figures: list[Path] = []
        figures.extend(_capacity_figure(frames, destination))
        if any(_has_distances(tables_) for tables_ in frames.values()):
            figures.extend(_density_figure(frames, outcome, destination))
            figures.extend(_distance_density_figure(frames, destination))
            figures.extend(_quality_figure(frames, destination))
            figures.extend(_ranking_figure(frames, outcome, destination))
            figures.extend(_country_heatmap(frames, destination))
            figures.extend(_response_shift(frames, outcome, destination))
        matched_figures, matched_table = _matched_figures(
            model, frames, outcome, destination / "matched"
        )
        figures.extend(matched_figures)

        table = _comparison_table(model, frames)
        tables[model.key] = table
        output_directory = CULTURE_ROOT / model.key / outcome.var
        output_directory.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_directory / "culture_comparison.csv", index=False)
        if not matched_table.empty:
            matched_table["model_key"] = model.key
            matched_table["model_label"] = model.label
            matched_table.to_csv(output_directory / "culture_matched_comparison.csv", index=False)
        _model_report(model, outcome, table, frames, matched_table, figures, output_directory)
        produced.extend(figures)

    if len(tables) > 1:
        cross_model = CULTURE_FIGURES / outcome.var
        cross_model.mkdir(parents=True, exist_ok=True)
        produced.extend(_cross_model_figure(tables, cross_model))
    report: Path | None = None
    if tables:
        combined = pd.concat(tables.values(), ignore_index=True)
        root = CULTURE_ROOT / outcome.var
        root.mkdir(parents=True, exist_ok=True)
        combined.to_csv(root / "culture_comparison.csv", index=False)
        report = _combined_report(outcome, tables, produced, root)

    return {
        "question": outcome.var,
        "models": per_model,
        "figures": len(produced),
        "report": str(report) if report else None,
    }
