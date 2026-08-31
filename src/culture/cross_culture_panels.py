from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from machine_bias_reproduction.figures import classical_mds
from machine_bias_reproduction.plates import (
    MUTED_INK,
    SEPARATOR,
    SURVEY_TONE,
    Tone,
    magnitude_colormap,
)

from .palette import MIXTRAL_ARCHIVED, arm_marker, arm_order, arm_tone, model_marker, model_tone
from .plate import MIXTRAL_INK, RULE_DASH, ModeFigure, empty_panel, per_mode, rank_axis

SURVEY = "WVS"
MIN_MAP_ENTITIES = 3


def _entity_tone(entity: str) -> Tone:
    return SURVEY_TONE if entity == SURVEY else arm_tone(entity)


def _entity_marker(entity: str) -> str:
    return "^" if entity == SURVEY else arm_marker(entity)


def pairwise_heatmap(
    matrices: dict[str, pd.DataFrame],
    label: str,
    destination: Path,
    stem: str,
) -> list[Path]:
    masked: dict[str, np.ndarray] = {}
    for mode, matrix in matrices.items():
        values = matrix.to_numpy(dtype=np.float64, copy=True)
        np.fill_diagonal(values, np.nan)
        masked[mode] = values
    finite = np.concatenate(
        [values.ravel() for values in masked.values() if values.size] or [np.array([np.nan])]
    )
    measurable = bool(np.any(np.isfinite(finite)))
    low = float(np.nanmin(finite)) if measurable else None
    high = float(np.nanmax(finite)) if measurable else None
    midpoint = float(np.nanmedian(finite)) if measurable else 0.0

    def build(mode: str) -> ModeFigure:
        figure, axis = plt.subplots(figsize=(8, 6))
        matrix = matrices[mode]
        if matrix.empty:
            empty_panel(axis, f"fewer than 2 variants with {mode} runs")
            return figure, [axis]
        values = masked[mode]
        image = axis.imshow(values, cmap=magnitude_colormap(), aspect="auto", vmin=low, vmax=high)
        axis.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=45, ha="right")
        axis.set_yticks(range(matrix.shape[0]), matrix.index)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = values[row, column]
                if np.isnan(value):
                    continue
                emphasized = SURVEY in (str(matrix.index[row]), str(matrix.columns[column]))
                axis.text(
                    column,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=SEPARATOR if value > midpoint else MUTED_INK,
                    fontweight="bold" if emphasized else "normal",
                )
        axis.set_title(label)
        figure.colorbar(
            image, ax=axis, fraction=0.046, pad=0.04, label="mean nEMD between variants"
        )
        return figure, [axis]

    return per_mode(build, stem, destination, align_axis=None)


def league_figure(league: pd.DataFrame, destination: Path) -> list[Path]:
    if league.empty:
        return []
    order = arm_order(league["culture"].unique())
    positions = {culture: index for index, culture in enumerate(order)}

    def build(mode: str) -> ModeFigure:
        figure, axis = plt.subplots(figsize=(9, 6))
        subset = league[league["mode"] == mode]
        if subset.empty:
            empty_panel(axis, f"no {mode} distances")
            return figure, [axis]
        for index, (model_key, group) in enumerate(subset.groupby("model_key", sort=False)):
            tone = model_tone(model_key, index)
            marker = model_marker(model_key, index)
            label = group["model_label"].iloc[0]
            axis.scatter(
                group["mean_nEMD"],
                [positions[culture] for culture in group["culture"]],
                label=label,
                marker=marker,
                s=52,
                color=tone.fill,
                edgecolors=tone.ink,
                linewidths=0.8,
            )
            anchored = group[np.isfinite(group["base_mean_nEMD"])]
            if not anchored.empty:
                axis.scatter(
                    anchored["base_mean_nEMD"],
                    [positions[culture] for culture in anchored["culture"]],
                    label=f"{label} — base variant",
                    marker="|",
                    s=140,
                    color=tone.ink,
                )
        drawn = False
        for culture, value in subset.groupby("culture")["mixtral_mean_nEMD"].first().items():
            if not np.isfinite(value):
                continue
            axis.scatter(
                [value],
                [positions[culture]],
                label="" if drawn else f"{MIXTRAL_ARCHIVED} — same countries",
                marker="|",
                s=140,
                color=MIXTRAL_INK,
            )
            drawn = True
        rank_axis(axis, order)
        axis.set_xlabel("Home-country subpopulation nEMD (lower is closer to the WVS)")
        return figure, [axis]

    return per_mode(build, "fig_model_culture_matched_comparison", destination)


def head_to_head_figure(cells: pd.DataFrame, destination: Path) -> list[Path]:
    if cells.empty:
        return []
    culture_a = cells["culture_a"].iloc[0]
    culture_b = cells["culture_b"].iloc[0]
    countries = cells["countries"].iloc[0]

    def build(mode: str) -> ModeFigure:
        figure, axis = plt.subplots(figsize=(6.8, 6.8))
        subset = cells[cells["mode"] == mode]
        if subset.empty:
            empty_panel(axis, f"no {mode} distances")
            return figure, [axis]
        for index, (model_key, group) in enumerate(subset.groupby("model_key", sort=False)):
            tone = model_tone(model_key, index)
            axis.scatter(
                group["nEMD_a"],
                group["nEMD_b"],
                label=group["model_label"].iloc[0],
                marker=model_marker(model_key, index),
                s=30,
                alpha=0.75,
                color=tone.fill,
                edgecolors=tone.ink,
                linewidths=0.5,
            )
        axis.axline(
            (0, 0),
            slope=1,
            color=MUTED_INK,
            linestyle=RULE_DASH,
            linewidth=1.0,
            label="equal distance",
        )
        limit = 1.05 * float(max(subset["nEMD_a"].max(), subset["nEMD_b"].max()))
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.set_aspect("equal")
        axis.set_xlabel(f"{culture_a} variant · subpopulation nEMD ({countries})")
        axis.set_ylabel(f"{culture_b} variant · subpopulation nEMD ({countries})")
        mean_delta = float(subset["delta_nEMD"].mean())
        share = float((subset["delta_nEMD"] < 0).mean())
        axis.annotate(
            f"{mode} · {len(subset)} cells · mean Δ {mean_delta:+.3f} · "
            f"{culture_b} closer in {share:.0%}",
            xy=(0.0, 1.01),
            xycoords="axes fraction",
            fontsize=8.5,
            color=MUTED_INK,
            va="bottom",
        )
        axis.legend(loc="upper left")
        return figure, [axis]

    return per_mode(build, "fig_culture_head_to_head", destination, align_axis=None)


def culture_map(
    matrices: dict[str, pd.DataFrame],
    label: str,
    destination: Path,
    stem: str,
) -> list[Path]:
    def build(mode: str) -> ModeFigure:
        figure, axes = plt.subplots(1, 2, figsize=(12, 5.4))
        matrix = matrices[mode]
        if len(matrix) < MIN_MAP_ENTITIES:
            message = f"fewer than {MIN_MAP_ENTITIES} variants with {mode} runs"
            empty_panel(axes[0], message)
            empty_panel(axes[1], message)
            return figure, [axes[0], axes[1]]
        coordinates = classical_mds(matrix.to_numpy(np.float64))
        for position, entity in enumerate(matrix.index):
            tone = _entity_tone(str(entity))
            axes[0].scatter(
                coordinates[position, 0],
                coordinates[position, 1],
                marker=_entity_marker(str(entity)),
                s=90 if entity == SURVEY else 64,
                color=tone.fill,
                edgecolors=tone.ink,
                linewidths=0.8,
            )
            axes[0].annotate(
                str(entity),
                xy=(coordinates[position, 0], coordinates[position, 1]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8.5,
                color=MUTED_INK,
            )
        axes[0].set_aspect("equal")
        axes[0].set_xlabel("Classical MDS dimension 1")
        axes[0].set_ylabel("Classical MDS dimension 2")
        axes[0].annotate(
            f"{mode} · {len(matrix)} variants",
            xy=(0.0, 1.01),
            xycoords="axes fraction",
            fontsize=8.5,
            color=MUTED_INK,
            va="bottom",
        )
        axes[0].set_title(label)
        links = linkage(squareform(matrix.to_numpy(np.float64), checks=False), method="average")
        dendrogram(
            links,
            labels=list(matrix.index),
            ax=axes[1],
            color_threshold=0.0,
            above_threshold_color=MUTED_INK,
        )
        for text in axes[1].get_xticklabels():
            text.set_rotation(30)
            text.set_ha("right")
            text.set_color(_entity_tone(text.get_text()).ink)
        axes[1].set_ylabel("mean nEMD between variants")
        return figure, [axes[0], axes[1]]

    return per_mode(build, stem, destination, align_axis=None)
