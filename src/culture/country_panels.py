from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from machine_bias_reproduction.plates import MUTED_INK, SEPARATOR, magnitude_colormap
from machine_bias_reproduction.questions import Question

from .density_panels import distance_density_figure, draw_culture_densities
from .matching import WVS_COUNTRIES, countries_for, country_of
from .palette import MODES
from .plate import ModeFigure, empty_panel, per_mode
from .ranking_panels import ranking_figure
from .registry import CultureModel
from .tables import has_distances, matched_frames, matched_table, series


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
            empty_panel(axis, f"no {mode} distances")
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

    return per_mode(build, "fig_culture_country_heatmap", destination, align_axis=None)


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
            empty_panel(axis, f"no {mode} distances")
            return figure, [axis]
        draw_culture_densities(axis, matched, values)
        axis.set_xlabel("nEMD (log scale)")
        axis.set_ylabel("Density (per natural-log unit)")
        axis.legend(loc="upper left")
        return figure, [axis]

    figures = list(per_mode(build, "fig_culture_matched_density", destination, align_axis="y"))
    figures.extend(distance_density_figure(matched, destination))
    figures.extend(ranking_figure(matched, question, destination, "fig_culture_matched_ranking"))
    return figures, matched_table(matched, model)
