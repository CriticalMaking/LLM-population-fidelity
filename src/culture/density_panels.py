from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from machine_bias_reproduction.figures import (
    NEMD_BREAKS,
    annotate_quality_bands,
    log_nemd_density,
)
from machine_bias_reproduction.metrics import DISTANCES
from machine_bias_reproduction.questions import Question

from .palette import arm_linestyle, arm_tone
from .plate import MIXTRAL_INK, ModeFigure, empty_panel, per_mode
from .registry import arm_display
from .tables import is_flagged, reference_distances, series


def draw_culture_densities(
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
        grid, density = log_nemd_density(values)
        flagged = is_flagged(frames[culture])
        axis.plot(
            grid,
            density,
            label=arm_display(culture),
            color=arm_tone(culture).ink,
            linewidth=1.4,
            linestyle=(0, (1, 1.6)) if flagged else arm_linestyle(culture),
            alpha=0.45 if flagged else 1.0,
        )
    if reference is not None and reference.size:
        grid, density = log_nemd_density(reference)
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
        axis.set_xticks(NEMD_BREAKS, [f"{threshold:.2f}" for threshold in NEMD_BREAKS])
        annotate_quality_bands(axis)


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
            empty_panel(axis, f"no {mode} distances")
            return figure, [axis]
        baseline = None
        if reference is not None:
            baseline = reference.loc[reference["method"] == mode, "nEMD"].to_numpy(np.float64)
        draw_culture_densities(axis, frames, values, reference=baseline)
        axis.set_xlabel("nEMD (log scale)")
        axis.set_ylabel("Density (per natural-log unit)")
        axis.legend(loc="upper left", ncol=2)
        return figure, [axis]

    return per_mode(build, "fig_culture_nemd_density", destination, align_axis="y")


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
                empty_panel(axis, f"no {mode} {name}")
                continue
            draw_culture_densities(axis, frames, values, column=name)
            axis.set_xlabel(f"{name} (log scale)")
        row[0].set_ylabel("Density (per natural-log unit)")
        if series(frames, mode):
            row[0].legend(loc="upper left", ncol=2)
        return figure, row

    return per_mode(build, "fig_culture_distance_density", destination, align_axis="y")
