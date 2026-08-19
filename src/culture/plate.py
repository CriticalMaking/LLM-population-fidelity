from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from machine_bias_reproduction.figures import save_plate
from machine_bias_reproduction.plates import GRID, MUTED_INK

from .palette import MODES, REFERENCE_TONES

MIXTRAL_INK = REFERENCE_TONES["Mixtral archived"].ink
REFERENCE_DASH = (0, (4, 2))
RULE_HALF_WIDTH = 0.44
RULE_DASH = (0, (4, 2))

ModeFigure = tuple[Any, list[Any]]


def empty_panel(axis: Any, message: str) -> None:
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


def align(built: dict[str, ModeFigure], which: str | None) -> None:
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


def per_mode(
    build: Callable[[str], ModeFigure],
    stem: str,
    destination: Path,
    *,
    align_axis: str | None = "x",
) -> list[Path]:
    built = {mode: build(mode) for mode in MODES}
    align(built, align_axis)
    produced: list[Path] = []
    for mode, (figure, _) in built.items():
        figure.tight_layout()
        produced.extend(save_plate(figure, destination / f"{stem}_{mode.lower()}"))
    return produced


def answer_rule(
    axis: Any,
    positions: np.ndarray,
    shares: np.ndarray,
    *,
    color: str,
    label: str,
    linestyle: Any = "solid",
) -> None:
    for index, (position, value) in enumerate(zip(positions, shares, strict=True)):
        axis.hlines(
            value,
            position - RULE_HALF_WIDTH,
            position + RULE_HALF_WIDTH,
            color=color,
            linestyle=linestyle,
            linewidth=1.4,
            label=label if index == 0 else "",
        )


def mixtral_rule(axis: Any, reference: pd.DataFrame | None, mode: str) -> None:
    if reference is None:
        return
    baseline = reference.loc[reference["method"] == mode, "nEMD"]
    if baseline.empty:
        return
    axis.axvline(
        float(baseline.mean()),
        color=MIXTRAL_INK,
        linestyle=REFERENCE_DASH,
        linewidth=1.0,
        label="Mixtral archived",
    )


def rank_axis(axis: Any, order: list[str]) -> None:
    axis.set_yticks(range(len(order)), order)
    axis.set_ylim(len(order) - 0.5, -0.5)
    axis.set_xlabel("Subpopulation nEMD (lower is closer to the WVS)")
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.legend()
