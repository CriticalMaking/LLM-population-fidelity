from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from machine_bias_reproduction.figures import save_plate
from machine_bias_reproduction.plates import MUTED_INK, SEPARATOR, STATUS_TONES

from .palette import MODES, arm_order, arm_tone
from .plate import REFERENCE_DASH
from .registry import PREFLIGHT_MIN_VALID_ANSWER_MASS


def is_low_capacity(row: pd.Series) -> bool:
    informative = row.get("informative")
    return informative is False or informative == 0


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
            linestyle=REFERENCE_DASH,
            linewidth=1.0,
            label=f"informative threshold ({PREFLIGHT_MIN_VALID_ANSWER_MASS:.2f})",
        )
        axes[1].set_xscale("log")
        axes[1].set_yticks(positions, order)
        axes[1].set_ylim(len(order) - 0.5, -0.5)
        axes[1].set_xlabel("Mean valid-answer mass (log scale)")
        axes[1].legend(loc="lower right")

    figure.tight_layout()
    return save_plate(figure, destination / "fig_culture_capacity")


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
    return save_plate(figure, destination / "fig_run_capacity")
