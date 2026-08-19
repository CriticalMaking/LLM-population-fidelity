from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from machine_bias_reproduction.metrics import QUALITY_LABELS
from machine_bias_reproduction.plates import QUALITY_RAMP, SEPARATOR
from machine_bias_reproduction.questions import Question

from .capacity_panels import is_low_capacity
from .palette import MODE_MARKERS, MODE_TONES, MODES, arm_order, model_marker, model_tone
from .plate import ModeFigure, empty_panel, mixtral_rule, per_mode, rank_axis
from .registry import CULTURE_MODELS
from .tables import has_distances, reference_distances


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
            empty_panel(axis, f"no {mode} quality bands")
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

    return per_mode(build, stem, destination, align_axis=None)


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
            empty_panel(axis, f"no {mode} distances")
            return figure, [axis]
        tone = MODE_TONES[mode]
        slots = [positions[culture] for culture in subset["culture"]]
        axis.scatter(
            subset["mean"],
            slots,
            label=f"{mode} mean",
            marker=MODE_MARKERS[mode],
            color=tone.fill,
            edgecolors=tone.ink,
            linewidths=0.8,
            s=46,
        )
        axis.scatter(
            subset["median"], slots, label=f"{mode} median", marker="|", color=tone.ink, s=140
        )
        mixtral_rule(axis, reference, mode)
        rank_axis(axis, order)
        return figure, [axis]

    return per_mode(build, stem, destination)


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
            flagged = subset.apply(is_low_capacity, axis=1)
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
            empty_panel(axis, f"no {mode} distances")
            return figure, [axis]
        mixtral_rule(axis, reference, mode)
        rank_axis(axis, order)
        return figure, [axis]

    return per_mode(build, "fig_model_culture_comparison", destination)
