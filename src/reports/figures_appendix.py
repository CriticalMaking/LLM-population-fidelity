from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from machine_bias_reproduction.data import load_wvs
from machine_bias_reproduction.figures import save_plate
from machine_bias_reproduction.plates import GRID, MUTED_INK, bar_layout, magnitude_steps
from machine_bias_reproduction.questions import QUESTIONS

from .figures_main import SERIES_TONES, mds_plate
from .load import QuestionData
from .series import FA_MODELS, NTP_MODELS


def figure_s1(destination: Path) -> list[Path]:
    wvs = load_wvs()
    decade = (wvs["i_surveyyear"] // 10 * 10).astype(int)
    countries = sorted(wvs["i_country"].dropna().unique())
    decades = sorted(decade.unique())
    questions = list(QUESTIONS)
    figure, axes = plt.subplots(
        len(questions),
        len(countries),
        figsize=(2.9 * len(countries), 2.6 * len(questions)),
        squeeze=False,
        sharey="row",
    )
    palette = magnitude_steps(len(decades))
    for row, var in enumerate(questions):
        question = QUESTIONS[var]
        values = question.normalize(wvs[var])
        for column, country in enumerate(countries):
            axis = axes[row][column]
            mask = wvs["i_country"] == country
            width, offsets = bar_layout(len(decades))
            positions = np.arange(question.levels)
            for index, decade_value in enumerate(decades):
                block = values[mask & (decade == decade_value)]
                if block.dropna().empty:
                    continue
                shares = (
                    block.value_counts(normalize=True)
                    .reindex(question.wvs_labels, fill_value=0.0)
                    .to_numpy(dtype=np.float64)
                )
                axis.bar(
                    positions + offsets[index],
                    shares,
                    width=width,
                    color=palette[index],
                    label=f"{decade_value}s" if row == 0 and column == 0 else None,
                )
            axis.set_xticks(positions, list(question.answer_columns), fontsize=7)
            if row == 0:
                axis.set_xlabel(country, fontsize=9, labelpad=6)
                axis.xaxis.set_label_position("top")
            if column == 0:
                axis.set_ylabel(question.label, fontsize=9)
    axes[0][0].legend(fontsize=7)
    figure.tight_layout()
    return save_plate(figure, destination / "Figure-S1-outcome-distributions")


def figures_s2_s4(loaded: dict[str, QuestionData], destination: Path) -> list[Path]:
    produced: list[Path] = []
    ntp = [f"NTP-{model}" for model in NTP_MODELS]
    fa = [f"FA-{model}" for model in FA_MODELS]
    produced.extend(
        mds_plate(loaded, ntp, destination / "Figure-S2-MDS-NTP", include_baselines=True)
    )
    produced.extend(mds_plate(loaded, fa, destination / "Figure-S4-MDS-FA", include_baselines=True))
    return produced


def figure_s9(fit: pd.DataFrame, destination: Path) -> list[Path]:
    if fit.empty:
        return []
    frame = fit[fit["model"].isin(("social", "center"))]
    questions = [var for var in QUESTIONS if var in set(frame["question"])]
    figure, axes = plt.subplots(
        1, len(questions), figsize=(3.6 * len(questions), 5.0), sharey=True, squeeze=False
    )
    for axis, var in zip(axes[0], questions, strict=True):
        subset = frame[frame["question"] == var]
        pivot = subset.pivot_table(
            index="series", columns="model", values="adjusted_r_squared"
        ).sort_index()
        positions = np.arange(len(pivot))
        for offset, model, marker in ((-0.15, "social", "o"), (0.15, "center", "s")):
            if model not in pivot.columns:
                continue
            axis.scatter(
                pivot[model].to_numpy(dtype=np.float64),
                positions + offset,
                label=model,
                marker=marker,
                s=46,
            )
        axis.set_yticks(positions, pivot.index)
        axis.set_ylim(len(pivot) - 0.5, -0.5)
        axis.set_xlim(0, 1)
        axis.grid(axis="x", color=GRID, linewidth=0.6)
        axis.set_xlabel(f"adjusted $R^2$ — {QUESTIONS[var].label}")
    axes[0][0].legend()
    figure.tight_layout()
    return save_plate(figure, destination / "Figure-S9-R2-soc-all")


def figure_s10(loaded: dict[str, QuestionData], destination: Path) -> list[Path]:
    series = ["NTP-Llama-3-70B", "FA-Llama-3-70B", "NTP-Mixtral-8x7B", "FA-Mixtral-8x7B"]
    return mds_plate(loaded, series, destination / "Figure-S10-MDS-compare-NTP-FA")


def figure_s15(loaded: dict[str, QuestionData], destination: Path) -> list[Path]:
    rows: list[dict[str, object]] = []
    for var, data in loaded.items():
        reference = data.wvs
        candidates = dict(data.series)
        if data.linear is not None:
            candidates["Linear"] = data.linear
        if data.random:
            candidates["Random"] = data.random[0]
        for name, props in candidates.items():
            for column in data.question.answer_columns:
                left = reference[column].to_numpy(dtype=np.float64)
                right = props[column].to_numpy(dtype=np.float64)
                if np.std(left) == 0 or np.std(right) == 0:
                    continue
                rows.append(
                    {
                        "question": var,
                        "series": name,
                        "answer": column,
                        "correlation": float(np.corrcoef(left, right)[0, 1]),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    order = sorted(frame["series"].unique())
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.boxplot(
        [frame.loc[frame["series"] == name, "correlation"].to_numpy() for name in order],
        tick_labels=order,
        showmeans=True,
    )
    for index, name in enumerate(order, start=1):
        values = frame.loc[frame["series"] == name, "correlation"].to_numpy()
        axis.scatter(
            np.full(values.shape, index) + np.random.default_rng(0).normal(0, 0.04, values.size),
            values,
            s=12,
            alpha=0.6,
            color=SERIES_TONES[name].fill if name in SERIES_TONES else MUTED_INK,
            edgecolors=SERIES_TONES[name].ink if name in SERIES_TONES else MUTED_INK,
            linewidths=0.4,
        )
    axis.set_ylabel("Correlation with WVS answer shares")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    figure.tight_layout()
    return save_plate(figure, destination / "Figure-S15-correlations")
