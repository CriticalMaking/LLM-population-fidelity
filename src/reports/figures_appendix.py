"""Appendix figures S1, S2, S4, S9, S10 and S15."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from machine_bias_reproduction.data import load_wvs
from machine_bias_reproduction.figures import _save
from machine_bias_reproduction.questions import QUESTIONS

from .figures_main import SERIES_COLORS, mds_plate
from .load import QuestionData
from .series import FA_MODELS, NTP_MODELS

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def figure_s1(destination: Path) -> list[Path]:
    """Figure S1 — distribution of the four outcome variables by country and decade.

    Upstream prints this to a device and never saves it (``6-results.R:1238``);
    it is written here so the appendix set is complete on disk.
    """
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
    palette = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(decades)))
    for row, var in enumerate(questions):
        question = QUESTIONS[var]
        values = question.normalize(wvs[var])
        for column, country in enumerate(countries):
            axis = axes[row][column]
            mask = wvs["i_country"] == country
            width = 0.8 / len(decades)
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
                    positions + index * width - 0.4 + width / 2,
                    shares,
                    width=width,
                    color=palette[index],
                    label=f"{decade_value}s" if row == 0 and column == 0 else None,
                )
            axis.set_xticks(positions, list(question.answer_columns), fontsize=7)
            if row == 0:
                axis.set_title(country, fontsize=9)
            if column == 0:
                axis.set_ylabel(question.label, fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    return _save(figure, destination / "Figure-S1-outcome-distributions")


def figures_s2_s4(loaded: dict[str, QuestionData], destination: Path) -> list[Path]:
    """Figures S2 and S4 — MDS for every series, with the Linear and Random baselines."""
    produced: list[Path] = []
    ntp = [f"NTP-{model}" for model in NTP_MODELS]
    fa = [f"FA-{model}" for model in FA_MODELS]
    produced.extend(
        mds_plate(loaded, ntp, destination / "Figure-S2-MDS-NTP", include_baselines=True)
    )
    produced.extend(mds_plate(loaded, fa, destination / "Figure-S4-MDS-FA", include_baselines=True))
    return produced


def figure_s9(fit: pd.DataFrame, destination: Path) -> list[Path]:
    """Figure S9 — adjusted R-squared of the social and centre models, all series."""
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
        axis.grid(axis="x", alpha=0.2)
        axis.set(xlabel="adjusted $R^2$", title=QUESTIONS[var].label)
    axes[0][0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    return _save(figure, destination / "Figure-S9-R2-soc-all")


def figure_s10(loaded: dict[str, QuestionData], destination: Path) -> list[Path]:
    """Figure S10 — MDS for Llama-3-70B and Mixtral-8x7B under both strategies."""
    series = ["NTP-Llama-3-70B", "FA-Llama-3-70B", "NTP-Mixtral-8x7B", "FA-Mixtral-8x7B"]
    return mds_plate(loaded, series, destination / "Figure-S10-MDS-compare-NTP-FA")


def figure_s15(loaded: dict[str, QuestionData], destination: Path) -> list[Path]:
    """Figure S15 — correlation between model and WVS answer shares, by series.

    One correlation per answer category, taken across subpopulations; the
    boxplot is over those categories and questions.
    """
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
            alpha=0.5,
            color=SERIES_COLORS.get(name, "#444444"),
        )
    axis.set_ylabel("Correlation with WVS answer shares")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    return _save(figure, destination / "Figure-S15-correlations")
