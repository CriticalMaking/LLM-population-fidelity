from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt

from machine_bias_reproduction.config import GLOBAL_SEED
from machine_bias_reproduction.data import load_subpops, load_wvs, social_predictors
from machine_bias_reproduction.figures import save_plate
from machine_bias_reproduction.plates import GRID, MUTED_INK
from machine_bias_reproduction.questions import Question

from .load import QuestionData

BOOTSTRAP_REPLICATES = 40

PREDICTORS = (
    "Sex_Female",
    "Age_<25",
    "Age_65-74",
    "Education_Low",
    "Education_Middle",
    "Employment_Unemployed",
    "Employment_Retired",
    "Marstat_Married",
    "Marstat_Widowed",
)


def _design(names: pd.Index) -> pd.DataFrame | None:
    wvs = load_wvs()
    subpops = load_subpops()["subpop"]
    predictors = social_predictors(wvs, subpops).reindex(names)
    available = [column for column in PREDICTORS if column in predictors.columns]
    if not available:
        return None
    frame = predictors[available].astype(np.float64)
    return frame.dropna()


def _fit(design: pd.DataFrame, props: pd.DataFrame, question: Question) -> pd.Series | None:
    scale = 100
    rows: list[np.ndarray] = []
    outcomes: list[int] = []
    values = props.reindex(design.index).to_numpy(dtype=np.float64)
    features = design.to_numpy(dtype=np.float64)
    for index in range(len(design)):
        shares = values[index]
        if not np.isfinite(shares).all():
            continue
        counts = np.rint(shares * scale).astype(int)
        for level, count in enumerate(counts):
            if count <= 0:
                continue
            rows.extend([features[index]] * count)
            outcomes.extend([level] * count)
    if len(set(outcomes)) < 2:
        return None
    exogenous = sm.add_constant(np.asarray(rows, dtype=np.float64), has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = sm.MNLogit(np.asarray(outcomes), exogenous).fit(disp=0, maxiter=200)
        except Exception:
            return None
    params = np.asarray(result.params, dtype=np.float64)
    labels = ["const", *design.columns]
    flat: dict[str, float] = {}
    for column in range(params.shape[1]):
        for row, label in enumerate(labels):
            if label == "const":
                continue
            flat[f"{label}|{question.answer_columns[column + 1]}"] = float(params[row, column])
    return pd.Series(flat)


def _bootstrap(
    design: pd.DataFrame,
    props: pd.DataFrame,
    question: Question,
    rng: np.random.Generator,
) -> pd.DataFrame:
    replicates: list[pd.Series] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = rng.choice(len(design), size=len(design), replace=True)
        fitted = _fit(design.iloc[sample], props.iloc[sample], question)
        if fitted is not None:
            replicates.append(fitted)
    return pd.DataFrame(replicates) if replicates else pd.DataFrame()


def figure_s16(
    loaded: dict[str, QuestionData], destination: Path
) -> tuple[list[Path], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(GLOBAL_SEED)
    for var, data in loaded.items():
        design = _design(data.names)
        if design is None or design.empty:
            continue
        truth = _fit(design, data.wvs, data.question)
        if truth is None:
            continue
        truth_boot = _bootstrap(design, data.wvs, data.question, rng)
        for name, props in data.series.items():
            fitted = _fit(design, props, data.question)
            if fitted is None:
                continue
            model_boot = _bootstrap(design, props, data.question, rng)
            for term in truth.index.intersection(fitted.index):
                rows.append(
                    {
                        "question": var,
                        "series": name,
                        "term": term,
                        "wvs": float(truth[term]),
                        "model": float(fitted[term]),
                        "wvs_se": (
                            float(truth_boot[term].std(ddof=1))
                            if term in truth_boot.columns
                            else np.nan
                        ),
                        "model_se": (
                            float(model_boot[term].std(ddof=1))
                            if term in model_boot.columns
                            else np.nan
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return [], frame

    produced: list[Path] = []
    for strategy in ("NTP", "FA"):
        subset = frame[frame["series"].str.startswith(strategy)]
        if subset.empty:
            continue
        series = sorted(subset["series"].unique())
        figure, axes = plt.subplots(
            1,
            len(series),
            figsize=(4.4 * len(series), 4.4),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for axis, name in zip(axes[0], series, strict=True):
            block = subset[subset["series"] == name]
            axis.errorbar(
                block["wvs"],
                block["model"],
                xerr=block["wvs_se"],
                yerr=block["model_se"],
                fmt="o",
                markersize=4,
                linewidth=0.9,
                capsize=2,
                alpha=0.8,
            )
            limits = [
                float(np.nanmin([block["wvs"].min(), block["model"].min()])),
                float(np.nanmax([block["wvs"].max(), block["model"].max()])),
            ]
            axis.plot(limits, limits, color=MUTED_INK, linewidth=0.8, linestyle=(0, (4, 2)))
            axis.axhline(0, color=GRID, linewidth=0.6)
            axis.axvline(0, color=GRID, linewidth=0.6)
            axis.set_xlabel(f"Ground-truth coefficient — {name}")
        axes[0][0].set_ylabel("Model coefficient")
        figure.tight_layout()
        produced.extend(save_plate(figure, destination / f"Figure-S16-cofdif-boot-{strategy}"))
    return produced, frame
