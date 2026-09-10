from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from scipy.stats import pearsonr, spearmanr

from machine_bias_reproduction.analysis import load_source
from machine_bias_reproduction.config import (
    EXPECTED_SUBPOPULATIONS,
    FIRST_REPLICATE,
    OUTPUTS_ROOT,
    paths_for,
)
from machine_bias_reproduction.data import PreparedData, country_of
from machine_bias_reproduction.metrics import nemd, pairwise_nemd
from machine_bias_reproduction.questions import Question, resolve_questions
from machine_bias_reproduction.r_rng import center_holdout_indices

from .palette import (
    MIXTRAL_ARCHIVED,
    MIXTRAL_FRESH,
    arm_order,
)
from .registry import ARMS, CULTURE_MODELS, arm_display, csv_stems, replicates, run_slug
from .tables import read_csv

MODES = ("ntp", "fa")

REPLICATE = "replicate"

RUN_KEYS: tuple[str, ...] = (
    "question",
    "question_label",
    "model_key",
    "model_label",
    "arm",
    "series",
    "mode",
    "group",
    "level",
)

REFERENCE_COUNTRIES: dict[str, str] = {"german": "Germany", "mexican": "Mexico"}

REFERENCE_CELLS: dict[str, int] = {"german": 144, "mexican": 131}

ADAPTABILITY_TABLE = OUTPUTS_ROOT / "culture" / "population_adaptability.csv"

STRUCTURE_TABLE = OUTPUTS_ROOT / "culture" / "population_structure.csv"

MODEL_INDEX = {key: index for index, key in enumerate(CULTURE_MODELS)}


class RunSource(NamedTuple):
    series: str
    source: str
    key: str | None
    label: str
    arm: str | None
    replicate: int


def sources() -> list[RunSource]:
    found = [
        RunSource(
            f"{model.label} ({arm_display(arm)})",
            run_slug(key, arm, replicate),
            key,
            model.label,
            arm,
            replicate,
        )
        for key, model in CULTURE_MODELS.items()
        for arm in arm_order(ARMS)
        for replicate in replicates(key, arm)
    ]
    for label, source in ((MIXTRAL_ARCHIVED, "archived"), (MIXTRAL_FRESH, "fresh")):
        found.append(RunSource(label, source, None, label, None, FIRST_REPLICATE))
    return found


def run_identity(run: RunSource, question: Question, mode: str) -> dict[str, Any]:
    return {
        "question": question.var,
        "question_label": question.label,
        "model_key": run.key,
        "model_label": run.label,
        "arm": run.arm,
        "series": run.series,
        "source": run.source,
        "mode": mode,
        REPLICATE: run.replicate,
    }


def across_replicates(
    frame: pd.DataFrame,
    keys: Sequence[str] = RUN_KEYS,
    spread: Sequence[str] = (),
) -> pd.DataFrame:
    if REPLICATE not in frame.columns or frame.empty:
        return frame
    present = [key for key in keys if key in frame.columns]
    rest = [column for column in frame.columns if column not in (*present, REPLICATE)]
    flags = [column for column in rest if is_bool_dtype(frame[column])]
    numeric = [column for column in rest if column not in flags and is_numeric_dtype(frame[column])]
    labels = [column for column in rest if column not in flags and column not in numeric]
    aggregates: dict[str, tuple[str, Any]] = {
        **{column: (column, "mean") for column in numeric},
        **{column: (column, "any") for column in flags},
        **{column: (column, "first") for column in labels},
        "n_replicates": (REPLICATE, "nunique"),
    }
    grouped = frame.groupby(present, dropna=False, sort=False)
    collapsed = grouped.agg(**aggregates)
    measured = [column for column in spread if column in numeric]
    if measured:
        collapsed = pd.concat([collapsed, grouped[measured].std(ddof=1).add_suffix("_sd")], axis=1)
    ordered = [*present, *rest, "n_replicates", *(f"{column}_sd" for column in measured)]
    return collapsed.reset_index()[ordered]


def load_run(
    source: str,
    key: str | None,
    arm: str | None,
    question: Question,
) -> PreparedData | None:
    stems = csv_stems(key, arm, question) if key and arm else None
    for modes in (("ntp", "fa"), ("fa",), ("ntp",)):
        try:
            return load_source(source, paths_for(source, question.var), question, stems, modes)
        except (FileNotFoundError, ValueError):
            continue
    return None


def country_names(prepared: PreparedData, country: str) -> pd.Index:
    labels = prepared.names.to_series(index=prepared.names)
    return prepared.names[(country_of(labels) == country).to_numpy()]


def center_r2(wvs: np.ndarray, llm: np.ndarray, mode: str) -> float:
    count = len(wvs)
    if count <= 21:
        return float("nan")
    indices = center_holdout_indices(count, mode)
    center = llm[indices].mean(axis=0)
    keep = np.ones(count, dtype=bool)
    keep[indices] = False
    response = pd.Series(np.log1p(nemd(wvs, llm))[keep], name="nEMD_log")
    frame = pd.DataFrame({"nEMD_center": np.log1p(nemd(wvs, center))[keep]})
    fit = sm.OLS(response, sm.add_constant(frame, has_constant="add")).fit()
    return float(fit.rsquared_adj)


def _arrays(
    prepared: PreparedData,
    props: pd.DataFrame,
    names: pd.Index,
) -> tuple[np.ndarray, np.ndarray]:
    columns = list(prepared.question.answer_columns)
    wvs = prepared.wvs_props.loc[names, columns].to_numpy(dtype=np.float64)
    llm = props.loc[names, columns].to_numpy(dtype=np.float64)
    return wvs, llm


def _reference_metrics(
    prepared: PreparedData,
    props: pd.DataFrame,
    key: str,
    country: str,
    mode: str,
) -> dict[str, Any]:
    names = country_names(prepared, country)
    if not len(names):
        return {
            f"n_cells_{key}": 0,
            f"e_mean_nemd_{key}": float("nan"),
            f"d_wvs_{key}": float("nan"),
            f"d_llm_{key}": float("nan"),
            f"adaptability_ratio_{key}": float("nan"),
            f"c_center_{key}_nemd": float("nan"),
            f"r2_center_{key}": float("nan"),
        }
    wvs, llm = _arrays(prepared, props, names)
    d_wvs = float(np.median(pairwise_nemd(wvs))) if len(names) > 1 else float("nan")
    d_llm = float(np.median(pairwise_nemd(llm))) if len(names) > 1 else float("nan")
    return {
        f"n_cells_{key}": len(names),
        f"e_mean_nemd_{key}": float(np.mean(nemd(wvs, llm))),
        f"d_wvs_{key}": d_wvs,
        f"d_llm_{key}": d_llm,
        f"adaptability_ratio_{key}": d_llm / d_wvs if d_wvs else float("nan"),
        f"c_center_{key}_nemd": float(nemd(wvs.mean(axis=0), llm.mean(axis=0))),
        f"r2_center_{key}": center_r2(wvs, llm, mode),
    }


def population_metrics(prepared: PreparedData, mode: str) -> dict[str, Any] | None:
    if mode not in prepared.modes():
        return None
    props = prepared.props(mode)
    wvs, llm = _arrays(prepared, props, prepared.names)
    d_wvs = float(np.median(pairwise_nemd(wvs)))
    d_llm = float(np.median(pairwise_nemd(llm)))
    row: dict[str, Any] = {
        "n_cells": len(prepared.names),
        "e_mean_nemd": float(np.mean(nemd(wvs, llm))),
        "d_wvs": d_wvs,
        "d_llm": d_llm,
        "adaptability_ratio": d_llm / d_wvs if d_wvs else float("nan"),
        "c_center_pop_nemd": float(nemd(wvs.mean(axis=0), llm.mean(axis=0))),
    }
    for key, country in REFERENCE_COUNTRIES.items():
        row.update(_reference_metrics(prepared, props, key, country, mode))
    return row


def structure_metrics(prepared: PreparedData, mode: str) -> dict[str, Any] | None:
    if mode not in prepared.modes():
        return None
    props = prepared.props(mode)
    row: dict[str, Any] = {"n_cells": len(prepared.names)}
    scopes: list[tuple[str, pd.Index]] = [("", prepared.names)]
    for key, country in REFERENCE_COUNTRIES.items():
        names = country_names(prepared, country)
        row[f"n_cells_{key}"] = len(names)
        scopes.append((f"_{key}", names))
    for suffix, names in scopes:
        wvs, llm = _arrays(prepared, props, names)
        if len(names) < 3:
            row[f"n_pairs{suffix}"] = 0
            row[f"rho_structure{suffix}"] = float("nan")
            row[f"rho_structure{suffix}_p_value"] = float("nan")
            row[f"pearson_structure{suffix}"] = float("nan")
            row[f"pearson_structure{suffix}_p_value"] = float("nan")
            continue
        survey = pairwise_nemd(wvs)
        model = pairwise_nemd(llm)
        rho = spearmanr(survey, model)
        pearson = pearsonr(survey, model)
        row[f"n_pairs{suffix}"] = len(survey)
        row[f"rho_structure{suffix}"] = float(rho.statistic)
        row[f"rho_structure{suffix}_p_value"] = float(rho.pvalue)
        row[f"pearson_structure{suffix}"] = float(pearson.statistic)
        row[f"pearson_structure{suffix}_p_value"] = float(pearson.pvalue)
    return row


def reported_center(source: str, question: Question) -> dict[str, float]:
    fit = read_csv(paths_for(source, question.var).outputs / "regression_fit.csv")
    if fit is None:
        return {}
    center = fit[fit["model"] == "center"]
    return dict(zip(center["mode"], center["adjusted_r_squared"], strict=True))


def _build(
    metrics: Any,
    questions: list[Question] | None = None,
    with_reported_center: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for question in questions or resolve_questions(None):
        for run in sources():
            prepared = load_run(run.source, run.key, run.arm, question)
            if prepared is None:
                continue
            reported = reported_center(run.source, question) if with_reported_center else {}
            for mode in MODES:
                values = metrics(prepared, mode)
                if values is None:
                    continue
                row = {**run_identity(run, question, mode), **values}
                if with_reported_center:
                    row["r2_center_reported"] = reported.get(mode, float("nan"))
                rows.append(row)
    return pd.DataFrame(rows)


def build_adaptability(questions: list[Question] | None = None) -> pd.DataFrame:
    return _build(population_metrics, questions, with_reported_center=True)


def build_structure(questions: list[Question] | None = None) -> pd.DataFrame:
    return _build(structure_metrics, questions)


def _complete(table: pd.DataFrame) -> pd.DataFrame:
    fullest = table.groupby("question")["n_cells"].transform("max")
    return table[table["n_cells"] == fullest]


def _cross_check_error(table: pd.DataFrame) -> None:
    row = table.iloc[0]
    outputs = paths_for(row["source"], row["question"]).outputs
    distances = read_csv(outputs / "subpopulation_distances.csv")
    if distances is None:
        return
    scored = distances[distances["method"] == row["mode"].upper()]
    if scored.empty:
        return
    assert np.isclose(scored["nEMD"].mean(), row["e_mean_nemd"]), (
        f"recomputed error disagrees with subpopulation_distances.csv for {row['source']}"
    )


def check_adaptability(table: pd.DataFrame) -> None:
    assert not table.empty
    complete = _complete(table)
    assert not complete.empty
    columns = ["d_wvs", *(f"d_wvs_{key}" for key in REFERENCE_COUNTRIES)]
    for (question, mode), group in complete.groupby(["question", "mode"]):
        for column in columns:
            spread = group[column].to_numpy(dtype=np.float64)
            assert np.allclose(spread, spread[0]), (
                f"survey dispersion moved for {column} on {question}/{mode}"
            )
    assert table["n_cells"].le(EXPECTED_SUBPOPULATIONS).all()
    for key, expected in REFERENCE_CELLS.items():
        assert (complete[f"n_cells_{key}"] == expected).all(), f"{key} cell count moved"
        assert table[f"n_cells_{key}"].le(expected).all()
    assert complete["adaptability_ratio"].ge(0).all()
    _cross_check_error(complete)


def check_structure(table: pd.DataFrame) -> None:
    assert not table.empty
    complete = _complete(table)
    assert not complete.empty
    for key, cells in REFERENCE_CELLS.items():
        expected = cells * (cells - 1) // 2
        assert (complete[f"n_pairs_{key}"] == expected).all(), f"{key} pair count moved"
    for column in ("rho_structure", "pearson_structure"):
        values = complete[column].to_numpy(dtype=np.float64)
        assert np.all((values >= -1.0) & (values <= 1.0))
