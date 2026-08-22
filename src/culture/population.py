from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr

from machine_bias_reproduction.analysis import load_source
from machine_bias_reproduction.config import EXPECTED_SUBPOPULATIONS, OUTPUTS_ROOT, paths_for
from machine_bias_reproduction.data import PreparedData, country_of
from machine_bias_reproduction.metrics import nemd, pairwise_nemd
from machine_bias_reproduction.questions import Question, resolve_questions
from machine_bias_reproduction.r_rng import center_holdout_indices

from .palette import (
    MIXTRAL_ARCHIVED,
    MIXTRAL_FRESH,
    arm_order,
)
from .registry import ARMS, CULTURE_MODELS, csv_stems
from .tables import read_csv

MODES = ("ntp", "fa")

REFERENCE_COUNTRY = "Germany"

GERMAN_CELLS = 144

ADAPTABILITY_TABLE = OUTPUTS_ROOT / "culture" / "population_adaptability.csv"

STRUCTURE_TABLE = OUTPUTS_ROOT / "culture" / "population_structure.csv"

MODEL_INDEX = {key: index for index, key in enumerate(CULTURE_MODELS)}


def sources() -> list[tuple[str, str, str | None, str, str | None]]:
    found: list[tuple[str, str, str | None, str, str | None]] = [
        (f"{model.label} ({arm})", f"culture/{key}/{arm}", key, model.label, arm)
        for key, model in CULTURE_MODELS.items()
        for arm in arm_order(ARMS)
    ]
    found.append((MIXTRAL_ARCHIVED, "archived", None, MIXTRAL_ARCHIVED, None))
    found.append((MIXTRAL_FRESH, "fresh", None, MIXTRAL_FRESH, None))
    return found


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


def german_names(prepared: PreparedData) -> pd.Index:
    labels = prepared.names.to_series(index=prepared.names)
    return prepared.names[(country_of(labels) == REFERENCE_COUNTRY).to_numpy()]


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


def population_metrics(prepared: PreparedData, mode: str) -> dict[str, Any] | None:
    if mode not in prepared.modes():
        return None
    props = prepared.props(mode)
    german = german_names(prepared)
    wvs, llm = _arrays(prepared, props, prepared.names)
    wvs_de, llm_de = _arrays(prepared, props, german)
    d_wvs = float(np.median(pairwise_nemd(wvs)))
    d_llm = float(np.median(pairwise_nemd(llm)))
    d_wvs_de = float(np.median(pairwise_nemd(wvs_de))) if len(german) > 1 else float("nan")
    d_llm_de = float(np.median(pairwise_nemd(llm_de))) if len(german) > 1 else float("nan")
    return {
        "n_cells": len(prepared.names),
        "n_cells_german": len(german),
        "e_mean_nemd": float(np.mean(nemd(wvs, llm))),
        "e_mean_nemd_german": float(np.mean(nemd(wvs_de, llm_de))) if len(german) else float("nan"),
        "d_wvs": d_wvs,
        "d_llm": d_llm,
        "adaptability_ratio": d_llm / d_wvs if d_wvs else float("nan"),
        "d_wvs_german": d_wvs_de,
        "d_llm_german": d_llm_de,
        "adaptability_ratio_german": d_llm_de / d_wvs_de if d_wvs_de else float("nan"),
        "c_center_pop_nemd": float(nemd(wvs.mean(axis=0), llm.mean(axis=0))),
        "c_center_german_nemd": (
            float(nemd(wvs_de.mean(axis=0), llm_de.mean(axis=0))) if len(german) else float("nan")
        ),
        "r2_center_german": center_r2(wvs_de, llm_de, mode) if len(german) else float("nan"),
    }


def structure_metrics(prepared: PreparedData, mode: str) -> dict[str, Any] | None:
    if mode not in prepared.modes():
        return None
    props = prepared.props(mode)
    german = german_names(prepared)
    row: dict[str, Any] = {"n_cells": len(prepared.names), "n_cells_german": len(german)}
    for suffix, names in (("", prepared.names), ("_german", german)):
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
        for series, source, key, label, arm in sources():
            prepared = load_run(source, key, arm, question)
            if prepared is None:
                continue
            reported = reported_center(source, question) if with_reported_center else {}
            for mode in MODES:
                values = metrics(prepared, mode)
                if values is None:
                    continue
                row = {
                    "question": question.var,
                    "question_label": question.label,
                    "model_key": key,
                    "model_label": label,
                    "arm": arm,
                    "series": series,
                    "source": source,
                    "mode": mode,
                    **values,
                }
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
    for (question, mode), group in complete.groupby(["question", "mode"]):
        spread = group["d_wvs"].to_numpy(dtype=np.float64)
        assert np.allclose(spread, spread[0]), f"survey dispersion moved for {question}/{mode}"
        german = group["d_wvs_german"].to_numpy(dtype=np.float64)
        assert np.allclose(german, german[0]), f"german dispersion moved for {question}/{mode}"
    assert (complete["n_cells_german"] == GERMAN_CELLS).all()
    assert table["n_cells"].le(EXPECTED_SUBPOPULATIONS).all()
    assert table["n_cells_german"].le(GERMAN_CELLS).all()
    assert complete["adaptability_ratio"].ge(0).all()
    _cross_check_error(complete)


def check_structure(table: pd.DataFrame) -> None:
    assert not table.empty
    complete = _complete(table)
    assert not complete.empty
    expected = GERMAN_CELLS * (GERMAN_CELLS - 1) // 2
    assert (complete["n_pairs_german"] == expected).all()
    for column in ("rho_structure", "pearson_structure"):
        values = complete[column].to_numpy(dtype=np.float64)
        assert np.all((values >= -1.0) & (values <= 1.0))
