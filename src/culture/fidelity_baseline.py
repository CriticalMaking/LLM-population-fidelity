from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from machine_bias_reproduction.config import FIRST_REPLICATE, OUTPUTS_ROOT
from machine_bias_reproduction.metrics import nemd
from machine_bias_reproduction.questions import Question, resolve_questions

from .fidelity import (
    COUNTRY_FAMILY,
    IDENTITY,
    MIN_CELLS_FOR_DISPERSION,
    POPULATION,
    FloatArray,
    answer_array,
    cell_facets,
    closeness_score,
    group_fidelity,
    grouped_scores,
    lead_with,
    loaded_runs,
)
from .matching import HOME
from .population import REPLICATE, RunSource, run_identity, sources
from .registry import BASE_ARM

MANUSCRIPT_MODELS: tuple[str, ...] = (
    "gemma4_31b",
    "gemma4_e4b",
    "qwen3_vl_8b",
    "qwen3_vl_2b",
    "llama3_2_3b",
    "muse_glimmer_30b",
    "terra",
)

MANUSCRIPT_ARMS: tuple[str, ...] = (BASE_ARM, "german", "spanish-mx")

HOME_SCOPE = "home"

SCOPES: tuple[str, ...] = ("population", "subgroups", HOME_SCOPE)

EVERY = "all"

BASELINE_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_baseline.csv"

BASELINE_SUMMARY_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_baseline_summary.csv"


def manuscript_runs(
    found: Sequence[RunSource] | None = None,
    every_replicate: bool = False,
) -> list[RunSource]:
    return [
        run
        for run in (sources() if found is None else found)
        if (every_replicate or run.replicate == FIRST_REPLICATE)
        and (run.key is None or (run.key in MANUSCRIPT_MODELS and run.arm in MANUSCRIPT_ARMS))
    ]


def survey_center_prediction(wvs: FloatArray) -> FloatArray:
    return np.tile(wvs.mean(axis=0), (wvs.shape[0], 1))


def leave_one_out_prediction(wvs: FloatArray) -> FloatArray:
    count = wvs.shape[0]
    if count < MIN_CELLS_FOR_DISPERSION:
        return np.full_like(wvs, np.nan)
    return np.asarray((wvs.sum(axis=0) - wvs) / (count - 1), dtype=np.float64)


def mean_accuracy(wvs: FloatArray, prediction: FloatArray) -> float:
    error = np.atleast_1d(np.asarray(nemd(wvs, prediction), dtype=np.float64))
    return closeness_score(float(error.mean()))


def above_null_score(score: float, null: float) -> float:
    if not (np.isfinite(score) and np.isfinite(null)) or null >= 1.0:
        return float("nan")
    return float((score - null) / (1.0 - null))


def baseline_readings(wvs: FloatArray, llm: FloatArray) -> dict[str, Any]:
    null = group_fidelity(wvs, survey_center_prediction(wvs))
    accuracy = mean_accuracy(wvs, llm)
    return {
        "n_cells": int(wvs.shape[0]),
        "score_accuracy": accuracy,
        "null_accuracy": null["score_accuracy"],
        "null_accuracy_loo": mean_accuracy(wvs, leave_one_out_prediction(wvs)),
        "null_score_center": null["score_center"],
        "null_adaptability_ratio": null["adaptability_ratio"],
        "null_model_flat": null["model_flat"],
        "null_pfs": null["pfs"],
        "accuracy_gap": accuracy - null["score_accuracy"],
        "above_null": above_null_score(accuracy, null["score_accuracy"]),
    }


def build_baseline(
    questions: list[Question] | None = None,
    runs: Sequence[RunSource] | None = None,
) -> pd.DataFrame:
    selected = manuscript_runs() if runs is None else list(runs)
    facets = cell_facets()
    rows: list[dict[str, Any]] = []
    for question in questions or resolve_questions(None):
        for run, prepared in loaded_runs(question, selected):
            names = prepared.names
            facet = facets.reindex(names)
            wvs = answer_array(prepared, prepared.wvs_props, names)
            for mode in prepared.modes():
                llm = answer_array(prepared, prepared.props(mode), names)
                identity = run_identity(run, question, mode)
                rows += grouped_scores(identity, facet, baseline_readings, wvs, llm)
    return lead_with(pd.DataFrame(rows), [*IDENTITY, "group", "level"])


def scope_rows(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    pooled = frame["group"].eq(POPULATION)
    if scope == "population":
        return frame[pooled]
    if scope == "subgroups":
        return frame[~pooled & frame["model_key"].notna()]
    if scope == HOME_SCOPE:
        return frame[frame["group"].eq(COUNTRY_FAMILY) & frame["level"].eq(frame["arm"].map(HOME))]
    raise ValueError(f"unknown scope: {scope}")


def _summary_row(block: pd.DataFrame) -> dict[str, Any]:
    above = block["above_null"].to_numpy(dtype=np.float64)
    shift = (block["null_accuracy"] - block["null_accuracy_loo"]).abs()
    return {
        "n_scores": len(block),
        "n_cells_min": int(block["n_cells"].min()),
        "n_cells_max": int(block["n_cells"].max()),
        "null_accuracy_min": float(block["null_accuracy"].min()),
        "null_accuracy_max": float(block["null_accuracy"].max()),
        "null_loo_shift_max": float(shift.max()),
        "accuracy_median": float(block["score_accuracy"].median()),
        "accuracy_min": float(block["score_accuracy"].min()),
        "accuracy_max": float(block["score_accuracy"].max()),
        "above_null_median": float(np.nanmedian(above)),
        "above_null_min": float(np.nanmin(above)),
        "above_null_max": float(np.nanmax(above)),
        "n_above_null": int(np.sum(above > 0.0)),
        "n_below_minus_one": int(np.sum(above < -1.0)),
    }


def baseline_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in SCOPES:
        selected = scope_rows(frame, scope)
        if selected.empty:
            continue
        for (question, mode), block in selected.groupby(["question", "mode"], sort=True):
            rows.append({"scope": scope, "question": question, "mode": mode, **_summary_row(block)})
        rows.append({"scope": scope, "question": EVERY, "mode": EVERY, **_summary_row(selected)})
    return pd.DataFrame(rows)


def check_baseline(frame: pd.DataFrame, groups: pd.DataFrame, runs: Sequence[RunSource]) -> None:
    keys = [*IDENTITY, "group", "level"]
    reference = groups[
        groups["source"].isin({run.source for run in runs}) & groups[REPLICATE].eq(FIRST_REPLICATE)
    ]
    merged = frame.merge(
        reference[[*keys, "n_cells", "score_accuracy"]],
        on=keys,
        how="inner",
        suffixes=("", "_ref"),
        validate="one_to_one",
    )
    assert len(merged) == len(frame) == len(reference), "the baseline scores other rows"
    assert merged["n_cells"].eq(merged["n_cells_ref"]).all(), "the baseline reads other cells"
    assert np.allclose(
        merged["score_accuracy"].to_numpy(dtype=np.float64),
        merged["score_accuracy_ref"].to_numpy(dtype=np.float64),
        atol=1e-12,
        equal_nan=True,
    ), "the baseline's model accuracy drifted from the groups table"
    scored = frame[frame["n_cells"].ge(MIN_CELLS_FOR_DISPERSION)]
    assert scored["null_model_flat"].all(), "the survey center varies across cells"
    assert np.allclose(scored["null_adaptability_ratio"].to_numpy(dtype=np.float64), 0.0)
    assert not scored["null_pfs"].gt(0.0).any(), "the survey center scores a positive PFS"
    assert np.allclose(frame["null_score_center"].to_numpy(dtype=np.float64), 1.0)
    assert not frame["above_null"].gt(1.0).any(), "an above-null score exceeds one"
