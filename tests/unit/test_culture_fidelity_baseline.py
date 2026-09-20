from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from culture import fidelity, fidelity_baseline
from culture.population import RunSource, run_identity
from machine_bias_reproduction.metrics import nemd
from machine_bias_reproduction.questions import resolve_question

CELLS = (
    "Germany 2018 Female 45-54 Middle Working Married",
    "Germany 1997 Male 25-34 High Student Single",
    "Mexico 2005 Female 55-64 Low Retired Widowed",
    "Mexico 2017 Male <25 High Working Single",
    "United States 1995 Female 75+ Middle Homemaker Cohabiting",
)

SURVEY = np.array(
    [
        [0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4],
        [0.25, 0.25, 0.25, 0.25],
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.7],
    ]
)

MODEL = np.array(
    [
        [0.35, 0.35, 0.2, 0.1],
        [0.15, 0.25, 0.3, 0.3],
        [0.2, 0.3, 0.3, 0.2],
        [0.6, 0.2, 0.1, 0.1],
        [0.2, 0.1, 0.2, 0.5],
    ]
)


def _run(key: str | None, arm: str | None, replicate: int = 1) -> RunSource:
    label = key or "Mixtral archived"
    source = f"culture/{key}/{arm}" if key else "archived"
    return RunSource(f"{label} ({arm})", source, key, label, arm, replicate)


def _frame(
    runs: list[RunSource], model: np.ndarray, score: Callable[..., dict[str, Any]]
) -> pd.DataFrame:
    facet = fidelity.cell_facets(CELLS).reindex(list(CELLS))
    question = resolve_question("d_happy")
    rows: list[dict[str, Any]] = []
    for run in runs:
        identity = run_identity(run, question, "ntp")
        rows += fidelity.grouped_scores(identity, facet, score, SURVEY, model)
    return pd.DataFrame(rows)


def test_the_survey_center_predicts_one_distribution_for_every_cell() -> None:
    prediction = fidelity_baseline.survey_center_prediction(SURVEY)

    assert prediction.shape == SURVEY.shape
    assert np.allclose(prediction, SURVEY.mean(axis=0))


def test_the_leave_one_out_center_drops_each_cell_from_its_own_prediction() -> None:
    prediction = fidelity_baseline.leave_one_out_prediction(SURVEY)

    assert np.allclose(prediction[0], SURVEY[1:].mean(axis=0))
    assert np.allclose(prediction[4], SURVEY[:4].mean(axis=0))
    assert np.isnan(fidelity_baseline.leave_one_out_prediction(SURVEY[:1])).all()


def test_the_survey_center_scores_its_accuracy_with_perfect_center_and_zero_pfs() -> None:
    reading = fidelity_baseline.baseline_readings(SURVEY, MODEL)
    expected = 1.0 - np.mean(nemd(SURVEY, np.tile(SURVEY.mean(axis=0), (len(SURVEY), 1))))

    assert reading["null_accuracy"] == pytest.approx(expected)
    assert reading["null_score_center"] == pytest.approx(1.0)
    assert reading["null_adaptability_ratio"] == 0.0
    assert reading["null_model_flat"]
    assert reading["null_pfs"] == 0.0
    assert reading["score_accuracy"] == pytest.approx(1.0 - np.mean(nemd(SURVEY, MODEL)))
    assert reading["accuracy_gap"] == pytest.approx(
        reading["score_accuracy"] - reading["null_accuracy"]
    )


def test_the_above_null_score_is_zero_at_the_center_one_at_the_survey_negative_below() -> None:
    center = fidelity_baseline.survey_center_prediction(SURVEY)
    worse = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(SURVEY), 1))

    at_center = fidelity_baseline.baseline_readings(SURVEY, center)["above_null"]
    at_survey = fidelity_baseline.baseline_readings(SURVEY, SURVEY)["above_null"]
    below = fidelity_baseline.baseline_readings(SURVEY, worse)["above_null"]

    assert at_center == pytest.approx(0.0)
    assert at_survey == pytest.approx(1.0)
    assert below < 0.0


def test_the_above_null_score_is_blank_when_the_null_is_perfect_or_missing() -> None:
    assert np.isnan(fidelity_baseline.above_null_score(0.9, 1.0))
    assert np.isnan(fidelity_baseline.above_null_score(float("nan"), 0.8))
    assert fidelity_baseline.above_null_score(0.9, 0.8) == pytest.approx(0.5)


def test_the_manuscript_runs_are_the_first_replicate_of_its_models_and_variants() -> None:
    found = [
        _run("gemma4_31b", "base"),
        _run("gemma4_31b", "german"),
        _run("gemma4_31b", "spanish-mx"),
        _run("gemma4_31b", "spanish"),
        _run("gemma4_31b", "german", replicate=2),
        _run("luna", "base"),
        _run("terra", "base"),
        _run(None, None),
    ]

    kept = fidelity_baseline.manuscript_runs(found)

    assert [(run.key, run.arm, run.replicate) for run in kept] == [
        ("gemma4_31b", "base", 1),
        ("gemma4_31b", "german", 1),
        ("gemma4_31b", "spanish-mx", 1),
        ("terra", "base", 1),
        (None, None, 1),
    ]


def test_the_home_scope_keeps_each_variant_inside_its_own_country() -> None:
    runs = [
        _run("gemma4_31b", "base"),
        _run("gemma4_31b", "german"),
        _run("gemma4_31b", "spanish-mx"),
    ]
    frame = _frame(runs, MODEL, fidelity_baseline.baseline_readings)

    home = fidelity_baseline.scope_rows(frame, "home")

    assert sorted(zip(home["arm"], home["level"], strict=True)) == [
        ("german", "Germany"),
        ("spanish-mx", "Mexico"),
    ]


def test_the_subgroup_scope_leaves_out_the_pooled_rows_and_the_mixtral_references() -> None:
    runs = [_run("gemma4_31b", "base"), _run(None, None)]
    frame = _frame(runs, MODEL, fidelity_baseline.baseline_readings)

    subgroups = fidelity_baseline.scope_rows(frame, "subgroups")

    assert not subgroups["group"].eq(fidelity.POPULATION).any()
    assert subgroups["model_key"].eq("gemma4_31b").all()
    with pytest.raises(ValueError, match="unknown scope"):
        fidelity_baseline.scope_rows(frame, "everything")


def test_the_summary_counts_the_runs_above_the_survey_center() -> None:
    runs = [_run("gemma4_31b", "base"), _run("gemma4_e4b", "base")]
    frame = _frame(runs, MODEL, fidelity_baseline.baseline_readings)
    frame.loc[frame["model_key"].eq("gemma4_31b"), "above_null"] = -0.5
    frame.loc[frame["model_key"].eq("gemma4_e4b"), "above_null"] = 0.25

    summary = fidelity_baseline.baseline_summary(frame)
    pooled = summary[summary["scope"].eq("population")]

    assert list(pooled["question"]) == ["d_happy", fidelity_baseline.EVERY]
    assert pooled["n_scores"].tolist() == [2, 2]
    assert pooled["n_above_null"].tolist() == [1, 1]
    assert pooled["above_null_median"].tolist() == [-0.125, -0.125]


def test_the_check_accepts_the_groups_table_and_refuses_a_drifted_accuracy() -> None:
    runs = [_run("gemma4_31b", "base"), _run(None, None)]
    baseline = _frame(runs, MODEL, fidelity_baseline.baseline_readings)
    groups = _frame(runs, MODEL, fidelity.group_fidelity)

    fidelity_baseline.check_baseline(baseline, groups, runs)

    drifted = groups.assign(score_accuracy=groups["score_accuracy"] + 0.01)
    with pytest.raises(AssertionError, match="drifted"):
        fidelity_baseline.check_baseline(baseline, drifted, runs)
