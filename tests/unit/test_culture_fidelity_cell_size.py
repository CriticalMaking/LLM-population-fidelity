from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from culture import fidelity, fidelity_cell_size
from culture.population import RunSource, run_identity
from machine_bias_reproduction.questions import resolve_question

MODEL_KEY = "gemma4_31b"

SURVEY = np.array(
    [
        [0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4],
        [0.25, 0.25, 0.25, 0.25],
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.7],
        [0.2, 0.2, 0.3, 0.3],
    ]
)

RELEASED = np.array(
    [
        [0.35, 0.35, 0.2, 0.1],
        [0.15, 0.25, 0.3, 0.3],
        [0.2, 0.3, 0.3, 0.2],
        [0.6, 0.2, 0.1, 0.1],
        [0.2, 0.1, 0.2, 0.5],
        [0.25, 0.25, 0.25, 0.25],
    ]
)

REVERSED = SURVEY[:, ::-1]

COUNTS = np.array([20.0, 35.0, 50.0, 30.0, 45.0, 60.0])

ANSWERS: dict[str, np.ndarray] = {
    "base": RELEASED,
    "german": REVERSED,
    "spanish-mx": 0.5 * (RELEASED + SURVEY),
}

ARM_COUNTS: dict[str, np.ndarray] = {
    "base": COUNTS,
    "german": 2.0 * COUNTS,
    "spanish-mx": COUNTS + 10.0,
}

OTHER_ANSWERS: dict[str, np.ndarray] = {
    arm: 0.5 * (answers + SURVEY[::-1]) for arm, answers in ANSWERS.items()
}

QUESTION_ANSWERS: dict[str, dict[str, np.ndarray]] = {
    "d_happy": ANSWERS,
    "d_polpos": OTHER_ANSWERS,
}

CELLS = (
    "Germany 2018 Female 45-54 Middle Working Married",
    "Germany 1997 Male 25-34 High Student Single",
    "Mexico 2005 Female 55-64 Low Retired Widowed",
    "Mexico 2017 Male <25 High Working Single",
    "United States 1995 Female 75+ Middle Homemaker Cohabiting",
    "United States 2017 Male 35-44 High Working Single",
)

MINIMUMS = (0, 30, 50, 60)

KEPT = {0: 6, 30: 5, 50: 2, 60: 1}


def _run(arm: str, replicate: int = 1) -> RunSource:
    source = f"culture/{MODEL_KEY}/{arm}"
    if replicate > 1:
        source = f"{source}/rep{replicate}"
    return RunSource(f"Gemma ({arm})", source, MODEL_KEY, "Gemma", arm, replicate)


def _frame(replicate: int = 1, variable: str = "d_happy") -> pd.DataFrame:
    question = resolve_question(variable)
    rows: list[dict[str, Any]] = []
    for arm, answers in QUESTION_ANSWERS[variable].items():
        identity = run_identity(_run(arm, replicate), question, "ntp")
        rows += fidelity_cell_size.sized_rows(
            identity, ARM_COUNTS[arm], MINIMUMS, fidelity.group_fidelity, SURVEY, answers
        )
    frame = pd.DataFrame(rows)
    scored = frame[frame["score_accuracy"].notna()]
    return frame.assign(binding_term=fidelity.binding_term(scored).reindex(frame.index))


def _paired(replicate: int = 1, variable: str = "d_happy") -> pd.DataFrame:
    question = resolve_question(variable)
    answers = QUESTION_ANSWERS[variable]
    rows: list[dict[str, Any]] = []
    for arm, tuned in answers.items():
        if arm == "base":
            continue
        identity = {
            **run_identity(_run(arm, replicate), question, "ntp"),
            "base_source": _run("base", replicate).source,
        }
        rows += fidelity_cell_size.sized_rows(
            identity,
            ARM_COUNTS[arm],
            MINIMUMS,
            fidelity.paired_fidelity,
            SURVEY,
            tuned,
            answers["base"],
        )
    return pd.DataFrame(rows)


def _both(build: Callable[[int, str], pd.DataFrame], replicate: int = 1) -> pd.DataFrame:
    return pd.concat(
        [build(replicate, variable) for variable in QUESTION_ANSWERS], ignore_index=True
    )


def _survey() -> pd.DataFrame:
    blank = float("nan")
    tables = []
    for variable, lift in (("d_happy", 0.0), ("d_polpos", -0.1)):
        exact = pd.DataFrame(
            {
                "question": variable,
                fidelity_cell_size.MINIMUM: list(MINIMUMS),
                "simulator": fidelity_cell_size.EXACT,
                "score_structure_mean": [0.8 + lift, 0.9 + lift, blank, blank],
                "pfs_mean": [0.85 + lift, 0.92 + lift, blank, blank],
            }
        )
        sampled = exact.assign(
            simulator=fidelity_cell_size.SAMPLED,
            score_structure_mean=[0.6 + lift, 0.7 + lift, blank, blank],
            pfs_mean=[0.75 + lift, 0.8 + lift, blank, blank],
        )
        tables += [exact, sampled]
    return pd.concat(tables, ignore_index=True)


def _published(frame: pd.DataFrame) -> pd.DataFrame:
    whole = frame[frame[fidelity_cell_size.MINIMUM].eq(fidelity_cell_size.EVERY_SIZE)]
    return whole.drop(columns=[fidelity_cell_size.MINIMUM, "n_respondents"]).reset_index(drop=True)


def _sized(frame: pd.DataFrame, arm: str, minimum: int) -> pd.Series:
    rows = frame[frame["arm"].eq(arm) & frame[fidelity_cell_size.MINIMUM].eq(minimum)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_a_minimum_keeps_the_cells_with_at_least_that_many_answers() -> None:
    frame = _frame()
    for minimum, kept in KEPT.items():
        row = _sized(frame, "base", minimum)
        assert row["n_cells"] == kept
        assert row["n_respondents"] == COUNTS[minimum <= COUNTS].sum()


def test_every_size_scores_what_group_fidelity_scores() -> None:
    row = _sized(_frame(), "base", 0)
    reference = fidelity.group_fidelity(SURVEY, RELEASED)
    for name in ("score_accuracy", "score_dispersion", "score_structure", "pfs", "score_center"):
        assert row[name] == pytest.approx(reference[name])


def test_a_larger_minimum_scores_the_larger_cells_alone() -> None:
    row = _sized(_frame(), "base", 30)
    keep = COUNTS >= 30
    reference = fidelity.group_fidelity(SURVEY[keep], RELEASED[keep])
    assert row["pfs"] == pytest.approx(reference["pfs"])
    assert row["rho_structure"] == pytest.approx(reference["rho_structure"])


def test_fewer_than_three_cells_go_unscored() -> None:
    frame = _frame()
    for minimum in (50, 60):
        row = _sized(frame, "base", minimum)
        assert np.isnan(row["score_accuracy"])
        assert np.isnan(row["pfs"])
        assert pd.isna(row["binding_term"])


def test_the_paired_rows_change_by_tuned_minus_released_on_the_kept_cells() -> None:
    row = _sized(_paired(), "german", 60)
    keep = ARM_COUNTS["german"] >= 60
    tuned = fidelity.group_fidelity(SURVEY[keep], REVERSED[keep])
    released = fidelity.group_fidelity(SURVEY[keep], RELEASED[keep])
    assert row[fidelity.delta_column("pfs")] == pytest.approx(tuned["pfs"] - released["pfs"])
    assert row["base_pfs"] == pytest.approx(released["pfs"])


def test_the_summary_counts_the_scored_combinations_of_the_first_run() -> None:
    frame = pd.concat([_both(_frame, 1), _both(_frame, 2)], ignore_index=True)
    paired = pd.concat([_both(_paired, 1), _both(_paired, 2)], ignore_index=True)
    summary = fidelity_cell_size.cell_size_summary(frame, paired, _survey())
    pooled = summary[summary["question"].eq(fidelity_cell_size.EVERY_QUESTION)].set_index(
        fidelity_cell_size.MINIMUM
    )
    assert pooled.loc[0, "n_combinations"] == 6
    assert pooled.loc[0, "n_scored"] == 6
    assert pooled.loc[50, "n_scored"] == 4
    assert pooled.loc[60, "n_scored"] == 2
    assert pooled.loc[30, "n_cells_min"] == 5
    assert pooled.loc[30, "n_cells_max"] == 6
    assert pooled.loc[0, "share_respondents_mean"] == pytest.approx(1.0)
    assert pooled.loc[30, "share_respondents_mean"] == pytest.approx((220.0 / 240.0 + 2.0) / 3.0)
    assert pooled.loc[50, "share_respondents_mean"] == pytest.approx(
        (110.0 / 240.0 + 440.0 / 480.0 + 185.0 / 300.0) / 3.0
    )
    first = _both(_frame, 1)
    whole = first[first[fidelity_cell_size.MINIMUM].eq(0)]
    assert pooled.loc[0, "pfs_median"] == pytest.approx(whole["pfs"].median())
    assert pooled.loc[0, "n_zero_pfs"] == int(whole["pfs"].eq(0.0).sum())
    bound = int(whole["binding_term"].eq("structure").sum())
    assert pooled.loc[0, "n_structure_binding"] == bound
    assert pooled.loc[0, "share_structure_binding"] == pytest.approx(bound / 6)


def test_a_question_row_reads_that_question_alone() -> None:
    summary = fidelity_cell_size.cell_size_summary(_both(_frame), _both(_paired), _survey())
    rows = summary[summary[fidelity_cell_size.MINIMUM].eq(0)].set_index("question")
    medians = {}
    for variable in QUESTION_ANSWERS:
        asked = _frame(1, variable)
        whole = asked[asked[fidelity_cell_size.MINIMUM].eq(0)]
        medians[variable] = whole["score_accuracy"].median()
        assert rows.loc[variable, "n_combinations"] == 3
        assert rows.loc[variable, "score_accuracy_median"] == pytest.approx(medians[variable])
        assert rows.loc[variable, "german_n_paired"] == 1
    assert medians["d_happy"] != pytest.approx(medians["d_polpos"])
    assert rows.loc[fidelity_cell_size.EVERY_QUESTION, "n_combinations"] == 6
    assert rows.loc[fidelity_cell_size.EVERY_QUESTION, "german_n_paired"] == 2


def test_the_summary_reads_the_survey_reference_and_the_paired_changes() -> None:
    summary = fidelity_cell_size.cell_size_summary(_both(_frame), _both(_paired), _survey())
    pooled = summary[summary["question"].eq(fidelity_cell_size.EVERY_QUESTION)].set_index(
        fidelity_cell_size.MINIMUM
    )
    assert pooled.loc[30, "exact_score_structure_min"] == pytest.approx(0.8)
    assert pooled.loc[30, "exact_score_structure_max"] == pytest.approx(0.9)
    assert pooled.loc[30, "exact_pfs_max"] == pytest.approx(0.92)
    assert pooled.loc[30, "sampled_score_structure_min"] == pytest.approx(0.6)
    assert pooled.loc[30, "sampled_score_structure_max"] == pytest.approx(0.7)
    assert pooled["exact_score_structure_min"].isna().loc[50]
    asked = summary[summary["question"].eq("d_polpos")].set_index(fidelity_cell_size.MINIMUM)
    assert asked.loc[30, "exact_score_structure_min"] == pytest.approx(0.8)
    assert asked.loc[30, "exact_score_structure_max"] == pytest.approx(0.8)
    paired = _both(_paired)
    tuned = paired[paired["arm"].eq("german") & paired[fidelity_cell_size.MINIMUM].eq(30)]
    assert pooled.loc[30, "german_n_paired"] == 2
    assert pooled.loc[30, "german_delta_pfs_mean"] == pytest.approx(
        tuned[fidelity.delta_column("pfs")].mean()
    )
    assert pooled.loc[60, "german_n_paired"] == 2
    assert pooled.loc[60, "spanish_mx_n_paired"] == 0
    assert pooled.loc[30, "spanish_mx_n_paired"] == 2


class _Prepared:
    def __init__(self, variable: str, answers: np.ndarray) -> None:
        self.question = resolve_question(variable)
        self.names = pd.Index(CELLS)
        columns = list(self.question.answer_columns)
        self.wvs_props = pd.DataFrame(SURVEY, index=self.names, columns=columns)
        self.answers = pd.DataFrame(answers, index=self.names, columns=columns)

    def modes(self) -> tuple[str, ...]:
        return ("ntp",)

    def props(self, mode: str) -> pd.DataFrame:
        return self.answers


def test_the_build_filters_every_array_by_the_counts_of_its_own_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = resolve_question("d_happy")
    runs = [_run("base"), _run("german")]
    loaded = [(run, _Prepared("d_happy", ANSWERS[str(run.arm)])) for run in runs]
    counted = pd.Series(COUNTS[::-1], index=list(CELLS)[::-1])
    monkeypatch.setattr(fidelity_cell_size, "loaded_runs", lambda asked, selected: loaded)
    monkeypatch.setattr(
        fidelity_cell_size,
        "run_weights",
        lambda prepared, names: counted.reindex(names).to_numpy(dtype=np.float64),
    )
    frame, paired = fidelity_cell_size.build_cell_size([question], runs, minimums=(0, 30))
    assert len(frame) == 4
    assert len(paired) == 2
    keep = COUNTS >= 30
    for arm in ("base", "german"):
        row = _sized(frame, arm, 30)
        reference = fidelity.group_fidelity(SURVEY[keep], ANSWERS[arm][keep])
        assert row["n_cells"] == 5
        assert row["pfs"] == pytest.approx(reference["pfs"])
        assert row["score_accuracy"] == pytest.approx(reference["score_accuracy"])
    tuned = _sized(paired, "german", 30)
    expected = fidelity.paired_fidelity(SURVEY[keep], REVERSED[keep], RELEASED[keep])
    assert tuned[fidelity.delta_column("score_accuracy")] == pytest.approx(
        expected[fidelity.delta_column("score_accuracy")]
    )
    assert tuned["base_source"] == _run("base").source
    assert _sized(frame, "base", 30)["binding_term"] in {"accuracy", "adaptability", "structure"}


def test_the_survey_reference_scores_a_redrawn_survey_against_the_survey(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = resolve_question("d_happy")
    shares = pd.DataFrame(SURVEY, columns=list(question.answer_columns))
    counts = pd.Series(COUNTS * 10.0)
    monkeypatch.setattr(fidelity_cell_size, "survey_cells", lambda asked: (shares, counts))
    reference = fidelity_cell_size.survey_reference(
        [question], minimums=(0, 300, 600), resamples=40, seed=7
    )
    again = fidelity_cell_size.survey_reference(
        [question], minimums=(0, 300, 600), resamples=40, seed=7
    )
    pd.testing.assert_frame_equal(reference, again)
    exact = reference[reference["simulator"].eq(fidelity_cell_size.EXACT)].set_index(
        fidelity_cell_size.MINIMUM
    )
    sampled = reference[reference["simulator"].eq(fidelity_cell_size.SAMPLED)].set_index(
        fidelity_cell_size.MINIMUM
    )
    assert list(exact["n_cells"]) == [6, 5, 1]
    lone = fidelity_cell_size.survey_reference([question], minimums=(0,), resamples=40, seed=7)
    shared = [column for column in lone.columns if column.endswith("_mean")]
    whole_rows = reference[reference[fidelity_cell_size.MINIMUM].eq(0)].reset_index(drop=True)
    pd.testing.assert_frame_equal(lone[shared], whole_rows[shared])
    assert list(exact["respondents_min"]) == [200.0, 300.0, 600.0]
    assert list(exact["respondents_max"]) == [600.0, 600.0, 600.0]
    assert exact.loc[0, "n_resamples"] == 40
    assert exact.loc[600, "n_resamples"] == 0
    whole = exact.select_dtypes("number").loc[0]
    assert whole["score_structure_low"] <= whole["score_structure_mean"]
    assert whole["score_structure_mean"] <= whole["score_structure_high"]
    assert 0.9 < whole["score_accuracy_mean"] < 1.0
    redrawn = sampled.select_dtypes("number").loc[0]
    assert redrawn["score_accuracy_mean"] < whole["score_accuracy_mean"]
    assert exact["pfs_mean"].isna().loc[600]


def test_the_plate_draws_the_minimums_that_every_combination_scores(tmp_path: Path) -> None:
    frame = _frame()
    assert fidelity_cell_size.plotted_minimums(frame) == [0, 30]
    survey = _survey().assign(
        n_cells=lambda table: table[fidelity_cell_size.MINIMUM].map(KEPT),
        score_structure_low=lambda table: table["score_structure_mean"] - 0.05,
        score_structure_high=lambda table: table["score_structure_mean"] + 0.05,
    )
    files = fidelity_cell_size.cell_size_plate(frame, survey, tmp_path)
    assert sorted(path.suffix for path in files) == [".pdf", ".png"]
    assert all(path.stat().st_size > 0 for path in files)


def test_the_check_reconciles_every_size_with_the_published_tables() -> None:
    frame, paired = _frame(), _paired()
    paired["binding_term"] = fidelity.binding_term(paired[paired["score_accuracy"].notna()])
    paired["base_binding_term"] = fidelity.binding_term(
        paired[paired["score_accuracy"].notna()], "base_"
    )
    matched = fidelity_cell_size.check_cell_size(
        frame, paired, _published(frame), _published(paired)
    )
    assert matched == 5


def test_the_check_refuses_a_score_that_drifted() -> None:
    frame = _frame()
    published = _published(frame)
    published["pfs"] = published["pfs"] + 0.01
    with pytest.raises(AssertionError, match="pfs drifted"):
        fidelity_cell_size.check_cell_size(frame, pd.DataFrame(), published, pd.DataFrame())


def test_the_check_refuses_a_larger_minimum_that_keeps_more_cells() -> None:
    frame = _frame()
    grown = frame[fidelity_cell_size.MINIMUM].eq(30) & frame["arm"].eq("base")
    frame.loc[grown, "n_cells"] = 7
    with pytest.raises(AssertionError, match="kept more cells"):
        fidelity_cell_size.check_cell_size(frame, pd.DataFrame(), _published(frame), pd.DataFrame())
