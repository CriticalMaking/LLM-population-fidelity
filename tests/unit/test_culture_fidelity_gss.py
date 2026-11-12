from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from culture import fidelity, fidelity_gss, fidelity_objectives, gss
from culture.population import RunSource, run_identity
from machine_bias_reproduction.questions import resolve_question

FIRST_MODEL, SECOND_MODEL = fidelity_gss.GSS_MODELS[:2]

SUBGROUPS: tuple[tuple[str, str], ...] = (
    ("SEX", "Male"),
    ("SEX", "Female"),
    ("EDUCATION", "College graduate/some postgrad"),
    ("EDUCATION", "Less than high school"),
)

QUESTIONS: dict[str, dict[str, Any]] = {
    "happy": {
        "options": ("Very happy", "Pretty happy", "Not too happy", "Refused"),
        "ordinal": (1.0, 2.0, 3.0),
        "responses": ((0.3, 0.5, 0.2), (0.25, 0.55, 0.2), (0.5, 0.4, 0.1), (0.1, 0.4, 0.5)),
    },
    "trust": {
        "options": ("Can trust", "Can't be too careful", "Depends", "Refused"),
        "ordinal": (1.0, 2.0, 1.5),
        "responses": ((0.3, 0.6, 0.1), (0.22, 0.66, 0.12), (0.5, 0.4, 0.1), (0.1, 0.75, 0.15)),
    },
    "tax": {
        "options": ("Too high", "About right", "Pays no tax", "Refused"),
        "ordinal": (1.0, 2.0, -1.0),
        "responses": (
            (0.45, 0.45, 0.1),
            (0.558, 0.342, 0.1),
            (0.27, 0.63, 0.1),
            (0.675, 0.225, 0.1),
        ),
    },
}

CELLS = (
    "Germany 2018 Female 45-54 Middle Working Married",
    "Germany 1997 Male 25-34 High Student Single",
    "Germany 2005 Female 55-64 Low Retired Widowed",
    "Mexico 2017 Male <25 High Working Single",
    "Mexico 1996 Female 35-44 Low Homemaker Married",
    "Mexico 2005 Male 65-74 Middle Retired Widowed",
    "United States 1995 Female 75+ Middle Homemaker Cohabiting",
    "United States 2017 Male 35-44 High Working Single",
    "United States 2006 Female 25-34 High Working Single",
)

WVS_SURVEY = np.array(
    [
        [0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4],
        [0.25, 0.25, 0.25, 0.25],
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.7],
        [0.2, 0.2, 0.3, 0.3],
        [0.5, 0.2, 0.2, 0.1],
        [0.1, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.2, 0.2],
    ]
)

WVS_FLAT = np.tile(np.array([0.3, 0.3, 0.2, 0.2]), (len(CELLS), 1))

WVS_RELEASED = 0.7 * WVS_SURVEY + 0.3 * WVS_FLAT


def _records() -> list[gss.GssRecord]:
    return [
        gss.GssRecord(
            qkey=qkey,
            attribute=attribute,
            group=group,
            question=f"About {qkey}?",
            options=question["options"],
            responses=question["responses"][index],
            ordinal=question["ordinal"],
            refusal_rate=0.0,
        )
        for qkey, question in QUESTIONS.items()
        for index, (attribute, group) in enumerate(SUBGROUPS)
    ]


def _letters(record: gss.GssRecord, weight: float, mass: float = 0.8) -> np.ndarray:
    flat = np.full(len(record.responses), 1.0 / len(record.responses))
    shares = weight * np.asarray(record.responses) + (1.0 - weight) * flat
    row = np.full(10, np.nan)
    row[: len(record.responses)] = 0.9 * mass * shares
    row[len(record.responses)] = 0.1 * mass
    return row


def _run(weight: float, mass: float = 0.8) -> fidelity_gss.RunRows:
    return {
        (record.qkey, record.attribute, record.group): _letters(record, weight, mass)
        for record in _records()
    }


def _runs() -> dict[tuple[str, str], fidelity_gss.RunRows]:
    return {
        (FIRST_MODEL, "base"): _run(0.5),
        (FIRST_MODEL, "subpop"): _run(1.0),
        (SECOND_MODEL, "base"): _run(0.25),
    }


def _source(key: str, arm: str, replicate: int = 1) -> RunSource:
    source = f"culture/{key}/{arm}" + (f"/rep{replicate}" if replicate > 1 else "")
    return RunSource(f"{key} ({arm})", source, key, key, arm, replicate)


def _wvs(mode: str = "ntp", replicate: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    facet = fidelity.cell_facets(CELLS).reindex(list(CELLS))
    answers = {"german": WVS_FLAT, "spanish-mx": WVS_FLAT, "subpop": WVS_SURVEY}
    levels: list[dict[str, Any]] = []
    shifts: list[dict[str, Any]] = []
    for name in ("d_happy", "d_trust"):
        question = resolve_question(name)
        for arm, tuned in answers.items():
            identity = run_identity(_source(FIRST_MODEL, arm, replicate), question, mode)
            levels += fidelity.grouped_scores(
                identity, facet, fidelity.group_fidelity, WVS_SURVEY, tuned
            )
            shifts += fidelity.grouped_scores(
                identity, facet, fidelity.paired_fidelity, WVS_SURVEY, tuned, WVS_RELEASED
            )
    groups, paired = pd.DataFrame(levels), pd.DataFrame(shifts)
    return (
        groups.assign(binding_term=fidelity.binding_term(groups)),
        paired.assign(binding_term=fidelity.binding_term(paired)),
    )


def test_the_survey_shares_follow_the_levels_and_leave_the_unordered_options_out() -> None:
    blocks = fidelity_gss.question_records(_records())

    assert list(blocks) == ["happy", "trust", "tax"]
    assert fidelity_gss.survey_array(blocks["trust"])[0] == pytest.approx([0.3, 0.1, 0.6])
    assert fidelity_gss.survey_array(blocks["tax"])[0] == pytest.approx([0.5, 0.5])
    assert fidelity_gss.survey_array(blocks["happy"]).shape == (4, 3)


def test_the_model_shares_drop_the_refusal_letter_and_follow_the_same_order() -> None:
    block = fidelity_gss.question_records(_records())["trust"]

    shares = fidelity_gss.model_array(block, _run(1.0))

    assert shares == pytest.approx(fidelity_gss.survey_array(block))
    assert shares.sum(axis=1) == pytest.approx(np.ones(4))


def test_a_subgroup_the_run_did_not_answer_is_blank() -> None:
    block = fidelity_gss.question_records(_records())["happy"]
    rows = _run(1.0)
    del rows[("happy", "SEX", "Male")]
    rows[("happy", "SEX", "Female")] = np.full(10, 0.0)

    shares = fidelity_gss.model_array(block, rows)

    assert np.isnan(shares[:2]).all()
    assert np.isfinite(shares[2:]).all()


def test_a_run_table_reads_back_as_the_letters_of_each_prompt() -> None:
    records = _records()
    probabilities = np.vstack([np.nan_to_num(_letters(record, 1.0)) for record in records])

    rows = fidelity_gss.run_rows(gss.run_frame(records, probabilities))

    assert len(rows) == len(records)
    assert rows[("trust", "SEX", "Female")][:4] == pytest.approx(_letters(records[5], 1.0)[:4])
    assert np.isnan(rows[("trust", "SEX", "Female")][4:]).all()


def test_every_run_is_scored_on_each_question_over_the_subgroups() -> None:
    groups, paired = fidelity_gss.build_gss(_records(), _runs())

    assert len(groups) == 3 * len(QUESTIONS)
    assert set(groups["question"]) == set(QUESTIONS)
    assert groups["n_cells"].eq(len(SUBGROUPS)).all()
    assert groups["mode"].eq("ntp").all()
    assert groups["group"].eq("population").all()
    assert groups["level"].eq(fidelity_gss.EVERY_SUBGROUP).all()
    assert groups[fidelity_gss.MASS].to_numpy() == pytest.approx(0.8)
    matched = groups[groups["arm"].eq("subpop")]
    assert matched["pfs"].to_numpy() == pytest.approx(1.0)
    assert matched["score_center"].to_numpy() == pytest.approx(1.0)
    block = fidelity_gss.question_records(_records())["happy"]
    expected = fidelity.group_fidelity(
        fidelity_gss.survey_array(block), fidelity_gss.model_array(block, _run(0.5))
    )
    released = groups[
        groups["question"].eq("happy")
        & groups["model_key"].eq(FIRST_MODEL)
        & groups["arm"].eq("base")
    ].iloc[0]
    assert released["pfs"] == pytest.approx(expected["pfs"])
    assert released["adaptability_ratio"] == pytest.approx(0.5)
    assert released["binding_term"] == "adaptability"
    fidelity_gss.check_gss(groups, paired)


def test_only_a_model_with_both_arms_is_paired() -> None:
    _, paired = fidelity_gss.build_gss(_records(), _runs())

    assert len(paired) == len(QUESTIONS)
    assert set(paired["model_key"]) == {FIRST_MODEL}
    assert paired["arm"].eq("subpop").all()
    assert paired["base_source"].eq(f"culture/{FIRST_MODEL}/base").all()
    change = paired[fidelity.delta_column("pfs")].to_numpy()
    assert change == pytest.approx((1.0 - paired["base_pfs"]).to_numpy())
    assert (change > 0.0).all()


def test_a_pair_is_scored_on_the_subgroups_both_arms_answered() -> None:
    runs = _runs()
    runs[(FIRST_MODEL, "base")][("happy", "SEX", "Male")] = np.full(10, 0.0)

    groups, paired = fidelity_gss.build_gss(_records(), runs)

    happy = paired[paired["question"].eq("happy")].iloc[0]
    assert happy["n_cells"] == len(SUBGROUPS) - 1
    tuned = groups[
        groups["question"].eq("happy")
        & groups["model_key"].eq(FIRST_MODEL)
        & groups["arm"].eq("subpop")
    ].iloc[0]
    assert tuned["n_cells"] == len(SUBGROUPS)
    assert paired[paired["question"].eq("trust")].iloc[0]["n_cells"] == len(SUBGROUPS)


def test_a_question_with_too_few_answered_subgroups_is_left_out() -> None:
    runs = {(FIRST_MODEL, "base"): _run(0.5)}
    for attribute, group in SUBGROUPS[:2]:
        runs[(FIRST_MODEL, "base")][("tax", attribute, group)] = np.full(10, 0.0)

    groups, paired = fidelity_gss.build_gss(_records(), runs)

    assert set(groups["question"]) == {"happy", "trust"}
    assert paired.empty


def test_the_runs_on_disk_are_found_by_their_tables(tmp_path: Path) -> None:
    records = _records()
    probabilities = np.vstack([np.nan_to_num(_letters(record, 0.5)) for record in records])
    table = gss.run_table(SECOND_MODEL, "base", tmp_path)
    table.parent.mkdir(parents=True)
    gss.run_frame(records, probabilities).to_csv(table, index=False)

    assert fidelity_gss.gss_runs(tmp_path) == [(SECOND_MODEL, "base")]
    loaded = fidelity_gss.load_runs(tmp_path)
    groups, _ = fidelity_gss.build_gss(records, loaded)
    assert set(groups["model_key"]) == {SECOND_MODEL}
    assert len(groups) == len(QUESTIONS)


def test_the_summary_reads_each_arm_over_every_question_and_over_the_shared_topics() -> None:
    groups, paired = fidelity_gss.build_gss(_records(), _runs())

    summary = fidelity_gss.gss_summary(groups, paired)

    assert set(summary["topic"]) == {"all", "Happiness", "Social trust"}
    pooled = summary[summary["topic"].eq("all")].set_index(["model_key", "arm"])
    assert pooled.loc[(FIRST_MODEL, "subpop"), "n_scores"] == len(QUESTIONS)
    assert pooled.loc[(FIRST_MODEL, "subpop"), "pfs_median"] == pytest.approx(1.0)
    assert pooled.loc[(FIRST_MODEL, "subpop"), "n_paired"] == len(QUESTIONS)
    assert pooled.loc[(FIRST_MODEL, "subpop"), "n_pfs_up"] == len(QUESTIONS)
    assert pd.isna(pooled.loc[(FIRST_MODEL, "base"), "n_paired"])
    assert pooled.loc[(SECOND_MODEL, "base"), "adaptability_ratio_median"] == pytest.approx(0.25)
    assert pooled.loc[(SECOND_MODEL, "base"), f"{fidelity_gss.MASS}_mean"] == pytest.approx(0.8)
    happiness = summary[summary["topic"].eq("Happiness")]
    assert happiness["n_scores"].eq(1).all()


def test_the_comparison_sets_each_method_on_its_own_survey_under_ntp() -> None:
    groups, paired = fidelity_gss.build_gss(_records(), _runs())
    wvs_groups, wvs_paired = _wvs()
    other_groups, other_paired = _wvs(mode="fa")
    later_groups, later_paired = _wvs(replicate=2)

    comparison = fidelity_gss.survey_comparison(
        pd.concat([wvs_groups, other_groups, later_groups]),
        pd.concat([wvs_paired, other_paired, later_paired]),
        groups,
        paired,
    )

    pooled = comparison[comparison["topic"].eq("all")]
    assert pooled[["survey", "cells", "arm"]].to_records(index=False).tolist() == [
        ("WVS", "Germany", "german"),
        ("WVS", "Mexico", "spanish-mx"),
        ("WVS", "United States", "subpop"),
        ("GSS", "SubPOP subgroups", "subpop"),
    ]
    assert pooled["n_paired"].tolist() == [2, 2, 2, len(QUESTIONS)]
    german = pooled.iloc[0]
    inside = fidelity_objectives.within_country(
        wvs_paired[wvs_paired["arm"].eq("german")], "Germany"
    )
    assert german[f"{fidelity.delta_column('pfs')}_mean"] == pytest.approx(
        inside[fidelity.delta_column("pfs")].mean()
    )
    assert german["n_pfs_down"] == 2
    assert german["base_adaptability_ratio_median"] == pytest.approx(0.7)
    assert german["adaptability_ratio_median"] == pytest.approx(0.0)
    survey = pooled.iloc[3]
    assert survey["n_pfs_up"] == len(QUESTIONS)
    assert survey["n_accuracy_up"] == len(QUESTIONS)
    assert survey["n_center_up_pfs_down"] == 0
    assert survey["n_structure_binding"] == 0
    happiness = comparison[comparison["topic"].eq("Happiness")]
    assert happiness["n_paired"].tolist() == [1, 1, 1, 1]
    assert set(comparison["topic"]) == {"all", "Happiness", "Social trust"}


def test_the_comparison_keeps_the_survey_rows_before_the_second_evaluation_has_run() -> None:
    wvs_groups, wvs_paired = _wvs()

    comparison = fidelity_gss.survey_comparison(
        wvs_groups, wvs_paired, pd.DataFrame(), pd.DataFrame()
    )

    assert set(comparison["survey"]) == {"WVS"}
    assert len(comparison[comparison["topic"].eq("all")]) == 3


def test_the_plate_stacks_every_question_over_the_shared_topics(tmp_path: Path) -> None:
    groups, _ = fidelity_gss.build_gss(_records(), _runs())

    blocks = fidelity_gss.gss_blocks(groups)
    files = fidelity_gss.gss_plate(groups, tmp_path)

    assert [header for header, _ in blocks] == [
        "All 3 GSS questions (median)",
        "GSS: happiness",
        "GSS: social trust",
    ]
    medians = blocks[0][1]
    assert medians["arm"].tolist() == ["base", "base", "subpop"]
    assert len(blocks[1][1]) == 3
    first = medians[medians["model_key"].eq(FIRST_MODEL) & medians["arm"].eq("base")].iloc[0]
    own = groups[groups["model_key"].eq(FIRST_MODEL) & groups["arm"].eq("base")]
    assert first["score_accuracy"] == pytest.approx(own["score_accuracy"].median())
    assert sorted(path.name for path in files) == [
        "fig_fidelity_components_gss_ntp.pdf",
        "fig_fidelity_components_gss_ntp.png",
    ]
    assert fidelity_gss.gss_plate(pd.DataFrame(), tmp_path) == []


def test_the_check_refuses_a_change_that_is_not_tuned_minus_released() -> None:
    groups, paired = fidelity_gss.build_gss(_records(), _runs())
    drifted = paired.assign(**{fidelity.delta_column("pfs"): 0.0})

    with pytest.raises(AssertionError):
        fidelity_gss.check_gss(groups, drifted)
    with pytest.raises(AssertionError):
        fidelity_gss.check_gss(pd.concat([groups, groups]), paired)
