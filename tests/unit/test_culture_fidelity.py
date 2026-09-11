from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from culture import archived_reference, fidelity, fidelity_plates
from culture.population import RunSource
from machine_bias_reproduction.data import Coverage, PreparedData
from machine_bias_reproduction.questions import QUESTIONS, resolve_question

CELLS = (
    "Germany 2018 Female 45-54 Middle Working Married",
    "Germany 1997 Male 25-34 High Student Single",
    "Mexico 2005 Female 55-64 Low Retired Widowed",
    "Russia 2017 Male <25 High NA Divorced or separated",
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

BASE = np.array(
    [
        [0.3, 0.3, 0.3, 0.1],
        [0.25, 0.25, 0.25, 0.25],
        [0.2, 0.3, 0.3, 0.2],
        [0.4, 0.3, 0.2, 0.1],
        [0.3, 0.2, 0.2, 0.3],
    ]
)


def test_a_cell_is_grouped_by_its_label_not_by_its_respondents() -> None:
    facets = fidelity.cell_facets(CELLS)

    assert list(facets.columns) == list(fidelity.FAMILIES)
    assert facets.loc[CELLS[0], "country"] == "Germany"
    assert facets.loc[CELLS[0], "wave"] == "2017-2018"
    assert facets.loc[CELLS[1], "wave"] == "1995-1997"
    assert facets.loc[CELLS[4], "marital_status"] == "Cohabiting"


def test_an_na_token_sits_out_its_own_family_and_counts_in_every_other() -> None:
    facets = fidelity.cell_facets(CELLS)

    assert facets["employment"].isna().loc[CELLS[3]]
    assert facets.loc[CELLS[3], "country"] == "Russia"
    assert fidelity.levels("employment", facets["employment"]) == [
        "Working",
        "Student",
        "Homemaker",
        "Retired",
    ]


def test_an_unparsable_label_is_refused_rather_than_grouped_wrongly() -> None:
    with pytest.raises(ValueError, match="unparsable"):
        fidelity.cell_facets(["Germany 2018 Female Middle Working Married"])


def test_dispersion_penalises_compression_and_over_spread_alike() -> None:
    assert fidelity.dispersion_score(0.5) == pytest.approx(0.5)
    assert fidelity.dispersion_score(2.0) == pytest.approx(0.5)
    assert fidelity.dispersion_score(1.0) == pytest.approx(1.0)
    assert fidelity.dispersion_score(0.0) == 0.0
    assert np.isnan(fidelity.dispersion_score(float("nan")))


def test_negative_structure_earns_no_credit_and_stops_the_whole_score() -> None:
    assert fidelity.structure_score(-0.4) == 0.0
    assert fidelity.structure_score(0.3) == pytest.approx(0.3)
    assert fidelity.geometric_mean([0.9, 0.9, 0.0]) == 0.0
    assert np.isnan(fidelity.geometric_mean([0.9, float("nan")]))
    assert fidelity.geometric_mean([0.25, 0.25]) == pytest.approx(0.25)


def test_a_group_is_scored_on_its_own_cells_and_its_own_pairs() -> None:
    values = fidelity.group_fidelity(SURVEY, MODEL)

    assert values["n_cells"] == 5
    assert values["n_pairs"] == 10
    assert values["e_q10_nemd"] <= values["e_median_nemd"] <= values["e_q90_nemd"]
    assert values["adaptability_ratio"] == pytest.approx(values["d_llm"] / values["d_wvs"])
    assert values["score_accuracy"] == pytest.approx(1.0 - values["e_mean_nemd"])
    assert values["score_center"] == pytest.approx(1.0 - values["c_center_nemd"])
    assert not values["model_flat"]
    assert values["pfs"] == pytest.approx(
        fidelity.geometric_mean(
            [values["score_accuracy"], values["score_dispersion"], values["score_structure"]]
        )
    )
    assert values["pfs_with_center"] != values["pfs"]


def test_a_model_with_one_answer_for_every_cell_scores_zero_not_undefined() -> None:
    flat = np.tile(SURVEY[0], (len(SURVEY), 1))
    values = fidelity.group_fidelity(SURVEY, flat)

    assert values["model_flat"]
    assert np.isnan(values["rho_structure"])
    assert values["score_structure"] == 0.0
    assert values["adaptability_ratio"] == 0.0
    assert values["pfs"] == 0.0


def test_a_paired_score_reads_both_conditions_off_the_same_cells() -> None:
    values = fidelity.paired_fidelity(SURVEY, MODEL, BASE)
    tuned = fidelity.group_fidelity(SURVEY, MODEL)
    reference = fidelity.group_fidelity(SURVEY, BASE)

    assert values["n_cells"] == 5
    for metric in fidelity.PAIRED_METRICS:
        assert values[metric] == pytest.approx(tuned[metric])
        assert values[f"base_{metric}"] == pytest.approx(reference[metric])
        assert values[fidelity.delta_column(metric)] == pytest.approx(
            tuned[metric] - reference[metric]
        )
    assert values["base_model_flat"] is False


def _prepared(names: tuple[str, ...], model: np.ndarray, modes: tuple[str, ...]) -> PreparedData:
    index = pd.Index(names, name="name")
    columns = list(QUESTIONS["d_happy"].answer_columns)
    survey = pd.DataFrame(SURVEY[: len(names)], index=index, columns=columns)
    answers = pd.DataFrame(model[: len(names)], index=index, columns=columns)
    return PreparedData(
        question=QUESTIONS["d_happy"],
        wvs=pd.DataFrame(),
        subpops=pd.DataFrame(),
        ntp_raw=pd.DataFrame() if "ntp" in modes else None,
        fa_raw=pd.DataFrame() if "fa" in modes else None,
        names=index,
        wvs_props=survey,
        ntp_props=answers if "ntp" in modes else None,
        fa_props=answers if "fa" in modes else None,
        social_predictors=pd.DataFrame(index=index),
        coverage=Coverage(0, 0, 0, 0, len(names), len(names)),
    )


def _run(arm: str | None, replicate: int = 1) -> RunSource:
    if arm is None:
        return RunSource("Mixtral archived", "archived", None, "Mixtral archived", None, 1)
    source = f"culture/gemma4_31b/{arm}" + ("" if replicate == 1 else f"/rep{replicate}")
    return RunSource(f"Gemma ({arm})", source, "gemma4_31b", "Gemma", arm, replicate)


def test_common_cells_is_the_intersection_over_every_run_named() -> None:
    runs = [
        _prepared(("e", "d", "c", "b", "a"), MODEL, ("fa",)),
        _prepared(("d", "c", "b", "a"), MODEL, ("fa",)),
        _prepared(("c", "b", "e"), MODEL, ("fa",)),
    ]

    assert list(fidelity.common_cells(runs)) == ["b", "c"]
    with pytest.raises(ValueError, match="no runs"):
        fidelity.common_cells([])


def test_each_replicate_is_paired_with_its_own_base_on_the_cells_every_replicate_kept() -> None:
    loaded = [
        (_run("base"), _prepared(("a", "b", "c", "d", "e"), BASE, ("ntp", "fa"))),
        (_run("base", 2), _prepared(("a", "b", "c", "d"), BASE, ("fa",))),
        (_run("german"), _prepared(("b", "c", "d", "e"), MODEL, ("ntp", "fa"))),
        (_run("german", 2), _prepared(("a", "b", "c", "d", "e"), MODEL, ("fa",))),
        (_run("german", 3), _prepared(("a", "b", "c", "d", "e"), MODEL, ("fa",))),
        (_run("spanish-mx"), _prepared(("a", "b", "c", "d", "e"), MODEL, ("fa",))),
        (_run(None), _prepared(("a", "b", "c", "d", "e"), MODEL, ("ntp", "fa"))),
    ]

    pairs = list(fidelity.paired_runs(loaded, "fa"))
    described = [
        (run.arm, run.replicate, base.replicate, list(common))
        for (run, _), (base, _), common in pairs
    ]
    assert described == [
        ("german", 1, 1, ["b", "c", "d"]),
        ("german", 2, 2, ["b", "c", "d"]),
        ("spanish-mx", 1, 1, ["a", "b", "c", "d", "e"]),
    ]

    ntp_pairs = list(fidelity.paired_runs(loaded, "ntp"))
    assert [(run.arm, list(common)) for (run, _), _, common in ntp_pairs] == [
        ("german", ["b", "c", "d", "e"])
    ]


def _identity(question: str, arm: str | None, series: str) -> dict[str, Any]:
    key = None if arm is None else "gemma4_31b"
    return {
        "question": question,
        "question_label": question,
        "model_key": key,
        "model_label": series,
        "arm": arm,
        "series": series,
        "source": "archived" if key is None else f"culture/{key}/{arm}",
        "mode": "ntp",
        "replicate": 1,
    }


def scored() -> pd.DataFrame:
    rows = []
    for question in ("d_happy", "d_trust"):
        for arm, series, answers in (
            ("base", "Gemma (as released)", BASE),
            ("german", "Gemma (german)", MODEL),
            (None, "Mixtral archived", MODEL),
        ):
            for group, level in ((fidelity.POPULATION, fidelity.EVERY_CELL), ("sex", "Female")):
                rows.append(
                    {
                        **_identity(question, arm, series),
                        "group": group,
                        "level": level,
                        **fidelity.group_fidelity(SURVEY, answers),
                    }
                )
    frame = pd.DataFrame(rows)
    frame["binding_term"] = fidelity.binding_term(frame)
    return frame


def paired() -> pd.DataFrame:
    rows = []
    for question in ("d_happy", "d_trust"):
        for group, level in ((fidelity.POPULATION, fidelity.EVERY_CELL), ("sex", "Female")):
            rows.append(
                {
                    **_identity(question, "german", "Gemma (german)"),
                    "base_source": "culture/gemma4_31b/base",
                    "group": group,
                    "level": level,
                    **fidelity.paired_fidelity(SURVEY, MODEL, BASE),
                }
            )
    frame = pd.DataFrame(rows)
    frame["binding_term"] = fidelity.binding_term(frame)
    frame["base_binding_term"] = fidelity.binding_term(frame, "base_")
    return frame


def test_the_binding_term_names_the_smallest_of_the_three() -> None:
    frame = scored()
    smallest = frame[["score_accuracy", "score_dispersion", "score_structure"]].idxmin(axis=1)
    expected = smallest.map(dict(fidelity.TERMS))

    assert frame["binding_term"].tolist() == expected.tolist()
    assert set(frame["binding_term"]) <= {"accuracy", "adaptability", "structure"}


def test_the_all_view_is_the_geometric_mean_across_the_topics() -> None:
    overall = fidelity.overall_fidelity(scored())
    pooled = overall[
        overall["series"].eq("Gemma (as released)") & overall["group"].eq(fidelity.POPULATION)
    ]
    expected = fidelity.group_fidelity(SURVEY, BASE)["pfs"]

    assert len(pooled) == 1
    assert pooled["n_questions"].iat[0] == 2
    assert pooled["pfs"].iat[0] == pytest.approx(expected)
    assert "delta_pfs_vs_base" not in overall.columns


def test_the_paired_all_view_differences_the_two_geometric_means() -> None:
    frame = paired()
    frame.loc[frame["question"].eq("d_trust"), "pfs"] = 0.9
    frame.loc[frame["question"].eq("d_happy"), "pfs"] = 0.4
    frame["base_pfs"] = 0.5
    frame["delta_pfs_vs_base"] = frame["pfs"] - frame["base_pfs"]

    overall = fidelity.overall_paired(frame)
    pooled = overall[overall["group"].eq(fidelity.POPULATION)]

    assert len(pooled) == 1
    assert pooled["pfs"].iat[0] == pytest.approx(0.6)
    assert pooled["base_pfs"].iat[0] == pytest.approx(0.5)
    assert pooled["delta_pfs_vs_base"].iat[0] == pytest.approx(0.1)
    assert fidelity.overall_paired(frame.iloc[0:0]).empty


def test_the_checks_accept_paired_rows_that_never_exceed_their_runs() -> None:
    groups = scored()

    fidelity._check_paired(groups, paired())
    fidelity._check_paired(groups, paired().iloc[0:0])

    oversized = paired()
    oversized["n_cells"] = 6
    with pytest.raises(AssertionError, match="exceed"):
        fidelity._check_paired(groups, oversized)

    uneven = pd.concat([paired(), paired().assign(replicate=2, n_cells=4)], ignore_index=True)
    with pytest.raises(AssertionError, match="different cells"):
        fidelity._check_paired(groups, uneven)


def test_the_root_holds_the_pooled_report_and_a_subfolder_holds_each_family(
    tmp_path: Path,
) -> None:
    groups = scored()
    shifts = paired()
    written = fidelity_plates.fidelity_plates(
        groups,
        fidelity.overall_fidelity(groups),
        shifts,
        fidelity.overall_paired(shifts),
        tmp_path,
    )

    assert sorted({path.suffix for path in written}) == [".pdf", ".png"]
    assert sorted(path.name for path in tmp_path.glob("*.png")) == [
        "fig_fidelity_cells_all_ntp.png",
        "fig_fidelity_cells_happiness_ntp.png",
        "fig_fidelity_cells_trust_ntp.png",
        "fig_fidelity_center_all_ntp.png",
        "fig_fidelity_center_happiness_ntp.png",
        "fig_fidelity_center_trust_ntp.png",
        "fig_fidelity_components_all_ntp.png",
        "fig_fidelity_components_happiness_ntp.png",
        "fig_fidelity_components_trust_ntp.png",
        "fig_fidelity_ranking_all_ntp.png",
        "fig_fidelity_ranking_happiness_ntp.png",
        "fig_fidelity_ranking_trust_ntp.png",
        "fig_fidelity_shift_all_ntp.png",
        "fig_fidelity_shift_happiness_ntp.png",
        "fig_fidelity_shift_trust_ntp.png",
        "fig_fidelity_subgroups_all.png",
        "fig_fidelity_subgroups_happiness.png",
        "fig_fidelity_subgroups_models_all_ntp.png",
        "fig_fidelity_subgroups_models_happiness_ntp.png",
        "fig_fidelity_subgroups_models_trust_ntp.png",
        "fig_fidelity_subgroups_trust.png",
    ]
    assert sorted(path.name for path in (tmp_path / "sex").glob("*.png")) == [
        "fig_fidelity_sex_all_ntp.png",
        "fig_fidelity_sex_components_all_ntp.png",
        "fig_fidelity_sex_components_happiness_ntp.png",
        "fig_fidelity_sex_components_trust_ntp.png",
        "fig_fidelity_sex_happiness_ntp.png",
        "fig_fidelity_sex_shift_all_ntp.png",
        "fig_fidelity_sex_shift_happiness_ntp.png",
        "fig_fidelity_sex_shift_trust_ntp.png",
        "fig_fidelity_sex_trust_ntp.png",
    ]
    assert not (tmp_path / "country").exists()


def test_the_subgroup_plate_lists_the_pooled_cells_first_and_each_family_after_a_spacer(
    tmp_path: Path,
) -> None:
    groups = scored()
    overall = fidelity.overall_fidelity(groups)

    rows = fidelity_plates.level_rows(groups)
    assert [(family, level) for family, level, _ in rows] == [
        (fidelity.POPULATION, fidelity.EVERY_CELL),
        (None, None),
        ("sex", "Female"),
    ]
    assert rows[0][2] == r"$\bf{Every\ subpopulation}$"
    assert rows[2][2] == r"$\bf{Sex}$: Female"

    shuffled = pd.DataFrame({"group": ["age", "country", "age"], "level": ["75+", "Mexico", "<25"]})
    assert [level for _, level, _ in fidelity_plates.level_rows(shuffled)] == [
        "Mexico",
        None,
        "<25",
        "75+",
    ]

    frame = fidelity_plates.subgroup_frame(groups, overall, "happiness", "ntp")
    assert set(frame["series"]) == {"Gemma (as released)", "Gemma (german)"}
    assert set(frame["group"]) == {fidelity.POPULATION, "sex"}
    assert fidelity_plates.subgroup_frame(groups, overall, "happiness", "fa").empty

    written = fidelity_plates.subgroups_plate(groups, overall, "happiness", tmp_path)
    assert sorted(path.name for path in written) == [
        "fig_fidelity_subgroups_happiness.pdf",
        "fig_fidelity_subgroups_happiness.png",
    ]
    assert fidelity_plates.subgroups_plate(groups, overall, "politics", tmp_path) == []

    offsets = fidelity_plates.variant_offsets(frame)
    assert list(offsets) == ["base", "german"]
    assert offsets["base"] < 0.0 < offsets["german"]
    named = fidelity_plates.subgroup_models_plate(groups, overall, "trust", "ntp", tmp_path)
    assert sorted(path.name for path in named) == [
        "fig_fidelity_subgroups_models_trust_ntp.pdf",
        "fig_fidelity_subgroups_models_trust_ntp.png",
    ]
    assert fidelity_plates.subgroup_models_plate(groups, overall, "trust", "fa", tmp_path) == []


def published() -> pd.DataFrame:
    rows = []
    for question in ("d_happy", "d_trust"):
        for model, mode, answers in (("GPT-4T", "ntp", MODEL), ("GPT-3", "fa", BASE)):
            for group, level in ((fidelity.POPULATION, fidelity.EVERY_CELL), ("sex", "Female")):
                rows.append(
                    {
                        **archived_reference.archived_identity(
                            model, resolve_question(question), mode
                        ),
                        "group": group,
                        "level": level,
                        **fidelity.group_fidelity(SURVEY, answers),
                    }
                )
    frame = pd.DataFrame(rows)
    frame["binding_term"] = fidelity.binding_term(frame)
    return frame


def test_the_proprietary_plate_reads_each_archived_series_in_its_own_mode(tmp_path: Path) -> None:
    groups = scored()
    overall = fidelity.overall_fidelity(groups)
    archived = published()
    off_mode = archived[archived["model_label"].eq("GPT-3")].assign(mode="ntp")
    archived = pd.concat([archived, off_mode], ignore_index=True)
    archived_overall = fidelity.overall_fidelity(archived)

    frame = fidelity_plates.proprietary_frame(groups, overall, archived, archived_overall, "trust")
    assert dict(zip(frame["series"], frame["mode"], strict=True)) == {
        "GPT-3 archived": "fa",
        "GPT-4T archived": "ntp",
    }
    pooled = fidelity_plates.proprietary_frame(groups, overall, archived, archived_overall, "all")
    assert "question" not in pooled.columns
    assert len(pooled) == 4
    starred = {fidelity_plates._proprietary_label(row) for row in frame.itertuples()}
    assert starred == {"GPT-3 archived, FA", "GPT-4T archived, NTP*"}

    written = fidelity_plates.proprietary_plate(
        groups, overall, archived, archived_overall, "trust", tmp_path
    )
    assert sorted(path.name for path in written) == [
        "fig_fidelity_subgroups_proprietary_trust.pdf",
        "fig_fidelity_subgroups_proprietary_trust.png",
    ]

    shifts = paired()
    fidelity_plates.fidelity_plates(
        groups,
        overall,
        shifts,
        fidelity.overall_paired(shifts),
        tmp_path / "with_archive",
        archived=archived,
    )
    assert sorted(path.name for path in (tmp_path / "with_archive").glob("*proprietary*.png")) == [
        "fig_fidelity_subgroups_proprietary_all.png",
        "fig_fidelity_subgroups_proprietary_happiness.png",
        "fig_fidelity_subgroups_proprietary_trust.png",
    ]


def test_replicates_are_drawn_as_one_marker_with_a_spread(tmp_path: Path) -> None:
    groups = scored()
    second = groups[groups["arm"].eq("german")].copy()
    second["replicate"] = 2
    second["pfs"] = second["pfs"] + 0.05
    shifts = paired()
    again = shifts.copy()
    again["replicate"] = 2
    again["delta_pfs_vs_base"] = again["delta_pfs_vs_base"] + 0.1
    stacked = pd.concat([groups, second], ignore_index=True)
    stacked_shifts = pd.concat([shifts, again], ignore_index=True)

    written = fidelity_plates.fidelity_plates(
        stacked,
        fidelity.overall_fidelity(stacked),
        stacked_shifts,
        fidelity.overall_paired(stacked_shifts),
        tmp_path,
    )

    assert (tmp_path / "fig_fidelity_shift_happiness_ntp.png") in written
