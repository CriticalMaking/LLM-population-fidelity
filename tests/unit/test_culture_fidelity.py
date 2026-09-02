from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from culture import fidelity, fidelity_plates

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


def scored() -> pd.DataFrame:
    rows = []
    for question in ("d_happy", "d_trust"):
        for key, arm, series in (
            ("gemma4_31b", "base", "Gemma (base)"),
            ("gemma4_31b", "german", "Gemma (german)"),
            (None, None, "Mixtral archived"),
        ):
            for group, level in ((fidelity.POPULATION, fidelity.EVERY_CELL), ("sex", "Female")):
                rows.append(
                    {
                        "question": question,
                        "question_label": question,
                        "model_key": key,
                        "model_label": series,
                        "arm": arm,
                        "series": series,
                        "source": "archived" if key is None else f"culture/{key}/{arm}",
                        "mode": "ntp",
                        "group": group,
                        "level": level,
                        "n_cells": 5,
                        "n_pairs": 10,
                        "e_mean_nemd": 0.2,
                        "e_q10_nemd": 0.1,
                        "e_q25_nemd": 0.15,
                        "e_median_nemd": 0.2,
                        "e_q75_nemd": 0.25,
                        "e_q90_nemd": 0.3,
                        "d_wvs": 0.2,
                        "d_llm": 0.1,
                        "adaptability_ratio": 0.5,
                        "rho_structure": 0.4,
                        "rho_structure_p_value": 0.01,
                        "model_flat": False,
                        "c_center_nemd": 0.05,
                        "score_accuracy": 0.8,
                        "score_dispersion": 0.5,
                        "score_structure": 0.25 if arm == "german" else 0.4,
                        "score_center": 0.95,
                        "pfs": 0.4 if arm == "german" else 0.5,
                        "pfs_with_center": 0.5 if arm == "german" else 0.6,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["binding_term"] = fidelity.binding_term(frame)
    return fidelity.base_deltas(frame, ("model_key", "mode", "question", "group", "level"))


def test_the_binding_term_names_the_smallest_of_the_three() -> None:
    assert set(scored()["binding_term"]) == {"structure"}


def test_a_finetuned_variant_is_differenced_against_its_own_base_and_a_reference_is_not() -> None:
    frame = scored()
    tuned = frame[frame["arm"].eq("german")]
    reference = frame[frame["series"].eq("Mixtral archived")]

    assert np.allclose(tuned["delta_pfs_vs_base"].to_numpy(), -0.1)
    assert reference["delta_pfs_vs_base"].isna().all()


def test_the_all_view_is_the_geometric_mean_across_the_topics() -> None:
    overall = fidelity.overall_fidelity(scored())
    pooled = overall[
        overall["series"].eq("Gemma (base)") & overall["group"].eq(fidelity.POPULATION)
    ]

    assert len(pooled) == 1
    assert pooled["n_questions"].iat[0] == 2
    assert pooled["pfs"].iat[0] == pytest.approx(0.5)
    assert pooled["binding_term"].iat[0] == "structure"


def test_the_root_holds_the_pooled_report_and_a_subfolder_holds_each_family(
    tmp_path: Path,
) -> None:
    groups = scored()
    written = fidelity_plates.fidelity_plates(groups, fidelity.overall_fidelity(groups), tmp_path)

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
