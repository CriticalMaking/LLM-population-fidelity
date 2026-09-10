from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from culture import population, population_plates
from machine_bias_reproduction.data import Coverage, PreparedData
from machine_bias_reproduction.plates import CATEGORICAL_MARKERS
from machine_bias_reproduction.questions import QUESTIONS

COLUMNS = ("A", "B", "C", "D")

CELLS = (
    "Germany 2018 Female 45-54 Middle Working Married",
    "Germany 2018 Male 25-34 High Working Single",
    "Germany 2006 Female 55-64 Low Retired Widowed",
    "Mexico 2018 Female 45-54 Middle Working Married",
    "Mexico 2005 Male 25-34 High Working Single",
    "Russia 2017 Female 35-44 Low Working Married",
)


def frame(rows: list[list[float]], names: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(rows, index=names, columns=list(COLUMNS))


def prepared(with_ntp: bool = True, with_fa: bool = True) -> PreparedData:
    names = pd.Index(CELLS, name="name")
    survey = frame(
        [
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4],
            [0.25, 0.25, 0.25, 0.25],
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.7],
            [0.3, 0.3, 0.2, 0.2],
        ],
        names,
    )
    model = frame(
        [
            [0.35, 0.35, 0.2, 0.1],
            [0.15, 0.25, 0.3, 0.3],
            [0.2, 0.3, 0.3, 0.2],
            [0.6, 0.2, 0.1, 0.1],
            [0.2, 0.1, 0.2, 0.5],
            [0.3, 0.25, 0.25, 0.2],
        ],
        names,
    )
    return PreparedData(
        question=QUESTIONS["d_happy"],
        wvs=pd.DataFrame(),
        subpops=pd.DataFrame(),
        ntp_raw=pd.DataFrame({"mass": [1.0]}) if with_ntp else None,
        fa_raw=pd.DataFrame() if with_fa else None,
        names=names,
        wvs_props=survey,
        ntp_props=model if with_ntp else None,
        fa_props=model if with_fa else None,
        social_predictors=pd.DataFrame(index=names),
        coverage=Coverage(0, 0, 0, 0, len(names), len(names)),
    )


def test_the_reference_cells_are_the_ones_whose_label_starts_with_that_country() -> None:
    assert list(population.country_names(prepared(), "Germany")) == list(CELLS[:3])
    assert list(population.country_names(prepared(), "Mexico")) == list(CELLS[3:5])
    assert list(population.country_names(prepared(), "Japan")) == []


def test_metrics_score_every_cell_and_each_reference_country_apart() -> None:
    values = population.population_metrics(prepared(), "ntp")
    assert values is not None

    assert values["n_cells"] == 6
    assert values["n_cells_german"] == 3
    assert values["n_cells_mexican"] == 2
    assert values["adaptability_ratio"] == pytest.approx(
        values["d_llm"] / values["d_wvs"]
    )
    for key in population.REFERENCE_COUNTRIES:
        assert values["e_mean_nemd"] != values[f"e_mean_nemd_{key}"]
        assert values["adaptability_ratio"] != values[f"adaptability_ratio_{key}"]
        assert values["c_center_pop_nemd"] != values[f"c_center_{key}_nemd"]
    assert values["e_mean_nemd_german"] != values["e_mean_nemd_mexican"]
    assert values["c_center_german_nemd"] != values["c_center_mexican_nemd"]


def test_metrics_report_a_reference_country_absent_from_the_run_as_empty() -> None:
    values = population._reference_metrics(
        prepared(), prepared().props("ntp"), "japanese", "Japan", "ntp"
    )

    assert values["n_cells_japanese"] == 0
    assert np.isnan(values["e_mean_nemd_japanese"])
    assert np.isnan(values["adaptability_ratio_japanese"])


def test_structure_correlates_every_pair_and_each_reference_country_apart() -> None:
    values = population.structure_metrics(prepared(), "fa")
    assert values is not None

    assert values["n_pairs"] == 15
    assert values["n_pairs_german"] == 3
    assert -1.0 <= values["rho_structure"] <= 1.0
    assert -1.0 <= values["pearson_structure"] <= 1.0
    assert values["rho_structure"] != values["rho_structure_german"]

    assert values["n_cells_mexican"] == 2
    assert values["n_pairs_mexican"] == 0
    assert np.isnan(values["rho_structure_mexican"])


def test_a_run_without_next_token_probability_scores_full_answers_only() -> None:
    fa_only = prepared(with_ntp=False)

    assert population.population_metrics(fa_only, "ntp") is None
    assert population.structure_metrics(fa_only, "ntp") is None
    assert population.population_metrics(fa_only, "fa") is not None
    assert population.structure_metrics(fa_only, "fa") is not None


def table() -> pd.DataFrame:
    rows = []
    for question, label in (("d_happy", "Happiness"), ("d_trust", "Trust")):
        for key, arm, series in (
            ("gemma4_31b", "base", "Gemma (as released)"),
            ("gemma4_31b", "german", "Gemma (german)"),
            (None, None, "Mixtral archived"),
        ):
            rows.append(
                {
                    "question": question,
                    "question_label": label,
                    "model_key": key,
                    "model_label": series,
                    "arm": arm,
                    "series": series,
                    "source": "archived" if key is None else f"culture/{key}/{arm}",
                    "mode": "ntp",
                    "n_cells": 6,
                    "n_cells_german": 3,
                    "n_cells_mexican": 2,
                    "e_mean_nemd": 0.08,
                    "e_mean_nemd_german": 0.05,
                    "e_mean_nemd_mexican": 0.11,
                    "d_wvs": 0.2,
                    "d_llm": 0.1,
                    "adaptability_ratio": 0.5,
                    "adaptability_ratio_german": 0.6,
                    "adaptability_ratio_mexican": 0.45,
                    "c_center_pop_nemd": 0.04,
                    "c_center_german_nemd": 0.03,
                    "c_center_mexican_nemd": 0.05,
                    "rho_structure": 0.4,
                    "rho_structure_german": 0.2,
                    "rho_structure_mexican": 0.1,
                    "pearson_structure": 0.35,
                    "pearson_structure_german": 0.25,
                    "pearson_structure_mexican": 0.15,
                }
            )
    return pd.DataFrame(rows)


def test_every_view_draws_one_panel_per_question_it_covers(tmp_path: Path) -> None:
    built = population_plates.adaptability_plate(table(), "ntp", "all", tmp_path)
    assert sorted(path.suffix for path in built) == [".pdf", ".png"]
    assert built[0].stem == "fig_population_adaptability_all_ntp"

    single = population_plates.adaptability_plate(table(), "ntp", "trust", tmp_path)
    assert single[0].stem == "fig_population_adaptability_trust_ntp"

    assert set(population_plates.VIEWS) == {
        "all",
        "happiness",
        "politics",
        "religious",
        "trust",
    }


def test_the_three_families_each_write_a_plate(tmp_path: Path) -> None:
    population_plates.adaptability_plate(table(), "ntp", "all", tmp_path)
    for country in population.REFERENCE_COUNTRIES:
        population_plates.center_plate(table(), "ntp", "all", tmp_path, country=country)
    population_plates.structure_plate(
        table(),
        "ntp",
        "all",
        "rho_structure",
        "Spearman",
        "fig_population_structure",
        tmp_path,
    )

    written = sorted(path.name for path in tmp_path.glob("*.png"))
    assert written == [
        "fig_population_adaptability_all_ntp.png",
        "fig_population_center_all_ntp.png",
        "fig_population_center_mexican_all_ntp.png",
        "fig_population_structure_all_ntp.png",
    ]


def test_each_variant_of_a_model_gets_its_own_fill() -> None:
    fills = {
        arm: population_plates.series_style("Gemma", "gemma4_31b", arm)["fillstyle"]
        for arm in ("base", "german", "spanish-mx")
    }

    assert fills["base"] == "none"
    assert len(set(fills.values())) == 3, "two variants of one model share a fill"


def test_a_tuned_variant_reaches_only_to_the_country_it_targets() -> None:
    assert population_plates.reference_keys("german") == ("german",)
    assert population_plates.reference_keys("spanish-mx") == ("mexican",)
    assert population_plates.reference_keys("turkish") == ()
    assert population_plates.reference_keys("base") == tuple(population.REFERENCE_COUNTRIES)
    assert population_plates.reference_keys(None) == tuple(population.REFERENCE_COUNTRIES)


def test_every_reference_country_gets_its_own_tick(tmp_path: Path) -> None:
    assert set(population_plates.REFERENCE_TICKS) == set(population.REFERENCE_COUNTRIES)
    assert set(population_plates.CENTER_STEMS) == set(population.REFERENCE_COUNTRIES)

    for along in ("x", "y"):
        marks = [
            population_plates.REFERENCE_TICKS[key][along]["marker"]
            for key in population.REFERENCE_COUNTRIES
        ]
        assert len(set(marks)) == len(marks), f"two reference ticks share a glyph on {along}"
        assert not set(marks) & set(CATEGORICAL_MARKERS), "a reference tick wears a series glyph"

    labels = [handle.get_label() for handle in population_plates.reference_handles("x")]
    assert labels == [
        "same run over the German cells alone",
        "same run over the Mexican cells alone",
    ]


def _replicated() -> pd.DataFrame:
    first = table()
    first["replicate"] = 1
    second = first[first["series"].eq("Gemma (german)")].copy()
    second["replicate"] = 2
    second["e_mean_nemd"] = 0.12
    second["adaptability_ratio"] = 0.7
    return pd.concat([first, second], ignore_index=True)


def test_across_replicates_averages_each_series_and_keeps_the_spread() -> None:
    collapsed = population.across_replicates(
        _replicated(), spread=("e_mean_nemd", "adaptability_ratio", "absent")
    )

    assert len(collapsed) == len(table())
    assert "replicate" not in collapsed.columns
    german = collapsed[
        collapsed["series"].eq("Gemma (german)") & collapsed["question"].eq("d_happy")
    ].iloc[0]
    assert german["n_replicates"] == 2
    assert german["e_mean_nemd"] == pytest.approx(0.10)
    assert german["e_mean_nemd_sd"] == pytest.approx(np.std([0.08, 0.12], ddof=1))
    assert german["adaptability_ratio"] == pytest.approx(0.6)
    assert german["source"] == "culture/gemma4_31b/german"
    single = collapsed[collapsed["series"].eq("Mixtral archived")].iloc[0]
    assert single["n_replicates"] == 1
    assert np.isnan(single["e_mean_nemd_sd"])
    assert "absent_sd" not in collapsed.columns
    assert population.across_replicates(table()).equals(table())


def test_replicated_runs_draw_one_marker_with_a_spread_bar(tmp_path: Path) -> None:
    built = population_plates.adaptability_plate(_replicated(), "ntp", "all", tmp_path)
    assert built[0].is_file()
    assert population_plates.replicated(population_plates._view_frame(_replicated(), "ntp", "all"))
    assert not population_plates.replicated(table())


def test_a_run_with_no_reference_columns_still_draws(tmp_path: Path) -> None:
    without = table()
    without["e_mean_nemd_german"] = np.nan
    without = without.drop(columns=["e_mean_nemd_mexican"])

    built = population_plates.adaptability_plate(without, "ntp", "all", tmp_path)
    assert built[0].is_file()
