from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from culture import population, population_plates
from machine_bias_reproduction.data import Coverage, PreparedData
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


def test_the_german_cells_are_the_ones_whose_label_starts_with_germany() -> None:
    assert list(population.german_names(prepared())) == list(CELLS[:3])


def test_metrics_score_every_cell_and_the_german_cells_apart() -> None:
    values = population.population_metrics(prepared(), "ntp")
    assert values is not None

    assert values["n_cells"] == 6
    assert values["n_cells_german"] == 3
    assert values["e_mean_nemd"] != values["e_mean_nemd_german"]
    assert values["adaptability_ratio"] != values["adaptability_ratio_german"]
    assert values["c_center_pop_nemd"] != values["c_center_german_nemd"]
    assert values["adaptability_ratio"] == pytest.approx(
        values["d_llm"] / values["d_wvs"]
    )


def test_structure_correlates_every_pair_and_the_german_pairs_apart() -> None:
    values = population.structure_metrics(prepared(), "fa")
    assert values is not None

    assert values["n_pairs"] == 15
    assert values["n_pairs_german"] == 3
    assert -1.0 <= values["rho_structure"] <= 1.0
    assert -1.0 <= values["pearson_structure"] <= 1.0
    assert values["rho_structure"] != values["rho_structure_german"]


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
            ("gemma4_31b", "base", "Gemma (base)"),
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
                    "e_mean_nemd": 0.08,
                    "e_mean_nemd_german": 0.05,
                    "d_wvs": 0.2,
                    "d_llm": 0.1,
                    "adaptability_ratio": 0.5,
                    "adaptability_ratio_german": 0.6,
                    "c_center_pop_nemd": 0.04,
                    "c_center_german_nemd": 0.03,
                    "rho_structure": 0.4,
                    "rho_structure_german": 0.2,
                    "pearson_structure": 0.35,
                    "pearson_structure_german": 0.25,
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
    population_plates.center_plate(table(), "ntp", "all", tmp_path)
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
        "fig_population_structure_all_ntp.png",
    ]


def test_a_run_with_no_german_reference_still_draws(tmp_path: Path) -> None:
    without = table()
    without["e_mean_nemd_german"] = np.nan

    built = population_plates.adaptability_plate(without, "ntp", "all", tmp_path)
    assert built[0].is_file()
