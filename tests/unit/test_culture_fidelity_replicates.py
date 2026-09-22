from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from culture import fidelity_replicates as replicates

SCORES = replicates.SCORES


def _population(values: dict[tuple[str, str, str], list[float]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (question, series, arm), runs in values.items():
        for replicate, pfs in enumerate(runs, start=1):
            rows.append(
                {
                    "question": question,
                    "series": series,
                    "arm": arm,
                    "mode": "fa",
                    "replicate": replicate,
                    "group": "population",
                    "level": "every retained cell",
                    "n_cells": 680 + replicate,
                    "adaptability_ratio": 0.5 + pfs,
                    "binding_term": "structure",
                    **dict.fromkeys(SCORES, pfs),
                }
            )
    return pd.DataFrame(rows)


GROUPS = _population(
    {
        ("d_happy", "A (base)", "base"): [0.30, 0.34],
        ("d_happy", "A (german)", "german"): [0.10, 0.12, 0.08],
        ("d_happy", "B (base)", "base"): [0.50, 0.50],
        ("d_trust", "A (base)", "base"): [0.20, 0.26],
        ("d_trust", "A (german)", "german"): [0.00, 0.06, 0.00],
        ("d_trust", "B (base)", "base"): [0.40, 0.42],
        ("d_trust", "Terra (base)", "base"): [0.60],
    }
)


def test_pooled_sd_of_two_runs_is_the_root_mean_half_squared_gap() -> None:
    table = pd.DataFrame({1: [0.1, 0.4, 0.2], 2: [0.3, 0.4, 0.5]})
    gaps = (table[1] - table[2]).to_numpy()
    assert replicates.pooled_sd(table) == pytest.approx(np.sqrt(np.mean(gaps**2 / 2.0)))


def test_pooled_sd_weights_each_pair_by_its_degrees_of_freedom() -> None:
    table = pd.DataFrame({1: [0.1, 0.2], 2: [0.3, 0.2], 3: [0.2, np.nan]})
    variance = table.var(axis=1, ddof=1).to_numpy()
    expected = np.sqrt((2 * variance[0] + 1 * variance[1]) / 3)
    assert replicates.pooled_sd(table) == pytest.approx(expected)


def test_a_single_run_never_enters_the_repeated_pairs() -> None:
    population = replicates.replicated_population(GROUPS)
    assert "Terra (base)" not in set(population["series"])
    assert len(population) == len(GROUPS) - 1


def test_spread_rows_add_up_to_the_pooled_row() -> None:
    spread = replicates.replicate_spread(GROUPS).set_index("question")
    assert list(spread.index) == ["d_happy", "d_trust", "all", "between conditions"]
    assert spread.loc["all", "degrees_of_freedom"] == 8
    assert spread.loc[["d_happy", "d_trust"], "degrees_of_freedom"].sum() == 8
    assert spread.loc["all", "n_three_runs"] == 2
    first = GROUPS[GROUPS["replicate"].eq(1) & ~GROUPS["series"].eq("Terra (base)")]
    between = np.sqrt(np.mean(np.square(first.groupby("question")["pfs"].std(ddof=1))))
    assert spread.loc["between conditions", "pfs_sd"] == pytest.approx(between)


def test_readings_count_the_zero_floor_and_a_stable_binding_term() -> None:
    retained = pd.DataFrame(
        [
            {"question": q, "series": s, "replicate": r, "subpopulation": cell}
            for (q, s), runs in GROUPS.groupby(["question", "series"])["replicate"]
            for r in runs
            for cell in ("a", "b", "c")
        ]
    )
    overall = GROUPS.assign(question="all")
    value = replicates.replicate_readings(GROUPS, overall, retained).set_index("reading")["value"]
    assert value["n_pairs"] == 6
    assert value["n_pairs_three_runs"] == 2
    assert value["pfs_zero_pairs_any_run"] == 1
    assert value["pfs_zero_pairs_every_run"] == 0
    assert value["binding_stable"] == 6
    assert value["retention_identical"] == 6
    assert value["pfs_gaps"] == 4 * 1 + 2 * 3
    assert value["pfs_sd_low"] <= value["pfs_sd"] <= value["pfs_sd_high"]


def test_paired_spread_reads_each_target_on_its_own_country() -> None:
    rows: list[dict[str, Any]] = []
    for arm, home in (("german", "Germany"), ("spanish-mx", "Mexico")):
        for question in ("d_happy", "d_trust"):
            for replicate, shift in ((1, 0.0), (2, 0.02)):
                for group, level in (("population", "every retained cell"), ("country", home)):
                    rows.append(
                        {
                            "question": question,
                            "model_key": "m",
                            "arm": arm,
                            "mode": "fa",
                            "replicate": replicate,
                            "group": group,
                            "level": level,
                            "delta_pfs_vs_base": -0.05 + shift,
                            "delta_score_center_vs_base": 0.03,
                        }
                    )
    spread = replicates.paired_spread(pd.DataFrame(rows))
    assert list(zip(spread["arm"], spread["scope"], strict=True)) == [
        ("german", "population"),
        ("german", "Germany"),
        ("spanish-mx", "population"),
        ("spanish-mx", "Mexico"),
    ]
    assert spread["n_comparisons"].eq(2).all()
    assert spread["delta_pfs_vs_base_run2"].to_numpy() == pytest.approx(-0.03)
    assert spread["delta_pfs_vs_base_sd"].to_numpy() == pytest.approx(0.02 / np.sqrt(2))
    assert spread["delta_score_center_vs_base_sign_stable"].eq(2).all()
