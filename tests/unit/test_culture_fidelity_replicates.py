from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from culture import fidelity, fidelity_replicates

RUNS: tuple[tuple[str, str, dict[str, int]], ...] = (
    ("gemma4_31b", "Gemma", {"base": 2, "german": 3, "spanish-mx": 3}),
    ("llama3_2_3b", "Llama", {"base": 2, "german": 2, "spanish-mx": 1}),
)

QUESTIONS = ("d_happy", "d_trust")


def _row(
    key: str,
    label: str,
    arm: str,
    question: str,
    mode: str,
    replicate: int,
    group: str,
    level: str,
) -> dict[str, Any]:
    return {
        "question": question,
        "question_label": question,
        "model_key": key,
        "model_label": label,
        "arm": arm,
        "series": f"{label} ({arm})",
        "source": f"culture/{key}/{arm}",
        "mode": mode,
        "replicate": replicate,
        "group": group,
        "level": level,
        "score_accuracy": 0.8 + 0.01 * replicate,
        "score_dispersion": 0.5,
        "score_structure": 0.1 * replicate,
        "score_center": 0.9,
        "pfs": 0.3 + 0.05 * replicate + (0.1 if arm == "german" else 0.0),
    }


def _groups() -> pd.DataFrame:
    rows = []
    for key, label, counts in RUNS:
        for arm, count in counts.items():
            for question in QUESTIONS:
                for replicate in range(1, count + 1):
                    for group, level in (
                        (fidelity.POPULATION, fidelity.EVERY_CELL),
                        ("sex", "Female"),
                    ):
                        rows.append(_row(key, label, arm, question, "fa", replicate, group, level))
                rows.append(
                    _row(
                        key,
                        label,
                        arm,
                        question,
                        "ntp",
                        1,
                        fidelity.POPULATION,
                        fidelity.EVERY_CELL,
                    )
                )
    rows.append(
        _row("terra", "Terra", "base", "d_happy", "fa", 1, fidelity.POPULATION, fidelity.EVERY_CELL)
    )
    return pd.DataFrame(rows)


def _pairs(table: pd.DataFrame) -> pd.DataFrame:
    pairs = fidelity_replicates.pair_rows(table)
    assert pairs["n_pairs"].eq(1).all()
    return pairs


def test_the_table_reads_one_spread_per_condition_and_question() -> None:
    table = fidelity_replicates.replicate_table(_groups())
    pairs = _pairs(table).set_index(["series", "question"])

    assert len(pairs) == 10
    assert "Llama (spanish-mx)" not in pairs.index.get_level_values("series")
    assert "Terra (base)" not in pairs.index.get_level_values("series")
    gemma = pairs.loc[("Gemma (german)", "d_happy")]
    assert gemma["n_runs"] == 3
    assert gemma["degrees_of_freedom"] == 2
    assert gemma["pfs_sd"] == pytest.approx(np.std([0.45, 0.50, 0.55], ddof=1))
    assert gemma["pfs_ci"] == pytest.approx(stats.t.ppf(0.975, 2) * gemma["pfs_sd"] / np.sqrt(3))
    assert gemma["score_center_sd"] == 0.0
    assert gemma["score_center_ci"] == 0.0
    llama = pairs.loc[("Llama (base)", "d_trust")]
    assert llama["n_runs"] == 2
    assert llama["score_accuracy_sd"] == pytest.approx(np.std([0.81, 0.82], ddof=1))
    assert llama["score_accuracy_ci"] == pytest.approx(
        stats.t.ppf(0.975, 1) * llama["score_accuracy_sd"] / np.sqrt(2)
    )


def test_the_pooled_rows_weight_each_pair_by_its_degrees_of_freedom() -> None:
    table = fidelity_replicates.replicate_table(_groups())
    pairs = _pairs(table)
    every = fidelity_replicates.EVERY

    def pooled(block: pd.DataFrame, score: str) -> float:
        freedom = block["n_runs"] - 1
        return float(np.sqrt((freedom * block[f"{score}_sd"] ** 2).sum() / freedom.sum()))

    condition = table[table["series"].eq("Gemma (german)") & table["question"].eq(every)].iloc[0]
    own = pairs[pairs["series"].eq("Gemma (german)")]
    assert condition["n_pairs"] == 2
    assert condition["n_runs"] == 6
    assert condition["degrees_of_freedom"] == 4
    assert condition["pfs_sd"] == pytest.approx(pooled(own, "pfs"))
    assert condition["pfs_ci"] == pytest.approx(
        stats.t.ppf(0.975, 4) * condition["pfs_sd"] / np.sqrt(3)
    )

    model = table[
        table["model_key"].eq("gemma4_31b") & table["arm"].eq(every) & table["question"].eq(every)
    ].iloc[0]
    mine = pairs[pairs["model_key"].eq("gemma4_31b")]
    assert model["n_pairs"] == 6
    assert model["degrees_of_freedom"] == 10
    assert model["pfs_sd"] == pytest.approx(pooled(mine, "pfs"))
    assert model["pfs_sd"] != pytest.approx(np.sqrt(np.mean(mine["pfs_sd"] ** 2)))

    question = table[
        table["model_label"].eq(every) & table["arm"].eq(every) & table["question"].eq("d_happy")
    ].iloc[0]
    assert pd.isna(question["model_key"])
    assert question["n_pairs"] == 5
    assert question["score_accuracy_sd"] == pytest.approx(
        pooled(pairs[pairs["question"].eq("d_happy")], "score_accuracy")
    )

    german = table[
        table["model_label"].eq(every) & table["arm"].eq("german") & table["question"].eq("d_happy")
    ].iloc[0]
    assert german["n_pairs"] == 2
    assert german["degrees_of_freedom"] == 3
    assert german["pfs_sd"] == pytest.approx(
        pooled(pairs[pairs["arm"].eq("german") & pairs["question"].eq("d_happy")], "pfs")
    )

    everything = table[
        table["model_label"].eq(every) & table["arm"].eq(every) & table["question"].eq(every)
    ].iloc[0]
    assert everything["n_pairs"] == 10
    assert everything["degrees_of_freedom"] == 14
    assert everything["pfs_sd"] == pytest.approx(pooled(pairs, "pfs"))
    assert everything["pfs_ci"] == pytest.approx(
        stats.t.ppf(0.975, 14) * everything["pfs_sd"] / np.sqrt(24 / 10)
    )
    assert len(table) == 10 + 5 + 2 + 6 + 3 + 2 + 1


def test_the_pooled_deviation_is_the_root_mean_square_unless_weighted() -> None:
    deviations = pd.Series([0.03, 0.04])

    assert fidelity_replicates.pooled_deviation(deviations) == pytest.approx(
        np.sqrt((0.03**2 + 0.04**2) / 2)
    )
    assert fidelity_replicates.pooled_deviation(deviations, pd.Series([1, 3])) == pytest.approx(
        np.sqrt((0.03**2 + 3 * 0.04**2) / 4)
    )


def test_the_spread_pools_the_run_to_run_deviation_across_questions() -> None:
    groups = _groups()

    spread = fidelity_replicates.replicate_spread(groups)
    own = spread[spread["series"].eq("Gemma (german)")].set_index("question")

    assert list(own.index) == [*QUESTIONS, fidelity_replicates.EVERY]
    assert own["n_runs"].tolist() == [3, 3, 3]
    runs = groups[
        groups["series"].eq("Gemma (german)")
        & groups["mode"].eq("fa")
        & groups["group"].eq(fidelity.POPULATION)
        & groups["question"].eq("d_happy")
    ]
    assert own.loc["d_happy", "score_accuracy_sd"] == pytest.approx(
        runs["score_accuracy"].std(ddof=1)
    )
    assert own.loc[fidelity_replicates.EVERY, "pfs_sd"] == pytest.approx(
        own.loc["d_happy", "pfs_sd"]
    )
    alone = spread[spread["series"].eq("Llama (spanish-mx)")]
    assert alone["n_runs"].tolist() == [1, 1, 1]
    assert alone["pfs_sd"].isna().all()
