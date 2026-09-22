from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from machine_bias_reproduction.config import FIRST_REPLICATE, GLOBAL_SEED, OUTPUTS_ROOT
from machine_bias_reproduction.questions import Question, resolve_questions

from .fidelity import POPULATION, loaded_runs
from .fidelity_baseline import HOME
from .population import REPLICATE, RunSource
from .registry import BASE_ARM

REPLICATE_MODE = "fa"

SCORES: tuple[str, ...] = (
    "score_accuracy",
    "score_dispersion",
    "score_structure",
    "pfs",
    "score_center",
)

PAIR_KEYS: tuple[str, ...] = ("question", "series")

SUBGROUP_KEYS: tuple[str, ...] = ("question", "series", "group", "level")

PAIRED_KEYS: tuple[str, ...] = ("question", "model_key")

PAIRED_DELTAS: tuple[str, ...] = ("delta_pfs_vs_base", "delta_score_center_vs_base")

COUNTRY_FAMILY = "country"

EVERY_QUESTION = "all"

BETWEEN = "between conditions"

COUNTS: tuple[str, ...] = ("n_pairs", "degrees_of_freedom", "n_three_runs")

MIN_RUNS = 2

MIN_ORDERED = 3

GAP_BOUNDS: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)

LARGEST_PAIRS = 3

BOOTSTRAP_DRAWS = 20_000

CONFIDENCE = 0.95

REPLICATES_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_replicates.csv"

REPLICATES_READINGS_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_replicates_readings.csv"

REPLICATES_PAIRED_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_replicates_paired.csv"


def repeated(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    runs = frame.groupby(list(keys))[REPLICATE].transform("nunique")
    return frame[runs.ge(MIN_RUNS)]


def by_run(frame: pd.DataFrame, keys: Sequence[str], column: str) -> pd.DataFrame:
    return frame.pivot_table(index=list(keys), columns=REPLICATE, values=column, aggfunc="first")


def degrees_of_freedom(table: pd.DataFrame) -> pd.Series:
    return table.notna().sum(axis=1) - 1


def pooled_sd(table: pd.DataFrame) -> float:
    freedom = degrees_of_freedom(table)
    total = freedom.sum()
    if not total:
        return float("nan")
    variance = table.var(axis=1, ddof=1)
    return float(np.sqrt((freedom * variance).sum() / total))


def spearman(first: pd.Series, second: pd.Series) -> float:
    return float(spearmanr(first, second).statistic)


def replicated_population(groups: pd.DataFrame) -> pd.DataFrame:
    fa = groups[groups["mode"].eq(REPLICATE_MODE) & groups["group"].eq(POPULATION)]
    return repeated(fa, PAIR_KEYS)


def replicated_subgroups(groups: pd.DataFrame) -> pd.DataFrame:
    fa = groups[groups["mode"].eq(REPLICATE_MODE) & ~groups["group"].eq(POPULATION)]
    return repeated(fa, SUBGROUP_KEYS)


def orderings(pfs: pd.DataFrame) -> dict[str, Any]:
    thrice = pfs.dropna() if pfs.shape[1] > MIN_RUNS else pfs.iloc[0:0]
    reading: dict[str, Any] = {
        "rho_pfs_runs_1_2": spearman(pfs[FIRST_REPLICATE], pfs[FIRST_REPLICATE + 1]),
        "n_three_runs": len(thrice),
        "rho_pfs_three_runs_min": np.nan,
        "rho_pfs_three_runs_max": np.nan,
    }
    if len(thrice) >= MIN_ORDERED:
        rhos = [spearman(thrice[a], thrice[b]) for a, b in combinations(thrice.columns, 2)]
        reading.update(rho_pfs_three_runs_min=min(rhos), rho_pfs_three_runs_max=max(rhos))
    return reading


def _spread_row(question: str, block: pd.DataFrame, keys: Sequence[str]) -> dict[str, Any]:
    pfs = by_run(block, keys, "pfs")
    return {
        "question": question,
        "n_pairs": len(pfs),
        "degrees_of_freedom": int(degrees_of_freedom(pfs).sum()),
        **{f"{score}_sd": pooled_sd(by_run(block, keys, score)) for score in SCORES},
        **orderings(pfs),
    }


def between_conditions(population: pd.DataFrame) -> pd.Series:
    first = population[population[REPLICATE].eq(FIRST_REPLICATE)]
    spread = first.groupby("question")[list(SCORES)].std(ddof=1)
    return spread.pow(2).mean().pow(0.5)


def replicate_spread(groups: pd.DataFrame) -> pd.DataFrame:
    population = replicated_population(groups)
    rows = [
        _spread_row(str(question), block, ["series"])
        for question, block in population.groupby("question", sort=True)
    ]
    rows.append(_spread_row(EVERY_QUESTION, population, PAIR_KEYS))
    between = between_conditions(population)
    rows.append({"question": BETWEEN, **{f"{score}_sd": between[score] for score in SCORES}})
    return pd.DataFrame(rows).astype(dict.fromkeys(COUNTS, "Int64"))


def bootstrap_interval(table: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float]:
    freedom = degrees_of_freedom(table).to_numpy(dtype=np.float64)
    variance = table.var(axis=1, ddof=1).to_numpy(dtype=np.float64)
    picks = rng.integers(0, len(variance), size=(BOOTSTRAP_DRAWS, len(variance)))
    draws = np.sqrt((freedom[picks] * variance[picks]).sum(axis=1) / freedom[picks].sum(axis=1))
    tail = (1.0 - CONFIDENCE) / 2.0 * 100.0
    low, high = np.percentile(draws, [tail, 100.0 - tail])
    return float(low), float(high)


def run_gaps(table: pd.DataFrame) -> pd.Series:
    gaps = [(table[a] - table[b]).abs().dropna() for a, b in combinations(table.columns, 2)]
    return pd.concat(gaps)


def retained_cells(
    runs: Sequence[RunSource],
    questions: list[Question] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for question in questions or resolve_questions(None):
        for run, prepared in loaded_runs(question, runs):
            if REPLICATE_MODE not in prepared.modes():
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "question": question.var,
                        "series": run.series,
                        REPLICATE: run.replicate,
                        "subpopulation": prepared.names,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def repeated_runs(runs: Sequence[RunSource], groups: pd.DataFrame) -> list[RunSource]:
    series = set(replicated_population(groups)["series"])
    return [run for run in runs if run.series in series]


def _population_readings(population: pd.DataFrame) -> dict[str, float]:
    pfs = by_run(population, PAIR_KEYS, "pfs")
    runs = pfs.notna().sum(axis=1)
    conditions = population.groupby("series")[REPLICATE].nunique()
    reading: dict[str, float] = {
        "n_conditions": conditions.size,
        "n_conditions_three_runs": int(conditions.eq(3).sum()),
        "n_pairs": len(pfs),
        "n_pairs_two_runs": int(runs.eq(2).sum()),
        "n_pairs_three_runs": int(runs.eq(3).sum()),
        "degrees_of_freedom": int(degrees_of_freedom(pfs).sum()),
    }
    between = between_conditions(population)
    rng = np.random.default_rng(GLOBAL_SEED)
    for score in SCORES:
        table = by_run(population, PAIR_KEYS, score)
        within = pooled_sd(table)
        share = within**2 / between[score] ** 2
        low, high = bootstrap_interval(table, rng)
        reading.update(
            {
                f"{score}_sd": within,
                f"{score}_sd_low": low,
                f"{score}_sd_high": high,
                f"{score}_between_sd": float(between[score]),
                f"{score}_variance_share": share,
                f"{score}_icc": 1.0 - share,
            }
        )
    return reading


def _gap_readings(pfs: pd.DataFrame) -> dict[str, float]:
    gaps = run_gaps(pfs)
    freedom = degrees_of_freedom(pfs)
    weight = (freedom * pfs.var(axis=1, ddof=1)).sort_values(ascending=False)
    reading: dict[str, float] = {
        "pfs_gaps": len(gaps),
        "pfs_gap_median": float(gaps.median()),
        "pfs_gap_max": float(gaps.max()),
        "pfs_largest_pairs_variance_share": float(weight.iloc[:LARGEST_PAIRS].sum() / weight.sum()),
    }
    for bound in GAP_BOUNDS:
        reading[f"pfs_gap_share_at_most_{bound:.2f}"] = float(gaps.le(bound).mean())
    return reading


def _floor_readings(pfs: pd.DataFrame) -> dict[str, float]:
    zero = pfs.le(0.0) & pfs.notna()
    touches = zero.any(axis=1)
    return {
        "pfs_zero_pairs_any_run": int(touches.sum()),
        "pfs_zero_pairs_every_run": int(zero.sum(axis=1).eq(pfs.notna().sum(axis=1)).sum()),
        "pfs_sd_never_zero": pooled_sd(pfs[~touches]),
        "pfs_sd_zero_in_a_run": pooled_sd(pfs[touches]),
    }


def _stability_readings(population: pd.DataFrame) -> dict[str, float]:
    binding = by_run(population, PAIR_KEYS, "binding_term")
    stable = binding.nunique(axis=1).eq(1)
    kept = binding[stable][FIRST_REPLICATE].value_counts()
    reading: dict[str, float] = {
        "binding_stable": int(stable.sum()),
        "binding_stable_structure": int(kept.get("structure", 0)),
        "binding_stable_adaptability": int(kept.get("adaptability", 0)),
        "binding_stable_accuracy": int(kept.get("accuracy", 0)),
    }
    released = population["arm"].eq(BASE_ARM)
    for name, block in (("released", population[released]), ("finetuned", population[~released])):
        ratio = by_run(block, PAIR_KEYS, "adaptability_ratio")
        side = ratio.lt(1.0).where(ratio.notna())
        reading[f"direction_pairs_{name}"] = len(ratio)
        reading[f"direction_stable_{name}"] = int(side.nunique(axis=1).eq(1).sum())
    return reading


def _retention_readings(population: pd.DataFrame, retained: pd.DataFrame) -> dict[str, float]:
    pfs = by_run(population, PAIR_KEYS, "pfs")
    sets = (
        retained.groupby([*PAIR_KEYS, REPLICATE])["subpopulation"]
        .agg(frozenset)
        .unstack(REPLICATE)
        .reindex(pfs.index)
    )
    identical = sets.apply(
        lambda row: len({value for value in row if isinstance(value, frozenset)}) == 1, axis=1
    )
    counts = by_run(population, PAIR_KEYS, "n_cells")
    gap = (counts.max(axis=1) - counts.min(axis=1))[~identical]
    return {
        "retention_identical": int(identical.sum()),
        "retention_count_gap_max": int(gap.max()) if len(gap) else 0,
        "retained_cells_min": int(counts.min().min()),
        "retained_cells_max": int(counts.max().max()),
    }


def _subgroup_readings(subgroups: pd.DataFrame) -> dict[str, float]:
    pfs = by_run(subgroups, SUBGROUP_KEYS, "pfs")
    floor = (pfs.le(0.0) & pfs.notna()).any(axis=1)
    return {
        "subgroup_scores": len(pfs),
        "subgroup_zero_share": float(floor.mean()),
        "subgroup_pfs_sd": pooled_sd(pfs),
        "subgroup_pfs_sd_never_zero": pooled_sd(pfs[~floor]),
        "subgroup_score_center_sd": pooled_sd(by_run(subgroups, SUBGROUP_KEYS, "score_center")),
    }


def _summary_readings(overall: pd.DataFrame) -> dict[str, float]:
    fa = overall[overall["mode"].eq(REPLICATE_MODE) & overall["group"].eq(POPULATION)]
    summary = repeated(fa, ["series"])
    return {
        f"summary_{score}_sd": pooled_sd(by_run(summary, ["series"], score)) for score in SCORES
    }


def replicate_readings(
    groups: pd.DataFrame,
    overall: pd.DataFrame,
    retained: pd.DataFrame,
) -> pd.DataFrame:
    population = replicated_population(groups)
    pfs = by_run(population, PAIR_KEYS, "pfs")
    readings = {
        **_population_readings(population),
        **_gap_readings(pfs),
        **_floor_readings(pfs),
        **_stability_readings(population),
        **_retention_readings(population, retained),
        **_subgroup_readings(replicated_subgroups(groups)),
        **_summary_readings(overall),
    }
    return pd.DataFrame({"reading": list(readings), "value": list(readings.values())})


def paired_spread(paired: pd.DataFrame) -> pd.DataFrame:
    fa = paired[paired["mode"].eq(REPLICATE_MODE)]
    rows: list[dict[str, Any]] = []
    for arm, home in HOME.items():
        scopes = {
            POPULATION: fa["group"].eq(POPULATION),
            home: fa["group"].eq(COUNTRY_FAMILY) & fa["level"].eq(home),
        }
        for scope, keep in scopes.items():
            block = repeated(fa[keep & fa["arm"].eq(arm)], PAIRED_KEYS)
            row: dict[str, Any] = {
                "arm": arm,
                "scope": scope,
                "n_comparisons": block.groupby(list(PAIRED_KEYS)).ngroups,
            }
            for delta in PAIRED_DELTAS:
                table = by_run(block, PAIRED_KEYS, delta)
                for replicate in table.columns:
                    row[f"{delta}_run{replicate}"] = float(table[replicate].mean())
                row[f"{delta}_sd"] = pooled_sd(table)
                sign = (table.gt(0.0).astype(int) - table.lt(0.0).astype(int)).where(table.notna())
                row[f"{delta}_sign_stable"] = int(sign.nunique(axis=1).eq(1).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def check_replicates(
    spread: pd.DataFrame,
    readings: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    per_question = spread[~spread["question"].isin({EVERY_QUESTION, BETWEEN})]
    every = spread[spread["question"].eq(EVERY_QUESTION)].iloc[0]
    assert per_question["degrees_of_freedom"].sum() == every["degrees_of_freedom"], (
        "the questions do not add up to the pooled degrees of freedom"
    )
    assert per_question["n_pairs"].sum() == every["n_pairs"], "a pair was pooled twice"
    value = readings.set_index("reading")["value"]
    assert value["n_pairs"] == value["n_pairs_two_runs"] + value["n_pairs_three_runs"], (
        "a pair holds neither two nor three runs"
    )
    for score in SCORES:
        assert value[f"{score}_sd_low"] <= value[f"{score}_sd"] <= value[f"{score}_sd_high"], (
            f"the {score} interval misses its own estimate"
        )
    sds = spread[[f"{score}_sd" for score in SCORES]].to_numpy(dtype=np.float64)
    assert np.isfinite(sds).all() and (sds >= 0.0).all(), "a pooled deviation is not a deviation"
    assert paired["n_comparisons"].nunique() == 1, "the paired scopes hold different comparisons"
