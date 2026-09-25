from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t

from machine_bias_reproduction.config import OUTPUTS_ROOT

from .fidelity import POPULATION
from .population import CONFIDENCE, REPLICATE

REPLICATES_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_replicates.csv"

SPREAD_MODE = "fa"

EVERY = "all"

SPREAD_SCORES: tuple[str, ...] = (
    "score_accuracy",
    "score_dispersion",
    "score_structure",
    "pfs",
    "score_center",
)

CONDITION: tuple[str, ...] = ("model_key", "model_label", "arm", "series")

QUESTION: tuple[str, ...] = ("question", "question_label")

IDENTITY: tuple[str, ...] = (*CONDITION, *QUESTION)

COUNTS: tuple[str, ...] = ("n_pairs", "n_runs", "degrees_of_freedom")

POOLED_KEYS: tuple[tuple[str, ...], ...] = (
    CONDITION,
    ("model_key", "model_label"),
    ("arm", *QUESTION),
    ("arm",),
    QUESTION,
    (),
)


def spread_column(score: str) -> str:
    return f"{score}_sd"


def interval_column(score: str) -> str:
    return f"{score}_ci"


def half_width(deviation: pd.Series, freedom: pd.Series, runs_per_pair: pd.Series) -> pd.Series:
    quantile = t.ppf((1.0 + CONFIDENCE) / 2.0, freedom.to_numpy(dtype=np.float64))
    scaled = deviation.to_numpy(dtype=np.float64) * quantile
    return pd.Series(
        scaled / np.sqrt(runs_per_pair.to_numpy(dtype=np.float64)), index=deviation.index
    )


def with_intervals(rows: pd.DataFrame) -> pd.DataFrame:
    runs_per_pair = rows["n_runs"] / rows["n_pairs"]
    intervals = {
        interval_column(score): half_width(
            rows[spread_column(score)], rows["degrees_of_freedom"], runs_per_pair
        )
        for score in SPREAD_SCORES
    }
    return rows.assign(**intervals)


def pooled_deviation(deviations: pd.Series, freedom: pd.Series | None = None) -> float:
    squared = np.square(deviations.to_numpy(dtype=np.float64))
    if freedom is None:
        return float(np.sqrt(squared.mean()))
    weights = freedom.to_numpy(dtype=np.float64)
    return float(np.sqrt((weights * squared).sum() / weights.sum()))


def run_spread(groups: pd.DataFrame) -> pd.DataFrame:
    pooled = groups[groups["group"].eq(POPULATION) & groups["mode"].eq(SPREAD_MODE)]
    grouped = pooled.groupby(list(IDENTITY), sort=False)
    spread = grouped[list(SPREAD_SCORES)].std(ddof=1).add_suffix("_sd")
    spread.insert(0, "n_runs", grouped[REPLICATE].nunique())
    return spread.reset_index()


def replicate_spread(groups: pd.DataFrame) -> pd.DataFrame:
    per_question = run_spread(groups)
    rows: list[dict[str, Any]] = []
    for identity, block in per_question.groupby(list(CONDITION), sort=False):
        rows.append(
            {
                **dict(zip(CONDITION, identity, strict=True)),
                "question": EVERY,
                "question_label": EVERY,
                "n_runs": int(block["n_runs"].min()),
                **{
                    spread_column(score): pooled_deviation(block[spread_column(score)])
                    for score in SPREAD_SCORES
                },
            }
        )
    every = pd.DataFrame(rows, columns=list(per_question.columns))
    return pd.concat([per_question, every], ignore_index=True)


def pair_rows(table: pd.DataFrame) -> pd.DataFrame:
    return table[~table[list(IDENTITY)].isin([EVERY]).any(axis=1)]


def _pair_rows(pairs: pd.DataFrame) -> pd.DataFrame:
    counted = pairs.assign(n_pairs=1, degrees_of_freedom=pairs["n_runs"].sub(1))
    return counted[[*IDENTITY, *COUNTS, *map(spread_column, SPREAD_SCORES)]]


def _pooled_identity(keys: Sequence[str], found: Sequence[Any]) -> dict[str, Any]:
    identity: dict[str, Any] = dict.fromkeys(IDENTITY, EVERY)
    identity["model_key"] = None
    identity.update(zip(keys, found, strict=True))
    return identity


def _pooled_row(block: pd.DataFrame) -> dict[str, Any]:
    freedom = block["n_runs"].sub(1)
    return {
        "n_pairs": len(block),
        "n_runs": int(block["n_runs"].sum()),
        "degrees_of_freedom": int(freedom.sum()),
        **{
            spread_column(score): pooled_deviation(block[spread_column(score)], freedom)
            for score in SPREAD_SCORES
        },
    }


def _blocks(pairs: pd.DataFrame, keys: Sequence[str]) -> list[tuple[tuple[Any, ...], pd.DataFrame]]:
    if not keys:
        return [((), pairs)]
    return [
        (found if isinstance(found, tuple) else (found,), block)
        for found, block in pairs.groupby(list(keys), sort=False)
    ]


def replicate_table(groups: pd.DataFrame) -> pd.DataFrame:
    spread = run_spread(groups)
    pairs = _pair_rows(spread[spread["n_runs"].gt(1)].reset_index(drop=True))
    rows: list[dict[Any, Any]] = pairs.to_dict("records")
    for keys in POOLED_KEYS:
        for found, block in _blocks(pairs, keys):
            rows.append({**_pooled_identity(keys, found), **_pooled_row(block)})
    return with_intervals(pd.DataFrame(rows, columns=list(pairs.columns)))
