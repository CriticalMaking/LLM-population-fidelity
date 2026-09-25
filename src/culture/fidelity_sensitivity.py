"""Score population fidelity under alternative metric choices.

The Population Fidelity Score fixes six implementation choices: the median pairwise
distance behind the adaptability ratio, the symmetric ``min(A, 1/A)``, Spearman's rho for
structure, the clip of a nonpositive rho to zero, the geometric mean, and equal weight
for every cell. This module recomputes every score with one choice changed at a time,
and with three changed at once, on the same runs, cells, common cells and nEMD distances,
so the manuscript can show which conclusions depend on which choice.

Every alternative is pure arithmetic on one row of *readings* taken once per set of
cells: the per-cell error, both condensed pairwise-distance vectors summarised by median
and by mean, Spearman, Pearson and Kendall over those vectors, and the accuracy and center
terms with and without respondent weights. The default variant reproduces
``fidelity.group_fidelity`` to the last digit, which ``check_sensitivity`` asserts against
the frames the fidelity notebook has just built before anything is written.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from machine_bias_reproduction.config import OUTPUTS_ROOT
from machine_bias_reproduction.data import PreparedData, respondent_counts
from machine_bias_reproduction.metrics import nemd, pairwise_nemd
from machine_bias_reproduction.questions import Question, resolve_questions

from .fidelity import (
    BASE_PREFIX,
    IDENTITY,
    MIN_CELLS_FOR_DISPERSION,
    MIN_CELLS_FOR_STRUCTURE,
    MIN_CONDITIONS_FOR_CORRELATION,
    PAIRED_IDENTITY,
    PAIRED_METRICS,
    POPULATION,
    TERMS,
    FloatArray,
    answer_array,
    binding_term,
    cell_facets,
    closeness_score,
    delta_column,
    dispersion_score,
    geometric_mean,
    grouped_scores,
    lead_with,
    loaded_runs,
    paired_runs,
    structure_score,
)
from .matching import HOME
from .population import (
    MODES,
    RUN_KEYS,
    STRUCTURE_TABLE,
    RunSource,
    across_replicates,
    run_identity,
)
from .registry import BASE_ARM, is_base
from .tables import read_csv

Dispersion = Literal["median", "mean"]

Adaptability = Literal["symmetric", "one_sided"]

Structure = Literal["spearman", "pearson", "kendall"]

Aggregate = Literal["geometric", "arithmetic"]


@dataclass(frozen=True, slots=True)
class Variant:
    """One way of turning the readings of a cell set into a score."""

    name: str
    label: str
    dispersion: Dispersion = "median"
    adaptability: Adaptability = "symmetric"
    structure: Structure = "spearman"
    clipped: bool = True
    aggregate: Aggregate = "geometric"
    weighted: bool = False

    @property
    def signed(self) -> bool:
        return not self.clipped


DESIGN: tuple[str, ...] = (
    "dispersion",
    "adaptability",
    "structure",
    "clipped",
    "aggregate",
    "weighted",
)

DEFAULT = Variant("default", "Default")

VARIANTS: tuple[Variant, ...] = (
    DEFAULT,
    Variant("mean_dispersion", "Mean pairwise distance", dispersion="mean"),
    Variant("one_sided_adaptability", "One-sided min(A, 1)", adaptability="one_sided"),
    Variant("pearson_structure", "Pearson r", structure="pearson"),
    Variant("kendall_structure", "Kendall tau", structure="kendall"),
    Variant("unclipped_structure", "Unclipped rho, signed root", clipped=False),
    Variant("arithmetic_mean", "Arithmetic mean", aggregate="arithmetic"),
    Variant("weighted_cells", "Respondent-weighted E and C", weighted=True),
    Variant(
        "combined",
        "Mean, unclipped Pearson, weighted",
        dispersion="mean",
        structure="pearson",
        clipped=False,
        weighted=True,
    ),
)

VARIANT = "variant"

EPS = 1e-12

CULTURES: tuple[str, ...] = tuple(HOME)

COEFFICIENTS: dict[str, str] = {
    "spearman": "rho_spearman",
    "pearson": "r_pearson",
    "kendall": "tau_kendall",
}

COUNTS: tuple[str, ...] = ("n_cells", "n_pairs", "n_respondents")

READINGS: tuple[str, ...] = (
    *COUNTS,
    "model_flat",
    "e_mean_nemd",
    "e_mean_nemd_weighted",
    "d_wvs_median",
    "d_llm_median",
    "d_wvs_mean",
    "d_llm_mean",
    *COEFFICIENTS.values(),
    "c_center_nemd",
    "c_center_nemd_weighted",
)

BASE_READINGS: tuple[str, ...] = tuple(
    f"{BASE_PREFIX}{column}" for column in READINGS if column not in COUNTS
)

DIAGNOSTICS: tuple[str, ...] = ("adaptability_ratio", "structure_coefficient")

SCORES: tuple[str, ...] = (
    "score_accuracy",
    "score_dispersion",
    "score_structure",
    "score_center",
    "pfs",
)

# The readings the default variant must reproduce, named as the fidelity tables name them.
DEFAULT_COLUMNS: dict[str, str] = {
    "n_cells": "n_cells",
    "n_pairs": "n_pairs",
    "e_mean_nemd": "e_mean_nemd",
    "d_wvs_median": "d_wvs",
    "d_llm_median": "d_llm",
    "adaptability_ratio": "adaptability_ratio",
    "structure_coefficient": "rho_structure",
    "c_center_nemd": "c_center_nemd",
    **{score: score for score in SCORES},
}

POOLED_KEYS: tuple[str, ...] = ("question", "series", "mode")

DELTA_PFS = delta_column("pfs")

DELTA_CENTER = delta_column("score_center")

READINGS_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_sensitivity_readings.csv"

PAIRED_READINGS_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_sensitivity_paired.csv"

SENSITIVITY_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_sensitivity.csv"


def _coefficients(
    survey: FloatArray, model: FloatArray, count: int, flat: bool
) -> dict[str, float]:
    blank = {column: float("nan") for column in COEFFICIENTS.values()}
    if count < MIN_CELLS_FOR_STRUCTURE or flat or float(np.ptp(survey)) == 0.0:
        return blank
    return {
        "rho_spearman": float(spearmanr(survey, model).statistic),
        "r_pearson": float(pearsonr(survey, model).statistic),
        "tau_kendall": float(kendalltau(survey, model).statistic),
    }


def readings(wvs: FloatArray, llm: FloatArray, weights: FloatArray) -> dict[str, Any]:
    """Everything a variant can read off one set of cells, each pair vector taken once.

    The unweighted readings follow ``fidelity.group_fidelity`` line for line; the weighted
    accuracy and center use the cells' respondent counts, normalised within the set.
    """
    count = int(wvs.shape[0])
    error = np.atleast_1d(np.asarray(nemd(wvs, llm), dtype=np.float64))
    paired = count >= MIN_CELLS_FOR_DISPERSION
    survey = pairwise_nemd(wvs) if paired else np.empty(0, dtype=np.float64)
    model = pairwise_nemd(llm) if paired else np.empty(0, dtype=np.float64)
    flat = bool(model.size > 0 and float(np.ptp(model)) == 0.0)
    counts = np.asarray(weights, dtype=np.float64)
    share = counts / counts.sum()
    return {
        "n_cells": count,
        "n_pairs": int(survey.size),
        "n_respondents": float(counts.sum()),
        "model_flat": flat,
        "e_mean_nemd": float(error.mean()),
        "e_mean_nemd_weighted": float(error @ share),
        "d_wvs_median": float(np.median(survey)) if paired else float("nan"),
        "d_llm_median": float(np.median(model)) if paired else float("nan"),
        "d_wvs_mean": float(survey.mean()) if paired else float("nan"),
        "d_llm_mean": float(model.mean()) if paired else float("nan"),
        **_coefficients(survey, model, count, flat),
        "c_center_nemd": float(nemd(wvs.mean(axis=0), llm.mean(axis=0))),
        "c_center_nemd_weighted": float(nemd(share @ wvs, share @ llm)),
    }


def paired_readings(
    wvs: FloatArray, tuned: FloatArray, base: FloatArray, weights: FloatArray
) -> dict[str, Any]:
    """The tuned condition's readings beside its as-released base, on the same cells."""
    row = readings(wvs, tuned, weights)
    reference = readings(wvs, base, weights)
    row.update({f"{BASE_PREFIX}{k}": v for k, v in reference.items() if k not in COUNTS})
    return row


def run_weights(prepared: PreparedData, names: pd.Index) -> FloatArray:
    """The respondent count of every named cell, or unit weights for a run without a survey."""
    if prepared.wvs.empty:
        return np.ones(len(names), dtype=np.float64)
    values = respondent_counts(prepared).reindex(names).to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or bool((values <= 0.0).any()):
        raise ValueError("every scored cell needs a positive respondent count")
    return values


def build_sensitivity(
    questions: list[Question] | None = None,
    runs: Sequence[RunSource] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The readings of every run, mode, group and level, and of every paired comparison.

    Shaped like ``fidelity.build_fidelity``'s groups and paired tables: one row per run,
    mode, group and level, the paired rows on the common cells of both conditions and
    carrying the base condition's readings under ``base_``.
    """
    facets = cell_facets()
    rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for question in questions or resolve_questions(None):
        loaded = loaded_runs(question, runs)
        for run, prepared in loaded:
            names = prepared.names
            facet = facets.reindex(names)
            wvs = answer_array(prepared, prepared.wvs_props, names)
            weights = run_weights(prepared, names)
            for mode in prepared.modes():
                llm = answer_array(prepared, prepared.props(mode), names)
                identity = run_identity(run, question, mode)
                rows += grouped_scores(identity, facet, readings, wvs, llm, weights)
        for mode in MODES:
            for (run, tuned_run), (base_run, base_data), common in paired_runs(loaded, mode):
                identity = {**run_identity(run, question, mode), "base_source": base_run.source}
                paired_rows += grouped_scores(
                    identity,
                    facets.reindex(common),
                    paired_readings,
                    answer_array(tuned_run, tuned_run.wvs_props, common),
                    answer_array(tuned_run, tuned_run.props(mode), common),
                    answer_array(base_data, base_data.props(mode), common),
                    run_weights(tuned_run, common),
                )
    frame = pd.DataFrame(rows)
    paired = pd.DataFrame(paired_rows)
    if not frame.empty:
        frame = lead_with(frame, [*IDENTITY, "group", "level"])
    if not paired.empty:
        paired = lead_with(paired, [*PAIRED_IDENTITY, "group", "level"])
    return frame, paired


def one_sided_score(ratio: float) -> float:
    """Adaptability that forgives amplification: 1 whenever the model spreads at least as far."""
    if not np.isfinite(ratio):
        return float("nan")
    if ratio <= 0.0:
        return 0.0
    return float(min(ratio, 1.0))


def signed_geometric_mean(values: npt.ArrayLike) -> float:
    """The geometric mean whose last term keeps its sign: odd, and continuous through zero."""
    array = np.asarray(values, dtype=np.float64)
    if not array.size or bool(np.isnan(array).any()):
        return float("nan")
    *positive, signed = array.tolist()
    if bool((np.asarray(positive) <= 0.0).any()) or signed == 0.0:
        return 0.0
    return float(np.sign(signed) * geometric_mean([*positive, abs(signed)]))


def arithmetic_mean(values: npt.ArrayLike) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or bool(np.isnan(array).any()):
        return float("nan")
    return float(array.mean())


def variant_row(reading: Mapping[Any, Any], variant: Variant, prefix: str = "") -> dict[str, float]:
    """One variant's diagnostics and scores from one row of readings."""

    def read(column: str) -> Any:
        return reading[f"{prefix}{column}"]

    d_wvs = float(read(f"d_wvs_{variant.dispersion}"))
    d_llm = float(read(f"d_llm_{variant.dispersion}"))
    ratio = d_llm / d_wvs if d_wvs > 0.0 else float("nan")
    flat = bool(read("model_flat"))
    coefficient = float(read(COEFFICIENTS[variant.structure]))
    suffix = "_weighted" if variant.weighted else ""
    accuracy = closeness_score(float(read(f"e_mean_nemd{suffix}")))
    center = closeness_score(float(read(f"c_center_nemd{suffix}")))
    if variant.adaptability == "symmetric":
        dispersion = dispersion_score(ratio)
    else:
        dispersion = one_sided_score(ratio)
    if flat:
        structure = 0.0
    elif variant.clipped:
        structure = structure_score(coefficient)
    else:
        structure = coefficient
    parts = [accuracy, dispersion, structure]
    if variant.aggregate == "arithmetic":
        pfs = arithmetic_mean(parts)
    elif variant.clipped:
        pfs = geometric_mean(parts)
    else:
        pfs = signed_geometric_mean(parts)
    return {
        "adaptability_ratio": ratio,
        "structure_coefficient": coefficient,
        "score_accuracy": accuracy,
        "score_dispersion": dispersion,
        "score_structure": structure,
        "score_center": center,
        "pfs": pfs,
    }


def _lead(frame: pd.DataFrame, keys: Sequence[str]) -> list[str]:
    """The identity and every reading, so a scored row still shows what it was read from."""
    wanted = (*keys, *READINGS, *BASE_READINGS)
    return [column for column in wanted if column in frame.columns]


def score_variants(frame: pd.DataFrame, variants: Sequence[Variant] = VARIANTS) -> pd.DataFrame:
    """Every variant's scores on every row of readings, one block of rows per variant."""
    if frame.empty:
        return frame
    records = frame.to_dict("records")
    scored = {
        variant.name: pd.DataFrame([variant_row(record, variant) for record in records])
        for variant in variants
    }
    table = _stacked_for(frame, _lead(frame, (*IDENTITY, "group", "level")), scored, variants)
    table["binding_term"] = binding_term(table)
    return table


def _stacked_for(
    frame: pd.DataFrame,
    lead: Sequence[str],
    scored: dict[str, pd.DataFrame],
    variants: Sequence[Variant],
) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for variant in variants:
        block = pd.concat([frame[list(lead)].reset_index(drop=True), scored[variant.name]], axis=1)
        block.insert(len(lead), VARIANT, variant.name)
        block.insert(len(lead) + 1, "variant_label", variant.label)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def score_paired(paired: pd.DataFrame, variants: Sequence[Variant] = VARIANTS) -> pd.DataFrame:
    """Every variant on every paired comparison: the tuned scores, the base scores, the deltas."""
    if paired.empty:
        return paired
    records = paired.to_dict("records")
    scored: dict[str, pd.DataFrame] = {}
    for variant in variants:
        tuned = pd.DataFrame([variant_row(record, variant) for record in records])
        base = pd.DataFrame(
            [variant_row(record, variant, BASE_PREFIX) for record in records]
        ).add_prefix(BASE_PREFIX)
        deltas = pd.DataFrame(
            {
                delta_column(metric): tuned[metric] - base[f"{BASE_PREFIX}{metric}"]
                for metric in (*DIAGNOSTICS, *SCORES)
            }
        )
        scored[variant.name] = pd.concat([tuned, base, deltas], axis=1)
    table = _stacked_for(
        paired, _lead(paired, (*PAIRED_IDENTITY, "group", "level")), scored, variants
    )
    table["binding_term"] = binding_term(table)
    table[f"{BASE_PREFIX}binding_term"] = binding_term(table, BASE_PREFIX)
    return table


def _averaged(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Replicates of one condition averaged, then the binding term re-read off the means."""
    if frame.empty:
        return frame
    collapsed = across_replicates(frame, keys)
    collapsed["binding_term"] = binding_term(collapsed)
    if f"{BASE_PREFIX}score_accuracy" in collapsed.columns:
        collapsed[f"{BASE_PREFIX}binding_term"] = binding_term(collapsed, BASE_PREFIX)
    return collapsed


def _pooled(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["group"].eq(POPULATION)]


def _unique(index: pd.MultiIndex, what: str) -> pd.MultiIndex:
    """The pooled keys of one variant, refused when a condition appears more than once."""
    if not index.is_unique:
        raise ValueError(f"{what} rows repeat a question, series and mode: replicates not averaged")
    return index


def _agreement(x: FloatArray, y: FloatArray) -> float:
    """Spearman over the pairs where both are finite, blank below three or without variation."""
    kept = np.isfinite(x) & np.isfinite(y)
    if int(kept.sum()) < MIN_CONDITIONS_FOR_CORRELATION:
        return float("nan")
    if float(np.ptp(x[kept])) == 0.0 or float(np.ptp(y[kept])) == 0.0:
        return float("nan")
    return float(spearmanr(x[kept], y[kept]).statistic)


def _column(frame: pd.DataFrame, column: str) -> FloatArray:
    return frame[column].to_numpy(dtype=np.float64)


def _binding(frame: pd.DataFrame, prefix: str) -> dict[str, int]:
    counts = frame["binding_term"].value_counts()
    return {f"{prefix}_{term}": int(counts.get(term, 0)) for _, term in TERMS}


def _levels(pooled: pd.DataFrame) -> dict[str, Any]:
    pfs = _column(pooled, "pfs")
    return {
        "n_pooled": len(pooled),
        "accuracy_mean": float(np.nanmean(_column(pooled, "score_accuracy"))),
        "accuracy_median": float(np.nanmedian(_column(pooled, "score_accuracy"))),
        "adaptability_mean": float(np.nanmean(_column(pooled, "score_dispersion"))),
        "adaptability_median": float(np.nanmedian(_column(pooled, "score_dispersion"))),
        "structure_mean": float(np.nanmean(_column(pooled, "score_structure"))),
        "structure_median": float(np.nanmedian(_column(pooled, "score_structure"))),
        "coefficient_mean": float(np.nanmean(_column(pooled, "structure_coefficient"))),
        "coefficient_median": float(np.nanmedian(_column(pooled, "structure_coefficient"))),
        "pfs_mean": float(np.nanmean(pfs)),
        "pfs_median": float(np.nanmedian(pfs)),
        "pfs_min": float(np.nanmin(pfs)),
        "pfs_nonpositive": int(np.sum(pfs <= 0.0)),
    }


def _mode_agreement(pooled: pd.DataFrame) -> dict[str, float]:
    """Per question, PFS rank agreement between the modes over the series scored under both."""
    values: list[float] = []
    for _, block in pooled.groupby("question", sort=True):
        wide = block.pivot(index="series", columns="mode", values="pfs")
        if not set(MODES) <= set(wide.columns):
            continue
        both = wide.dropna(subset=list(MODES))
        values.append(_agreement(_column(both, MODES[0]), _column(both, MODES[1])))
    finite = [value for value in values if np.isfinite(value)]
    return {
        "mode_agreement_min": min(finite) if finite else float("nan"),
        "mode_agreement_max": max(finite) if finite else float("nan"),
    }


def _released(pooled: pd.DataFrame) -> dict[str, int]:
    """The sweep's as-released conditions, read for compression against amplification."""
    released = pooled[
        pooled["arm"].map(lambda arm: isinstance(arm, str) and is_base(arm))
        & pooled["model_key"].notna()
    ]
    ratio = _column(released, "adaptability_ratio")
    return {
        "released_n": len(released),
        "released_compressed": int(np.sum(ratio < 1.0)),
        "released_amplified": int(np.sum(ratio > 1.0)),
    }


def _groups(levels: pd.DataFrame) -> dict[str, int]:
    coefficient = _column(levels, "structure_coefficient")
    return {
        "groups_n": len(levels),
        **_binding(levels, "groups_binding"),
        "groups_negative_coefficient": int(np.sum(coefficient < 0.0)),
        "groups_coefficient_below_minus_010": int(np.sum(coefficient < -0.10)),
    }


def _sign(values: FloatArray) -> FloatArray:
    return np.sign(np.where(np.abs(values) <= EPS, 0.0, values))


def _directions(delta: FloatArray) -> tuple[int, int, int]:
    finite = np.isfinite(delta)
    improve = int(np.sum(finite & (delta > EPS)))
    decline = int(np.sum(finite & (delta < -EPS)))
    return improve, decline, int(finite.sum()) - improve - decline


def _own_cell_delta(scored: pd.DataFrame, culture: str) -> pd.DataFrame:
    """The same comparison taken on each condition's own cells rather than on their common set.

    Differencing two scores read over different cell sets lets a change in response
    coverage enter the comparison; the paired columns avoid it, and the gap between the
    two constructions is what the appendix reports.
    """
    keys = ["question", "model_key", "mode", "group", "level"]
    tuned = scored[scored["arm"].eq(culture)][[*keys, "pfs"]]
    base = scored[scored["arm"].eq(BASE_ARM)][[*keys, "pfs"]]
    merged = tuned.merge(base, on=keys, how="inner", suffixes=("_tuned", "_base"))
    return merged.assign(delta=merged["pfs_tuned"] - merged["pfs_base"])


def _finetuning(
    variant_paired: pd.DataFrame, default_paired: pd.DataFrame, scored: pd.DataFrame
) -> dict[str, Any]:
    """Each fine-tuning arm's paired changes, pooled and inside its home population."""
    out: dict[str, Any] = {}
    for culture in CULTURES:
        prefix = culture.replace("-", "_")
        rows = variant_paired[variant_paired["arm"].eq(culture)]
        reference = _pooled(default_paired[default_paired["arm"].eq(culture)])
        index = _unique(pd.MultiIndex.from_frame(reference[list(POOLED_KEYS)]), f"{culture} paired")
        pooled = _pooled(rows).set_index(list(POOLED_KEYS)).reindex(index)
        delta = _column(pooled, DELTA_PFS)
        improve, decline, unchanged = _directions(delta)
        out[f"{prefix}_n"] = len(pooled)
        out[f"{prefix}_delta_pfs_mean"] = float(np.nanmean(delta)) if len(delta) else float("nan")
        out[f"{prefix}_improve"] = improve
        out[f"{prefix}_decline"] = decline
        out[f"{prefix}_unchanged"] = unchanged
        agreement = _sign(delta) == _sign(_column(reference, DELTA_PFS))
        out[f"{prefix}_sign_agreement_vs_default"] = (
            float(agreement.mean()) if len(delta) else float("nan")
        )
        center = _column(pooled, DELTA_CENTER)
        out[f"{prefix}_delta_center_mean"] = (
            float(np.nanmean(center)) if len(center) else float("nan")
        )
        out[f"{prefix}_center_improve"] = _directions(center)[0]
        home = rows[rows["group"].eq("country") & rows["level"].eq(HOME[culture])]
        home_delta = _column(home, DELTA_PFS)
        home_center = _column(home, DELTA_CENTER)
        out[f"{prefix}_home_n"] = len(home)
        out[f"{prefix}_home_delta_pfs_mean"] = (
            float(np.nanmean(home_delta)) if len(home_delta) else float("nan")
        )
        out[f"{prefix}_home_improve"], out[f"{prefix}_home_decline"], _ = _directions(home_delta)
        out[f"{prefix}_home_delta_center_mean"] = (
            float(np.nanmean(home_center)) if len(home_center) else float("nan")
        )
        out[f"{prefix}_home_center_improve"] = _directions(home_center)[0]
        own = _own_cell_delta(scored, culture)
        own_pooled = _column(own[own["group"].eq(POPULATION)], "delta")
        own_home = _column(
            own[own["group"].eq("country") & own["level"].eq(HOME[culture])], "delta"
        )
        out[f"{prefix}_delta_pfs_mean_own_cells"] = (
            float(np.nanmean(own_pooled)) if own_pooled.size else float("nan")
        )
        out[f"{prefix}_home_delta_pfs_mean_own_cells"] = (
            float(np.nanmean(own_home)) if own_home.size else float("nan")
        )
    return out


def sensitivity_summary(
    frame: pd.DataFrame,
    paired: pd.DataFrame,
    variants: Sequence[Variant] = VARIANTS,
) -> pd.DataFrame:
    """One row per variant: the levels, counts and agreements the manuscript quotes.

    Replicates are averaged before anything is counted, and every comparison with the
    default variant aligns the pooled rows on question, series and mode.
    """
    scored = _averaged(score_variants(frame, variants), (*RUN_KEYS, VARIANT))
    scored_paired = _averaged(score_paired(paired, variants), (*RUN_KEYS, VARIANT))
    default = _pooled(scored[scored[VARIANT].eq(DEFAULT.name)])
    index = _unique(pd.MultiIndex.from_frame(default[list(POOLED_KEYS)]), "pooled")
    default_pfs = _column(default, "pfs")
    questions = sorted(default["question"].unique())
    rows: list[dict[str, Any]] = []
    for variant in variants:
        block = scored[scored[VARIANT].eq(variant.name)]
        pooled = _pooled(block).set_index(list(POOLED_KEYS)).reindex(index)
        pfs = _column(pooled, "pfs")
        row: dict[str, Any] = {
            VARIANT: variant.name,
            "variant_label": variant.label,
            **{field: getattr(variant, field) for field in DESIGN},
            **_levels(pooled),
            **_binding(pooled, "binding"),
            "binding_agreement_vs_default": float(
                np.mean(pooled["binding_term"].to_numpy() == default["binding_term"].to_numpy())
            ),
            "rank_vs_default": _agreement(pfs, default_pfs),
        }
        for question in questions:
            mask = index.get_level_values("question") == question
            row[f"rank_vs_default_{question}"] = _agreement(pfs[mask], default_pfs[mask])
        for column, term in TERMS:
            row[f"rank_vs_{term}"] = _agreement(pfs, _column(pooled, column))
        row.update(_mode_agreement(pooled.reset_index()))
        row.update(_released(pooled.reset_index()))
        row.update(_groups(block[~block["group"].eq(POPULATION)]))
        if not scored_paired.empty:
            row.update(
                _finetuning(
                    scored_paired[scored_paired[VARIANT].eq(variant.name)],
                    scored_paired[scored_paired[VARIANT].eq(DEFAULT.name)],
                    block,
                )
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _close(left: pd.Series, right: pd.Series, what: str) -> None:
    assert np.allclose(
        left.to_numpy(dtype=np.float64),
        right.to_numpy(dtype=np.float64),
        atol=1e-12,
        equal_nan=True,
    ), f"the default variant drifted from the fidelity table on {what}"


def _check_default(scored: pd.DataFrame, reference: pd.DataFrame, keys: Sequence[str]) -> None:
    default = scored[scored[VARIANT].eq(DEFAULT.name)]
    assert len(default) == len(reference), "the default variant scores a different row set"
    renamed = reference.rename(
        columns={column: f"{column}_ref" for column in reference.columns if column not in keys}
    )
    merged = default.merge(renamed, on=list(keys), how="inner", validate="one_to_one")
    assert len(merged) == len(reference), "the default variant scores a different row set"
    for ours, theirs in DEFAULT_COLUMNS.items():
        _close(merged[ours], merged[f"{theirs}_ref"], theirs)
    assert merged["model_flat"].eq(merged["model_flat_ref"]).all(), "default flat flags differ"
    assert merged["binding_term"].eq(merged["binding_term_ref"]).all(), "default binding term"
    if f"{BASE_PREFIX}score_accuracy" not in reference.columns:
        return
    for metric in PAIRED_METRICS:
        if metric == "pfs_with_center":
            continue
        ours = next(k for k, v in DEFAULT_COLUMNS.items() if v == metric)
        _close(
            merged[f"{BASE_PREFIX}{ours}"], merged[f"{BASE_PREFIX}{metric}_ref"], f"base {metric}"
        )
        if ours in (*DIAGNOSTICS, *SCORES):
            _close(
                merged[delta_column(ours)], merged[f"{delta_column(metric)}_ref"], f"delta {metric}"
            )
        else:
            expected = merged[ours] - merged[f"{BASE_PREFIX}{ours}"]
            _close(expected, merged[f"{delta_column(metric)}_ref"], f"delta {metric}")
    assert (
        merged[f"{BASE_PREFIX}binding_term"].eq(merged[f"{BASE_PREFIX}binding_term_ref"]).all()
    ), "default base binding term differs"


def _check_bounds(scored: pd.DataFrame, variants: Sequence[Variant]) -> None:
    sizes = scored.groupby(VARIANT).size()
    assert sizes.nunique() == 1, "the variants score different row sets"
    for variant in variants:
        block = scored[scored[VARIANT].eq(variant.name)]
        for column in SCORES:
            values = _column(block, column)
            finite = values[np.isfinite(values)]
            low = -1.0 if variant.signed and column in ("score_structure", "pfs") else 0.0
            assert np.all((finite >= low) & (finite <= 1.0)), (
                f"{variant.name}: {column} out of bounds"
            )


def _check_survey(frame: pd.DataFrame) -> None:
    for (question, mode), block in _pooled(frame).groupby(["question", "mode"]):
        complete = block[block["n_cells"].eq(block["n_cells"].max())]
        for column in ("d_wvs_median", "d_wvs_mean"):
            spread = _column(complete, column)
            assert np.allclose(spread, spread[0]), f"{column} moved for {question}/{mode}"
    assert frame["n_respondents"].ge(frame["n_cells"]).all(), "a cell with no respondents"


def _check_pearson(frame: pd.DataFrame) -> None:
    stored = read_csv(STRUCTURE_TABLE)
    if stored is None or "pearson_structure" not in stored.columns:
        return
    keys = ["question", "source", "mode"]
    merged = _pooled(frame).merge(stored[[*keys, "pearson_structure"]], on=keys, how="inner")
    kept = merged.dropna(subset=["r_pearson", "pearson_structure"])
    close = np.isclose(_column(kept, "r_pearson"), _column(kept, "pearson_structure"))
    named = ", ".join(
        "/".join(map(str, row)) for row in kept.loc[~close, keys].itertuples(index=False)
    )
    assert close.all(), f"pooled Pearson disagrees with population_structure.csv on {named}"


def check_sensitivity(
    frame: pd.DataFrame,
    paired: pd.DataFrame,
    groups: pd.DataFrame,
    paired_reference: pd.DataFrame,
    summary: pd.DataFrame,
    variants: Sequence[Variant] = VARIANTS,
) -> None:
    """Refuse readings whose default variant does not reproduce the fidelity tables.

    The pooled Pearson coefficients are also read against ``population_structure.csv``
    when that table is present; the check is skipped when it is not.
    """
    assert not frame.empty, "no readings to check"
    scored = score_variants(frame, variants)
    _check_default(scored, groups, (*IDENTITY, "group", "level"))
    _check_bounds(scored, variants)
    _check_survey(frame)
    _check_pearson(frame)
    if not paired.empty:
        scored_paired = score_paired(paired, variants)
        _check_default(scored_paired, paired_reference, (*PAIRED_IDENTITY, "group", "level"))
        _check_bounds(scored_paired, variants)
    assert len(summary) == len(variants), "one summary row per variant"
    default = summary[summary[VARIANT].eq(DEFAULT.name)].iloc[0]
    assert np.isclose(default["rank_vs_default"], 1.0), "the default does not rank like itself"
    assert default["binding_agreement_vs_default"] == 1.0, "the default binds unlike itself"
