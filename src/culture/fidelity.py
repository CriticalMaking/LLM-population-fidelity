from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import rankdata, spearmanr

from machine_bias_reproduction.config import (
    EXPECTED_SUBPOPULATIONS,
    GLOBAL_SEED,
    OUTPUTS_ROOT,
    paths_for,
)
from machine_bias_reproduction.data import PreparedData, load_subpops
from machine_bias_reproduction.metrics import nemd, pairwise_nemd
from machine_bias_reproduction.questions import Question, resolve_questions

from .matching import WVS_COUNTRIES
from .population import (
    MODES,
    REPLICATE,
    RunSource,
    across_replicates,
    load_run,
    run_identity,
    sources,
)
from .registry import BASE_ARM, is_base
from .tables import read_csv

FloatArray = npt.NDArray[np.float64]

POPULATION = "population"

EVERY_CELL = "every retained cell"

MISSING_TOKEN = "NA"

MIN_CELLS_FOR_DISPERSION = 2

MIN_CELLS_FOR_STRUCTURE = 3

MIN_CONDITIONS_FOR_CORRELATION = 3

AGE_BANDS: tuple[str, ...] = ("<25", "25-34", "35-44", "45-54", "55-64", "65-74", "75+")

WAVES: dict[str, str] = {
    "1995": "1995-1997",
    "1996": "1995-1997",
    "1997": "1995-1997",
    "2005": "2005-2006",
    "2006": "2005-2006",
    "2017": "2017-2018",
    "2018": "2017-2018",
}

FAMILIES: tuple[str, ...] = (
    "country",
    "wave",
    "sex",
    "age",
    "education",
    "employment",
    "marital_status",
)

FAMILY_LABELS: dict[str, str] = {
    POPULATION: "All retained subpopulations",
    "country": "Country",
    "wave": "Survey wave",
    "sex": "Sex",
    "age": "Age band",
    "education": "Education",
    "employment": "Employment",
    "marital_status": "Marital status",
}

LEVEL_ORDER: dict[str, tuple[str, ...]] = {
    "country": WVS_COUNTRIES,
    "wave": ("1995-1997", "2005-2006", "2017-2018"),
    "sex": ("Female", "Male"),
    "age": AGE_BANDS,
    "education": ("Low", "Middle", "High"),
    "employment": ("Working", "Student", "Homemaker", "Unemployed", "Retired"),
    "marital_status": ("Single", "Cohabiting", "Married", "Divorced or separated", "Widowed"),
}

CELL_PATTERN = re.compile(
    r"^(?P<country>.+?) (?P<year>\d{4}) "
    r"(?P<sex>Male|Female|NA) "
    r"(?P<age><25|25-34|35-44|45-54|55-64|65-74|75\+|NA) "
    r"(?P<education>High|Low|Middle|NA) "
    r"(?P<employment>Homemaker|Retired|Student|Unemployed|Working|NA) "
    r"(?P<marital_status>Cohabiting|Divorced or separated|Married|Single|Widowed|NA)$"
)

QUANTILES: tuple[tuple[str, float], ...] = (
    ("e_q10_nemd", 0.10),
    ("e_q25_nemd", 0.25),
    ("e_median_nemd", 0.50),
    ("e_q75_nemd", 0.75),
    ("e_q90_nemd", 0.90),
)

SCORE_COLUMNS: tuple[str, ...] = (
    "score_accuracy",
    "score_dispersion",
    "score_structure",
    "score_center",
    "pfs",
    "pfs_with_center",
)

GEOMETRIC_ACROSS_QUESTIONS: tuple[str, ...] = ("pfs", "pfs_with_center")

OVERALL_METRICS: tuple[str, ...] = (
    *SCORE_COLUMNS,
    "e_mean_nemd",
    *(name for name, _ in QUANTILES),
    "adaptability_ratio",
    "rho_structure",
    "c_center_nemd",
)

PAIRED_METRICS: tuple[str, ...] = (
    "e_mean_nemd",
    "d_llm",
    "adaptability_ratio",
    "rho_structure",
    "c_center_nemd",
    *SCORE_COLUMNS,
)

BASE_PREFIX = "base_"

TERMS: tuple[tuple[str, str], ...] = (
    ("score_accuracy", "accuracy"),
    ("score_dispersion", "adaptability"),
    ("score_structure", "structure"),
)

IDENTITY: tuple[str, ...] = (
    "question",
    "question_label",
    "model_key",
    "model_label",
    "arm",
    "series",
    "source",
    "mode",
    REPLICATE,
)

PAIRED_IDENTITY: tuple[str, ...] = (*IDENTITY, "base_source")

SERVED = "served"

EVERY_SERIES = "all"

CORRELATION_SCOPES: tuple[str, ...] = (SERVED, EVERY_SERIES)

CORRELATION_KEYS: tuple[str, ...] = ("question", "question_label", "mode", "group", "level")

CORRELATION_STATISTICS: tuple[str, ...] = ("spearman", "pearson")

INTERVAL_PARTS: tuple[str, ...] = ("", "_ci_low", "_ci_high")

CENTER_RESAMPLES = 10_000

CONFIDENCE = 0.95

CELLS_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_cells.csv"

GROUPS_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_groups.csv"

OVERALL_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_overall.csv"

PAIRED_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_paired.csv"

PAIRED_OVERALL_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_paired_overall.csv"

CENTER_CORRELATION_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_center_correlation.csv"

CENTER_SUMMARY_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_center_summary.csv"


def delta_column(metric: str) -> str:
    return f"delta_{metric}_vs_base"


def _parse(name: str) -> dict[str, str]:
    matched = CELL_PATTERN.fullmatch(name)
    if matched is None:
        raise ValueError(f"unparsable subpopulation label: {name}")
    return matched.groupdict()


def cell_facets(labels: Iterable[str] | None = None) -> pd.DataFrame:
    known = labels if labels is not None else load_subpops()["subpop"].unique()
    names = pd.Index(sorted(set(known)), name="subpopulation")
    frame = pd.DataFrame([_parse(str(name)) for name in names], index=names)
    frame["wave"] = frame.pop("year").map(WAVES)
    return frame[list(FAMILIES)].replace(MISSING_TOKEN, np.nan)


def levels(family: str, present: pd.Series) -> list[str]:
    seen = set(present.dropna().unique())
    return [level for level in LEVEL_ORDER[family] if level in seen]


def geometric_mean(values: npt.ArrayLike) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or bool(np.isnan(array).any()):
        return float("nan")
    if bool((array <= 0).any()):
        return 0.0
    return float(np.exp(np.log(array).mean()))


def _geometric_mean_of(values: pd.Series) -> float:
    return geometric_mean(pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64))


def closeness_score(error: float) -> float:
    return float(np.clip(1.0 - error, 0.0, 1.0))


def dispersion_score(ratio: float) -> float:
    if not np.isfinite(ratio):
        return float("nan")
    if ratio <= 0.0:
        return 0.0
    return float(min(ratio, 1.0 / ratio))


def structure_score(rho: float) -> float:
    return float(np.clip(rho, 0.0, 1.0))


def group_fidelity(wvs: FloatArray, llm: FloatArray) -> dict[str, Any]:
    count = int(wvs.shape[0])
    error = np.atleast_1d(np.asarray(nemd(wvs, llm), dtype=np.float64))
    paired = count >= MIN_CELLS_FOR_DISPERSION
    survey = pairwise_nemd(wvs) if paired else np.empty(0, dtype=np.float64)
    model = pairwise_nemd(llm) if paired else np.empty(0, dtype=np.float64)
    d_wvs = float(np.median(survey)) if paired else float("nan")
    d_llm = float(np.median(model)) if paired else float("nan")
    ratio = d_llm / d_wvs if d_wvs > 0.0 else float("nan")
    flat = bool(model.size > 0 and float(np.ptp(model)) == 0.0)
    if count >= MIN_CELLS_FOR_STRUCTURE and not flat:
        ranked = spearmanr(survey, model)
        rho = float(ranked.statistic)
        p_value = float(ranked.pvalue)
    else:
        rho = float("nan")
        p_value = float("nan")
    center = float(nemd(wvs.mean(axis=0), llm.mean(axis=0)))
    row: dict[str, Any] = {
        "n_cells": count,
        "n_pairs": int(survey.size),
        "e_mean_nemd": float(error.mean()),
        **{name: float(np.quantile(error, share)) for name, share in QUANTILES},
        "d_wvs": d_wvs,
        "d_llm": d_llm,
        "adaptability_ratio": ratio,
        "rho_structure": rho,
        "rho_structure_p_value": p_value,
        "model_flat": flat,
        "c_center_nemd": center,
    }
    row["score_accuracy"] = closeness_score(row["e_mean_nemd"])
    row["score_dispersion"] = dispersion_score(ratio)
    row["score_structure"] = 0.0 if flat else structure_score(rho)
    row["score_center"] = closeness_score(center)
    parts = [row["score_accuracy"], row["score_dispersion"], row["score_structure"]]
    row["pfs"] = geometric_mean(parts)
    row["pfs_with_center"] = geometric_mean([*parts, row["score_center"]])
    return row


def paired_fidelity(wvs: FloatArray, tuned: FloatArray, base: FloatArray) -> dict[str, Any]:
    row = group_fidelity(wvs, tuned)
    reference = group_fidelity(wvs, base)
    for metric in PAIRED_METRICS:
        row[f"{BASE_PREFIX}{metric}"] = reference[metric]
        row[delta_column(metric)] = row[metric] - reference[metric]
    row[f"{BASE_PREFIX}model_flat"] = reference["model_flat"]
    return row


def binding_term(frame: pd.DataFrame, prefix: str = "") -> pd.Series:
    names = {f"{prefix}{column}": term for column, term in TERMS}
    return frame[list(names)].idxmin(axis=1).map(names)


def lead_with(frame: pd.DataFrame, lead: Sequence[str]) -> pd.DataFrame:
    rest = [column for column in frame.columns if column not in lead]
    return frame[[*lead, *rest]]


def common_cells(runs: Iterable[PreparedData]) -> pd.Index:
    names: pd.Index | None = None
    for prepared in runs:
        names = prepared.names if names is None else names.intersection(prepared.names)
    if names is None:
        raise ValueError("no runs to share cells between")
    return names.sort_values()


Loaded = tuple[RunSource, PreparedData]

Pair = tuple[Loaded, Loaded, pd.Index]


def paired_runs(loaded: Sequence[Loaded], mode: str) -> Iterator[Pair]:
    scored = [(run, prepared) for run, prepared in loaded if mode in prepared.modes()]
    by_arm: dict[tuple[str, str], dict[int, Loaded]] = {}
    for run, prepared in scored:
        if run.key is not None and run.arm is not None:
            by_arm.setdefault((run.key, run.arm), {})[run.replicate] = (run, prepared)
    for (key, arm), tuned in by_arm.items():
        base = by_arm.get((key, BASE_ARM))
        if is_base(arm) or base is None:
            continue
        shared = sorted(set(tuned) & set(base))
        if not shared:
            continue
        common = common_cells([*(tuned[r][1] for r in shared), *(base[r][1] for r in shared)])
        for replicate in shared:
            yield tuned[replicate], base[replicate], common


def grouped_scores(
    identity: dict[str, Any],
    facet: pd.DataFrame,
    score: Callable[..., dict[str, Any]],
    *arrays: FloatArray,
) -> list[dict[str, Any]]:
    def scored(keep: npt.NDArray[np.bool_]) -> dict[str, Any]:
        return score(*(array[keep] for array in arrays))

    rows = [
        {
            **identity,
            "group": POPULATION,
            "level": EVERY_CELL,
            **scored(np.ones(len(facet), dtype=bool)),
        }
    ]
    for family in FAMILIES:
        values = facet[family]
        for level in levels(family, values):
            rows.append(
                {
                    **identity,
                    "group": family,
                    "level": level,
                    **scored(values.eq(level).to_numpy()),
                }
            )
    return rows


def loaded_runs(question: Question, runs: Sequence[RunSource] | None = None) -> list[Loaded]:
    found: list[Loaded] = []
    for run in runs if runs is not None else sources():
        prepared = load_run(run.source, run.key, run.arm, question)
        if prepared is not None:
            found.append((run, prepared))
    return found


def answer_array(prepared: PreparedData, props: pd.DataFrame, names: pd.Index) -> FloatArray:
    columns = list(prepared.question.answer_columns)
    return props.loc[names, columns].to_numpy(dtype=np.float64)


def build_fidelity(
    questions: list[Question] | None = None,
    runs: Sequence[RunSource] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    facets = cell_facets()
    cell_frames: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for question in questions or resolve_questions(None):
        loaded = loaded_runs(question, runs)
        for run, prepared in loaded:
            names = prepared.names
            facet = facets.reindex(names)
            wvs = answer_array(prepared, prepared.wvs_props, names)
            for mode in prepared.modes():
                llm = answer_array(prepared, prepared.props(mode), names)
                identity = run_identity(run, question, mode)
                error = np.atleast_1d(np.asarray(nemd(wvs, llm), dtype=np.float64))
                cell_frames.append(
                    facet.assign(
                        **identity,
                        subpopulation=names,
                        nEMD=error,
                        score_accuracy=np.clip(1.0 - error, 0.0, 1.0),
                    ).reset_index(drop=True)
                )
                rows += grouped_scores(identity, facet, group_fidelity, wvs, llm)
        for mode in MODES:
            for (run, tuned_run), (base_run, base_data), common in paired_runs(loaded, mode):
                identity = {**run_identity(run, question, mode), "base_source": base_run.source}
                paired_rows += grouped_scores(
                    identity,
                    facets.reindex(common),
                    paired_fidelity,
                    answer_array(tuned_run, tuned_run.wvs_props, common),
                    answer_array(tuned_run, tuned_run.props(mode), common),
                    answer_array(base_data, base_data.props(mode), common),
                )
    cells = lead_with(pd.concat(cell_frames, ignore_index=True), [*IDENTITY, "subpopulation"])
    groups = pd.DataFrame(rows)
    groups["binding_term"] = binding_term(groups)
    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        paired["binding_term"] = binding_term(paired)
        paired[f"{BASE_PREFIX}binding_term"] = binding_term(paired, BASE_PREFIX)
        paired = lead_with(paired, [*PAIRED_IDENTITY, "group", "level"])
    return cells, lead_with(groups, [*IDENTITY, "group", "level"]), paired


def _across_questions(
    frame: pd.DataFrame,
    keys: Sequence[str],
    metrics: Sequence[str],
    flags: Sequence[str],
) -> pd.DataFrame:
    aggregates: dict[str, tuple[str, Any]] = {
        "n_questions": ("question", "nunique"),
        "n_cells": ("n_cells", "min"),
    }
    for metric in metrics:
        geometric = metric.removeprefix(BASE_PREFIX) in GEOMETRIC_ACROSS_QUESTIONS
        aggregates[metric] = (metric, _geometric_mean_of if geometric else "mean")
    aggregates.update({flag: (flag, "any") for flag in flags})
    return frame.groupby(list(keys), dropna=False).agg(**aggregates).reset_index()


def overall_fidelity(groups: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "series",
        "source",
        "model_key",
        "model_label",
        "arm",
        "mode",
        REPLICATE,
        "group",
        "level",
    ]
    overall = _across_questions(groups, keys, OVERALL_METRICS, ("model_flat",))
    overall["binding_term"] = binding_term(overall)
    return overall


def overall_paired(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return paired
    keys = [
        "series",
        "source",
        "base_source",
        "model_key",
        "model_label",
        "arm",
        "mode",
        REPLICATE,
        "group",
        "level",
    ]
    metrics = [*PAIRED_METRICS, *(f"{BASE_PREFIX}{metric}" for metric in PAIRED_METRICS)]
    overall = _across_questions(paired, keys, metrics, ("model_flat", f"{BASE_PREFIX}model_flat"))
    for metric in PAIRED_METRICS:
        overall[delta_column(metric)] = overall[metric] - overall[f"{BASE_PREFIX}{metric}"]
    overall["binding_term"] = binding_term(overall)
    overall[f"{BASE_PREFIX}binding_term"] = binding_term(overall, BASE_PREFIX)
    return overall


def _varies(values: FloatArray, kept: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    highest = np.where(kept, values, -np.inf).max(axis=-1)
    return np.asarray(highest > np.where(kept, values, np.inf).min(axis=-1))


def _moment_correlation(
    x: FloatArray, y: FloatArray, kept: npt.NDArray[np.bool_], defined: npt.NDArray[np.bool_]
) -> FloatArray:
    count = np.maximum(kept.sum(axis=-1, keepdims=True), 1)
    dx = np.where(kept, x - np.where(kept, x, 0.0).sum(axis=-1, keepdims=True) / count, 0.0)
    dy = np.where(kept, y - np.where(kept, y, 0.0).sum(axis=-1, keepdims=True) / count, 0.0)
    numerator = (dx * dy).sum(axis=-1)
    denominator = np.sqrt((dx**2).sum(axis=-1) * (dy**2).sum(axis=-1))
    result = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=result, where=defined & (denominator > 0.0))
    return result


def correlations(x: FloatArray, y: FloatArray) -> dict[str, FloatArray]:
    """Spearman and Pearson along the last axis, over the pairs where both are finite.

    Blank where fewer than three pairs remain or either side does not vary.
    """
    kept = np.isfinite(x) & np.isfinite(y)
    defined = (
        (kept.sum(axis=-1) >= MIN_CONDITIONS_FOR_CORRELATION) & _varies(x, kept) & _varies(y, kept)
    )
    ranked = [rankdata(np.where(kept, values, np.inf), axis=-1) for values in (x, y)]
    return {
        "spearman": _moment_correlation(ranked[0], ranked[1], kept, defined),
        "pearson": _moment_correlation(x, y, kept, defined),
    }


def interval(draws: FloatArray) -> tuple[FloatArray, FloatArray]:
    """The percentile interval of the last axis, blank where no draw is defined."""
    tail = 50.0 * (1.0 - CONFIDENCE)
    low = np.full(draws.shape[:-1], np.nan)
    high = np.full(draws.shape[:-1], np.nan)
    finite = np.isfinite(draws).any(axis=-1)
    if finite.any():
        low[finite], high[finite] = np.nanpercentile(draws[finite], [tail, 100.0 - tail], axis=-1)
    return low, high


CenterBlock = tuple[dict[str, Any], pd.DataFrame, dict[str, FloatArray]]


def _center_blocks(groups: pd.DataFrame, resamples: int, seed: int) -> Iterator[CenterBlock]:
    conditions = across_replicates(groups)
    rng = np.random.default_rng(seed)
    blocks = ["question", "question_label", "mode"]
    for scope in CORRELATION_SCOPES:
        scoped = conditions[conditions["model_key"].notna()] if scope == SERVED else conditions
        for keys, block in scoped.groupby(blocks, dropna=False, sort=False):
            levels = block[["group", "level"]].drop_duplicates().reset_index(drop=True)
            series = pd.unique(block["series"])
            pfs, center = (
                block.pivot(index=["group", "level"], columns="series", values=column)
                .reindex(index=pd.MultiIndex.from_frame(levels), columns=series)
                .to_numpy(dtype=np.float64)
                for column in ("pfs", "score_center")
            )
            sample = rng.integers(0, len(series), size=(resamples, len(series)))
            draws = {
                statistic: np.empty((len(levels), resamples))
                for statistic in CORRELATION_STATISTICS
            }
            for row in range(len(levels)):
                for statistic, values in correlations(
                    pfs[row, sample], center[row, sample]
                ).items():
                    draws[statistic][row] = values
            table = levels.assign(n_conditions=(np.isfinite(pfs) & np.isfinite(center)).sum(axis=1))
            for statistic, values in correlations(pfs, center).items():
                low, high = interval(draws[statistic])
                table[statistic] = values
                table[f"{statistic}_ci_low"] = low
                table[f"{statistic}_ci_high"] = high
            yield {"scope": scope, **dict(zip(blocks, keys, strict=True))}, table, draws


def center_correlation(
    groups: pd.DataFrame, resamples: int = CENTER_RESAMPLES, seed: int = GLOBAL_SEED
) -> pd.DataFrame:
    """PFS against center alignment across the model conditions of each level.

    Replicates are averaged first, so a condition counts once. ``served`` keeps the
    sweep's conditions, as the subgroup boxes do; ``all`` adds the Mixtral references,
    as the component profile does. The interval resamples the conditions with
    replacement and takes the percentiles of the recomputed correlation.
    """
    columns = [
        "scope",
        *CORRELATION_KEYS,
        "n_conditions",
        *(f"{name}{part}" for name in CORRELATION_STATISTICS for part in INTERVAL_PARTS),
    ]
    tables = [
        table.assign(**identity) for identity, table, _ in _center_blocks(groups, resamples, seed)
    ]
    return pd.concat(tables, ignore_index=True)[columns]


def center_correlation_summary(
    groups: pd.DataFrame, resamples: int = CENTER_RESAMPLES, seed: int = GLOBAL_SEED
) -> pd.DataFrame:
    """The pooled correlation beside the mean and range of the per-level ones.

    Both intervals come from the same resamples as ``center_correlation``: the mean is
    recomputed over the levels inside every resample, so levels that share conditions
    are not treated as independent.
    """
    rows: list[dict[str, Any]] = []
    for identity, table, draws in _center_blocks(groups, resamples, seed):
        pooled = table["group"].eq(POPULATION).to_numpy()
        levels = table[~pooled]
        row: dict[str, Any] = {
            **identity,
            "n_conditions": int(table["n_conditions"].to_numpy()[pooled][0]),
            "n_levels": int(levels["spearman"].notna().sum()),
        }
        for statistic in CORRELATION_STATISTICS:
            level_draws = draws[statistic][~pooled]
            defined = np.isfinite(level_draws)
            totals = np.where(defined, level_draws, 0.0).sum(axis=0)
            means = np.full(totals.shape, np.nan)
            np.divide(totals, defined.sum(axis=0), out=means, where=defined.any(axis=0))
            low, high = interval(means[np.newaxis])
            for part in INTERVAL_PARTS:
                values = table[f"{statistic}{part}"].to_numpy(dtype=np.float64)
                row[f"{POPULATION}_{statistic}{part}"] = float(values[pooled][0])
            row.update(
                {
                    f"{statistic}_mean": float(levels[statistic].mean()),
                    f"{statistic}_mean_ci_low": float(low[0]),
                    f"{statistic}_mean_ci_high": float(high[0]),
                    f"{statistic}_min": float(levels[statistic].min()),
                    f"{statistic}_max": float(levels[statistic].max()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def cell_summary(cells: pd.DataFrame) -> pd.DataFrame:
    keys = ["question", "question_label", "mode", "subpopulation", *FAMILIES]
    summary = (
        cells.groupby(keys, dropna=False)
        .agg(
            n_runs=("source", "nunique"),
            mean_nEMD=("nEMD", "mean"),
            median_nEMD=("nEMD", "median"),
            min_nEMD=("nEMD", "min"),
            max_nEMD=("nEMD", "max"),
            mean_score_accuracy=("score_accuracy", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(["question", "mode", "mean_nEMD"], ascending=[True, True, False])


def _population(groups: pd.DataFrame) -> pd.DataFrame:
    return groups[groups["group"].eq(POPULATION)]


def cross_check_cells(cells: pd.DataFrame) -> int:
    reconciled = 0
    for (source, question, mode), run in cells.groupby(["source", "question", "mode"]):
        stored = read_csv(
            paths_for(str(source), str(question)).outputs / "subpopulation_distances.csv"
        )
        if stored is None:
            continue
        split = stored[stored["method"].eq(str(mode).upper())].set_index("subpopulation")["nEMD"]
        if split.empty:
            continue
        shared = run.set_index("subpopulation")["nEMD"].reindex(split.index).dropna()
        assert np.allclose(shared.to_numpy(), split.reindex(shared.index).to_numpy()), (
            f"recomputed cells disagree with subpopulation_distances.csv for {source}/{question}"
        )
        reconciled += 1
    return reconciled


def _check_paired(groups: pd.DataFrame, paired: pd.DataFrame) -> None:
    if paired.empty:
        return
    assert not paired["arm"].map(is_base).any()
    assert paired["n_cells"].gt(0).all()
    own = _population(groups).set_index(["question", "mode", "source"])["n_cells"]
    pooled = _population(paired)
    for column in ("source", "base_source"):
        shared = pooled.set_index(["question", "mode", column])["n_cells"]
        assert shared.le(own.reindex(shared.index)).all(), f"paired cells exceed the {column} run"
    per_pair = pooled.groupby(["question", "mode", "model_key", "arm"])["n_cells"].nunique()
    assert per_pair.eq(1).all(), "replicates of one pair were scored on different cells"
    for metric in PAIRED_METRICS:
        expected = paired[metric] - paired[f"{BASE_PREFIX}{metric}"]
        assert np.allclose(paired[delta_column(metric)], expected, equal_nan=True)


def check_fidelity(cells: pd.DataFrame, groups: pd.DataFrame, paired: pd.DataFrame) -> None:
    assert not cells.empty
    assert not groups.empty
    assert set(groups["group"]) == {POPULATION, *FAMILIES}
    assert groups["n_cells"].le(EXPECTED_SUBPOPULATIONS).all()
    pooled = _population(groups).set_index(["question", "mode", "source"])["n_cells"]
    for family in FAMILIES:
        family_rows = groups[groups["group"].eq(family)]
        totals = family_rows.groupby(["question", "mode", "source"])["n_cells"].sum()
        assert totals.le(pooled.reindex(totals.index)).all(), f"{family} covers unseen cells"
    scored = groups[list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)
    finite = scored[np.isfinite(scored)]
    assert np.all((finite >= 0.0) & (finite <= 1.0))
    population = _population(groups)
    for (question, mode), group in population.groupby(["question", "mode"]):
        complete = group[group["n_cells"].eq(group["n_cells"].max())]
        spread = complete["d_wvs"].to_numpy(dtype=np.float64)
        assert np.allclose(spread, spread[0]), f"survey dispersion moved for {question}/{mode}"
    assert cells.groupby(["question", "mode", "source"])["subpopulation"].nunique().gt(0).all()
    _check_paired(groups, paired)
