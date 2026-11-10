from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from matplotlib import rc_context
from pandas.api.types import is_numeric_dtype
from scipy.stats import spearmanr

from machine_bias_reproduction.config import FIGURES_ROOT, FIRST_REPLICATE, OUTPUTS_ROOT
from machine_bias_reproduction.metrics import nemd, pairwise_nemd
from machine_bias_reproduction.questions import Question, resolve_questions

from .fidelity import (
    BASE_PREFIX,
    IDENTITY,
    MIN_CELLS_FOR_DISPERSION,
    MIN_CELLS_FOR_STRUCTURE,
    PAIRED_IDENTITY,
    PAIRED_METRICS,
    POPULATION,
    TERMS,
    FloatArray,
    answer_array,
    binding_term,
    cell_facets,
    delta_column,
    geometric_mean,
    group_fidelity,
    grouped_scores,
    lead_with,
    loaded_runs,
    overall_fidelity,
    overall_paired,
    paired_runs,
    structure_score,
)
from .fidelity_baseline import HOME_SCOPE, SCOPES, manuscript_runs, scope_rows
from .fidelity_plates import (
    SPREAD,
    components_plate,
    home_components_plate,
    home_shifts,
    shift_plate,
    view_frame,
)
from .fidelity_sensitivity import run_weights
from .matching import HOME
from .population import MODES, REPLICATE, RunSource, across_replicates, run_identity
from .population_plates import PLATE_TEXT, VIEWS

Structure = Literal["unsigned", "gated", "capped", "signed"]

ORIGINAL = "original"

EXACT_TIES = "exact"

STANDARD_ERROR_TIES = "se"

TIE_TOLERANCES: dict[str, float] = {EXACT_TIES: 0.0, "01": 0.01, "02": 0.02, "05": 0.05}

TIE_STANDARD_ERRORS = 1.96

TIE_RULES: tuple[str, ...] = (*TIE_TOLERANCES, STANDARD_ERROR_TIES)

FLOAT_TIE = 1e-12

MIN_PAIRS_FOR_SIGNED = 3

MIN_SCORES_FOR_RANK = 3

RECONCILE_TOLERANCE = 1e-12

EVERY_LEVEL = "levels"

SUMMARY_SCOPES: tuple[str, ...] = (*SCOPES, EVERY_LEVEL)


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    label: str
    structure: Structure = "unsigned"
    ties: str = EXACT_TIES


VARIANTS: tuple[Variant, ...] = (
    Variant(ORIGINAL, "Original"),
    Variant("gated", "Sign-gated", "gated"),
    Variant("capped", "Orientation-capped", "capped"),
    Variant("signed", "Signed distances", "signed"),
    Variant("signed_tie_01", "Signed distances, ties within 0.01", "signed", "01"),
    Variant("signed_tie_02", "Signed distances, ties within 0.02", "signed", "02"),
    Variant("signed_tie_05", "Signed distances, ties within 0.05", "signed", "05"),
    Variant(
        "signed_tie_se",
        "Signed distances, ties within 1.96 standard errors",
        "signed",
        STANDARD_ERROR_TIES,
    ),
)

ALTERNATIVES: tuple[Variant, ...] = tuple(
    variant for variant in VARIANTS if variant.name != ORIGINAL
)

PLATE_VARIANTS: tuple[str, ...] = ("capped", "signed")

ORIENTATION_FIGURES = FIGURES_ROOT / "fidelity" / "orientation"

ORIENTATION_GROUPS_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_orientation_groups.csv"

ORIENTATION_PAIRED_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_orientation_paired.csv"

ORIENTATION_SUMMARY_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_orientation_summary.csv"


def signed_column(ties: str) -> str:
    return "rho_signed" if ties == EXACT_TIES else f"rho_signed_tie_{ties}"


def pairs_column(ties: str) -> str:
    return "n_pairs_signed" if ties == EXACT_TIES else f"n_pairs_signed_tie_{ties}"


def structure_column(variant: Variant) -> str:
    return "score_structure" if variant.name == ORIGINAL else f"score_structure_{variant.name}"


def pfs_column(variant: Variant) -> str:
    return "pfs" if variant.name == ORIGINAL else f"pfs_{variant.name}"


def variant_named(variant: Variant | str) -> Variant:
    if isinstance(variant, Variant):
        return variant
    for known in VARIANTS:
        if known.name == variant:
            return known
    raise ValueError(f"unknown variant: {variant}")


ORIENTATION_READINGS: tuple[str, ...] = (
    "rho_orientation",
    *(signed_column(rule) for rule in TIE_RULES),
)

VARIANT_SCORES: tuple[str, ...] = tuple(
    column
    for variant in ALTERNATIVES
    for column in (structure_column(variant), pfs_column(variant))
)

ORIENTATION_METRICS: tuple[str, ...] = (*ORIENTATION_READINGS, *VARIANT_SCORES)

REVERSED = "orientation_reversed"

PAIR_COUNTS: tuple[str, ...] = tuple(pairs_column(rule) for rule in TIE_RULES)

ADDED_COLUMNS: tuple[str, ...] = (*ORIENTATION_METRICS, REVERSED, *PAIR_COUNTS)

VIEWED_SCORES: tuple[str, ...] = ("score_structure", "pfs", "pfs_with_center")

CENTERED_TERMS: tuple[str, ...] = (*(column for column, _ in TERMS), "score_center")


def anchor_scores(props: FloatArray) -> FloatArray:
    reference = np.zeros(props.shape[1], dtype=np.float64)
    reference[0] = 1.0
    return np.atleast_1d(np.asarray(nemd(props, reference), dtype=np.float64))


def standard_errors(props: FloatArray, counts: FloatArray) -> FloatArray:
    scale = np.linspace(0.0, 1.0, props.shape[1])
    variance = props @ scale**2 - (props @ scale) ** 2
    return np.asarray(np.sqrt(np.clip(variance, 0.0, None) / counts), dtype=np.float64)


def directions(gap: FloatArray) -> FloatArray:
    return np.asarray(np.where(np.abs(gap) > FLOAT_TIE, np.sign(gap), 0.0), dtype=np.float64)


def orientation_rho(survey_anchor: FloatArray, model_anchor: FloatArray) -> float:
    if survey_anchor.size < MIN_CELLS_FOR_STRUCTURE:
        return float("nan")
    if float(np.ptp(survey_anchor)) == 0.0 or float(np.ptp(model_anchor)) == 0.0:
        return float("nan")
    return float(spearmanr(survey_anchor, model_anchor).statistic)


def signed_rho(survey: FloatArray, model: FloatArray) -> float:
    if survey.size < MIN_PAIRS_FOR_SIGNED:
        return float("nan")
    if float(np.ptp(survey)) == 0.0 or float(np.ptp(model)) == 0.0:
        return float("nan")
    return float(spearmanr(survey, model).statistic)


def signed_readings(wvs: FloatArray, llm: FloatArray, counts: FloatArray) -> dict[str, Any]:
    count = int(wvs.shape[0])
    first, second = np.triu_indices(count, k=1)
    paired = count >= MIN_CELLS_FOR_DISPERSION
    survey = pairwise_nemd(wvs) if paired else np.empty(0, dtype=np.float64)
    distance = pairwise_nemd(llm) if paired else np.empty(0, dtype=np.float64)
    survey_anchor, model_anchor = anchor_scores(wvs), anchor_scores(llm)
    survey_gap = survey_anchor[second] - survey_anchor[first]
    model_gap = model_anchor[second] - model_anchor[first]
    model = directions(survey_gap) * directions(model_gap) * distance
    error = standard_errors(wvs, counts)
    limits: dict[str, Any] = {
        **TIE_TOLERANCES,
        STANDARD_ERROR_TIES: TIE_STANDARD_ERRORS * np.hypot(error[first], error[second]),
    }
    row: dict[str, Any] = {}
    for rule in TIE_RULES:
        kept = np.abs(survey_gap) > np.maximum(limits[rule], FLOAT_TIE)
        row[signed_column(rule)] = signed_rho(survey[kept], model[kept])
        row[pairs_column(rule)] = int(kept.sum())
    return row


def variant_structure(row: Mapping[str, Any], variant: Variant) -> float:
    original = float(row["score_structure"])
    if variant.structure == "unsigned" or not np.isfinite(original):
        return original
    if variant.structure == "signed":
        rho = float(row[signed_column(variant.ties)])
        if bool(row["model_flat"]) or not np.isfinite(rho):
            return 0.0
        return structure_score(rho)
    orientation = float(row["rho_orientation"])
    credit = structure_score(orientation) if np.isfinite(orientation) else 0.0
    if variant.structure == "gated":
        return original if credit > 0.0 else 0.0
    return min(original, credit)


def orientation_fidelity(wvs: FloatArray, llm: FloatArray, counts: FloatArray) -> dict[str, Any]:
    row = group_fidelity(wvs, llm)
    orientation = orientation_rho(anchor_scores(wvs), anchor_scores(llm))
    row["rho_orientation"] = orientation
    row[REVERSED] = bool(row["rho_structure"] > 0.0 and orientation < 0.0)
    row.update(signed_readings(wvs, llm, counts))
    for variant in ALTERNATIVES:
        structure = variant_structure(row, variant)
        row[structure_column(variant)] = structure
        row[pfs_column(variant)] = geometric_mean(
            [row["score_accuracy"], row["score_dispersion"], structure]
        )
    return row


def paired_orientation(
    wvs: FloatArray, tuned: FloatArray, base: FloatArray, counts: FloatArray
) -> dict[str, Any]:
    row = orientation_fidelity(wvs, tuned, counts)
    reference = orientation_fidelity(wvs, base, counts)
    for metric in (*PAIRED_METRICS, *ORIENTATION_METRICS):
        row[f"{BASE_PREFIX}{metric}"] = reference[metric]
        row[delta_column(metric)] = row[metric] - reference[metric]
    for flag in ("model_flat", REVERSED):
        row[f"{BASE_PREFIX}{flag}"] = reference[flag]
    return row


def build_orientation(
    questions: list[Question] | None = None,
    runs: Sequence[RunSource] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = manuscript_runs(every_replicate=True) if runs is None else list(runs)
    facets = cell_facets()
    rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for question in questions or resolve_questions(None):
        loaded = loaded_runs(question, selected)
        for run, prepared in loaded:
            names = prepared.names
            facet = facets.reindex(names)
            wvs = answer_array(prepared, prepared.wvs_props, names)
            counts = run_weights(prepared, names)
            for mode in prepared.modes():
                llm = answer_array(prepared, prepared.props(mode), names)
                identity = run_identity(run, question, mode)
                rows += grouped_scores(identity, facet, orientation_fidelity, wvs, llm, counts)
        for mode in MODES:
            for (run, tuned_run), (base_run, base_data), common in paired_runs(loaded, mode):
                identity = {**run_identity(run, question, mode), "base_source": base_run.source}
                paired_rows += grouped_scores(
                    identity,
                    facets.reindex(common),
                    paired_orientation,
                    answer_array(tuned_run, tuned_run.wvs_props, common),
                    answer_array(tuned_run, tuned_run.props(mode), common),
                    answer_array(base_data, base_data.props(mode), common),
                    run_weights(tuned_run, common),
                )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["binding_term"] = binding_term(frame)
        frame = lead_with(frame, [*IDENTITY, "group", "level"])
    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        paired["binding_term"] = binding_term(paired)
        paired[f"{BASE_PREFIX}binding_term"] = binding_term(paired, BASE_PREFIX)
        paired = lead_with(paired, [*PAIRED_IDENTITY, "group", "level"])
    return frame, paired


def _metric_of(column: str) -> str:
    if column.startswith("delta_") and column.endswith("_vs_base"):
        return column.removeprefix("delta_").removesuffix("_vs_base")
    return column.removeprefix(BASE_PREFIX)


def _geometric_rows(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    values = frame[list(columns)].to_numpy(dtype=np.float64)
    return pd.Series([geometric_mean(row) for row in values], index=frame.index)


def variant_view(frame: pd.DataFrame, variant: Variant | str) -> pd.DataFrame:
    if frame.empty:
        return frame
    chosen = variant_named(variant)
    added = [column for column in frame.columns if _metric_of(column) in ADDED_COLUMNS]
    view = frame.drop(columns=added)
    if chosen.name == ORIGINAL:
        return view
    prefixes = [
        prefix for prefix in ("", BASE_PREFIX) if f"{prefix}score_structure" in frame.columns
    ]
    for prefix in prefixes:
        view[f"{prefix}score_structure"] = frame[f"{prefix}{structure_column(chosen)}"]
        view[f"{prefix}pfs"] = frame[f"{prefix}{pfs_column(chosen)}"]
        view[f"{prefix}pfs_with_center"] = _geometric_rows(
            view, [f"{prefix}{term}" for term in CENTERED_TERMS]
        )
        view[f"{prefix}binding_term"] = binding_term(view, prefix)
    if BASE_PREFIX in prefixes:
        for metric in VIEWED_SCORES:
            view[delta_column(metric)] = view[metric] - view[f"{BASE_PREFIX}{metric}"]
    return view


def _first(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[frame[REPLICATE].eq(FIRST_REPLICATE)]


def _rank_agreement(original: FloatArray, scored: FloatArray) -> float:
    kept = np.isfinite(original) & np.isfinite(scored)
    if int(kept.sum()) < MIN_SCORES_FOR_RANK:
        return float("nan")
    if float(np.ptp(original[kept])) == 0.0 or float(np.ptp(scored[kept])) == 0.0:
        return float("nan")
    return float(spearmanr(original[kept], scored[kept]).statistic)


def _level_row(block: pd.DataFrame, variant: Variant) -> dict[str, Any]:
    original = block["pfs"].to_numpy(dtype=np.float64)
    scored = block[pfs_column(variant)].to_numpy(dtype=np.float64)
    structure = block[structure_column(variant)]
    changed = ~np.isclose(scored, original, equal_nan=True)
    terms = block[[column for column, _ in TERMS]].assign(score_structure=structure)
    return {
        "n_scores": len(block),
        "n_changed": int(changed.sum()),
        "n_raised": int((changed & (scored > original)).sum()),
        "n_lowered": int((changed & (scored < original)).sum()),
        "n_zero_pfs": int((scored == 0.0).sum()),
        "n_structure_binding": int(binding_term(terms).eq("structure").sum()),
        "n_reversed": int(block[REVERSED].sum()),
        "structure_mean": float(structure.mean()),
        "structure_median": float(structure.median()),
        "pfs_median": float(np.nanmedian(scored)),
        "pfs_mean": float(np.nanmean(scored)),
        "rank_vs_original": _rank_agreement(original, scored),
    }


def _shift_row(block: pd.DataFrame, variant: Variant, culture: str) -> dict[str, Any]:
    slug = culture.replace("-", "_")
    if block.empty:
        return {
            f"{slug}_n_paired": 0,
            f"{slug}_delta_pfs_mean": float("nan"),
            f"{slug}_n_pfs_up": 0,
            f"{slug}_n_pfs_down": 0,
            f"{slug}_sign_agreement": float("nan"),
        }
    moved = block[delta_column(pfs_column(variant))].to_numpy(dtype=np.float64)
    original = block[delta_column("pfs")].to_numpy(dtype=np.float64)
    kept = np.isfinite(moved) & np.isfinite(original)
    agreed = directions(moved[kept]) == directions(original[kept])
    return {
        f"{slug}_n_paired": len(block),
        f"{slug}_delta_pfs_mean": float(np.nanmean(moved)),
        f"{slug}_n_pfs_up": int((directions(moved[kept]) > 0.0).sum()),
        f"{slug}_n_pfs_down": int((directions(moved[kept]) < 0.0).sum()),
        f"{slug}_sign_agreement": float(agreed.mean()) if agreed.size else float("nan"),
    }


def summary_rows(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == EVERY_LEVEL:
        return frame[frame["group"].ne(POPULATION)]
    return scope_rows(frame, scope)


def orientation_summary(frame: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    levels, shifts = _first(frame), _first(paired)
    rows: list[dict[str, Any]] = []
    for scope in SUMMARY_SCOPES:
        block = summary_rows(levels, scope)
        if block.empty:
            continue
        moved = shifts if shifts.empty else summary_rows(shifts, scope)
        for variant in VARIANTS:
            row: dict[str, Any] = {
                "scope": scope,
                "variant": variant.name,
                "variant_label": variant.label,
                **_level_row(block, variant),
            }
            for culture in HOME:
                tuned = moved if moved.empty else moved[moved["arm"].eq(culture)]
                row.update(_shift_row(tuned, variant, culture))
            rows.append(row)
    return pd.DataFrame(rows)


def _reconcile(frame: pd.DataFrame, published: pd.DataFrame, keys: Sequence[str]) -> int:
    reference = published[published["source"].isin(set(frame["source"]))]
    columns = [column for column in reference.columns if column not in keys]
    missing = [column for column in columns if column not in frame.columns]
    assert not missing, f"the orientation table lost {missing}"
    merged = frame.merge(
        reference,
        on=list(keys),
        how="inner",
        suffixes=("", "_published"),
        validate="one_to_one",
    )
    assert len(merged) == len(frame) == len(reference), "the orientation table scores other rows"
    for column in columns:
        ours, theirs = merged[column], merged[f"{column}_published"]
        if is_numeric_dtype(ours) and is_numeric_dtype(theirs):
            assert np.allclose(
                ours.to_numpy(dtype=np.float64),
                theirs.to_numpy(dtype=np.float64),
                atol=RECONCILE_TOLERANCE,
                equal_nan=True,
            ), f"{column} drifted from the published table"
        else:
            assert ours.fillna("").eq(theirs.fillna("")).all(), (
                f"{column} drifted from the published table"
            )
    return len(merged)


def _ordered(frame: pd.DataFrame, lower: str, upper: str) -> bool:
    low = frame[lower].to_numpy(dtype=np.float64)
    high = frame[upper].to_numpy(dtype=np.float64)
    kept = np.isfinite(low) & np.isfinite(high)
    return bool(np.all(low[kept] <= high[kept] + RECONCILE_TOLERANCE))


def _check_readings(frame: pd.DataFrame) -> None:
    scored = frame[list(VARIANT_SCORES)].to_numpy(dtype=np.float64)
    finite = scored[np.isfinite(scored)]
    assert np.all((finite >= 0.0) & (finite <= 1.0)), "a variant score left the unit interval"
    readings = frame[list(ORIENTATION_READINGS)].to_numpy(dtype=np.float64)
    bounded = np.abs(readings[np.isfinite(readings)])
    assert np.all(bounded <= 1.0 + RECONCILE_TOLERANCE), "a correlation left its range"
    assert _ordered(frame, "pfs_capped", "pfs_gated"), "the cap scored above the gate"
    assert _ordered(frame, "pfs_gated", "pfs"), "the gate scored above the original"
    widening = [pairs_column(rule) for rule in TIE_TOLERANCES]
    for wider, narrower in zip(widening[1:], widening[:-1], strict=True):
        assert frame[wider].le(frame[narrower]).all(), "a wider tolerance kept more pairs"
    for column in PAIR_COUNTS:
        assert frame[column].le(frame["n_pairs"]).all(), "more signed pairs than pairs"


def check_orientation(
    frame: pd.DataFrame,
    paired: pd.DataFrame,
    groups: pd.DataFrame,
    paired_reference: pd.DataFrame,
) -> int:
    assert not frame.empty
    _check_readings(frame)
    matched = _reconcile(frame, groups, [*IDENTITY, "group", "level"])
    if paired.empty:
        return matched
    _check_readings(paired)
    for metric in ORIENTATION_METRICS:
        expected = paired[metric] - paired[f"{BASE_PREFIX}{metric}"]
        assert np.allclose(paired[delta_column(metric)], expected, equal_nan=True), (
            f"{metric} changes by something other than tuned minus base"
        )
    return matched + _reconcile(paired, paired_reference, [*PAIRED_IDENTITY, "group", "level"])


def _variant_plates(frame: pd.DataFrame, paired: pd.DataFrame, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    groups, overall, shifts, shifts_overall = (
        across_replicates(table, spread=SPREAD)
        for table in (frame, overall_fidelity(frame), paired, overall_paired(paired))
    )
    produced: list[Path] = []
    for mode in MODES:
        for view in VIEWS:
            pooled = view_frame(groups, overall, view, mode, POPULATION)
            if pooled.empty:
                continue
            produced += components_plate(pooled, mode, view, destination)
            produced += home_components_plate(groups, overall, view, mode, destination)
            moved = view_frame(shifts, shifts_overall, view, mode, POPULATION)
            if not moved.empty:
                produced += shift_plate(moved, mode, view, destination)
            homed = home_shifts(shifts, shifts_overall, view, mode)
            if not homed.empty:
                produced += shift_plate(homed, mode, view, destination, HOME_SCOPE)
    return produced


def orientation_plates(
    frame: pd.DataFrame,
    paired: pd.DataFrame,
    destination: Path = ORIENTATION_FIGURES,
    variants: Sequence[str] = PLATE_VARIANTS,
) -> list[Path]:
    produced: list[Path] = []
    with rc_context(cast(Any, PLATE_TEXT)):
        for name in variants:
            produced += _variant_plates(
                variant_view(frame, name), variant_view(paired, name), destination / name
            )
    return produced
