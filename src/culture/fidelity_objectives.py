from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from matplotlib import rc_context

from machine_bias_reproduction.config import FIGURES_ROOT, FIRST_REPLICATE, OUTPUTS_ROOT
from machine_bias_reproduction.questions import Question

from .fidelity import (
    COUNTRY_FAMILY,
    IDENTITY,
    PAIRED_METRICS,
    POPULATION,
    SCORE_COLUMNS,
    build_fidelity,
    check_fidelity,
    delta_column,
)
from .fidelity_baseline import MANUSCRIPT_ARMS, MANUSCRIPT_MODELS, build_baseline
from .fidelity_plates import (
    BLOCK_LABELS,
    SPREAD,
    Block,
    center_gap_plate,
    component_blocks_plate,
    components_plate,
    condition_blocks_plate,
    shift_plate,
    view_frame,
)
from .matching import HOME
from .population import MODES, REPLICATE, RunSource, across_replicates, sources
from .population_plates import PLATE_TEXT, VIEWS, arm_plot_name
from .registry import BASE_ARM, DISTRIBUTION_TRAINED_ARMS, SUBPOP_ARM, is_served

OBJECTIVE_MODELS: tuple[str, ...] = tuple(key for key in MANUSCRIPT_MODELS if not is_served(key))

COMMON_BACKBONE = "qwen3_vl_2b"

DISTRIBUTION_ARMS: tuple[str, ...] = DISTRIBUTION_TRAINED_ARMS

OBJECTIVE_ARMS: tuple[str, ...] = (*MANUSCRIPT_ARMS, *DISTRIBUTION_ARMS)

SUBGROUPS = "subgroups"

OBJECTIVE_HOMES: dict[str, str] = {**HOME, SUBPOP_ARM: "United States"}

HOME_COUNTRIES: tuple[str, ...] = tuple(dict.fromkeys(OBJECTIVE_HOMES.values()))

SCOPES: tuple[str, ...] = (POPULATION, SUBGROUPS, *HOME_COUNTRIES)

BLOCK_SCOPES: tuple[str, ...] = (POPULATION, *HOME_COUNTRIES)

STRUCTURE_TERM = "structure"

LEVEL_SCORES: tuple[str, ...] = (
    "score_accuracy",
    "adaptability_ratio",
    "score_dispersion",
    "score_structure",
    "pfs",
    "score_center",
)

LEVEL_STATISTICS: tuple[str, ...] = ("median", "min", "max")

CHANGE_STATISTIC = "mean"

RECONCILE_TOLERANCE = 1e-12

OBJECTIVES_FIGURES = FIGURES_ROOT / "fidelity" / "objectives"

OBJECTIVES_MODELS_FIGURES = OBJECTIVES_FIGURES / "models"

CENTER_GAP_STEM = "fig_fidelity_center_gap_questions"

OBJECTIVES_GROUPS_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_objectives_groups.csv"

OBJECTIVES_PAIRED_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_objectives_paired.csv"

OBJECTIVES_BASELINE_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_objectives_baseline.csv"

OBJECTIVES_SUMMARY_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_objectives_summary.csv"

OBJECTIVES_REPLICATES_TABLE = (
    OUTPUTS_ROOT / "culture" / "population_fidelity_objectives_replicates.csv"
)


def objective_runs(found: Sequence[RunSource] | None = None) -> list[RunSource]:
    return [
        run
        for run in (sources() if found is None else found)
        if run.key in OBJECTIVE_MODELS and run.arm in OBJECTIVE_ARMS
    ]


def first_runs(runs: Sequence[RunSource]) -> list[RunSource]:
    return [run for run in runs if run.replicate == FIRST_REPLICATE]


def build_objectives(
    questions: list[Question] | None = None,
    runs: Sequence[RunSource] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = objective_runs() if runs is None else list(runs)
    cells, groups, paired = build_fidelity(questions, selected)
    check_fidelity(cells, groups, paired)
    return cells, groups, paired


def objective_baseline(
    questions: list[Question] | None = None,
    runs: Sequence[RunSource] | None = None,
) -> pd.DataFrame:
    selected = objective_runs() if runs is None else list(runs)
    return build_baseline(questions, first_runs(selected))


def _first(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame[REPLICATE].eq(FIRST_REPLICATE)]


def within_country(frame: pd.DataFrame, country: str) -> pd.DataFrame:
    return frame[frame["group"].eq(COUNTRY_FAMILY) & frame["level"].eq(country)]


def _scoped(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    pooled = frame["group"].eq(POPULATION)
    if scope == POPULATION:
        return frame[pooled]
    if scope == SUBGROUPS:
        return frame[~pooled]
    return within_country(frame, scope)


def _variant(frame: pd.DataFrame, key: str, arm: str) -> pd.DataFrame:
    return frame[frame["model_key"].eq(key) & frame["arm"].eq(arm)]


def _backbone(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    return frame[frame["model_key"].eq(key)]


def level_column(score: str, statistic: str) -> str:
    return f"{score}_{statistic}"


def change_column(score: str) -> str:
    return f"{delta_column(score)}_{CHANGE_STATISTIC}"


def level_row(block: pd.DataFrame) -> dict[str, Any]:
    levels = block[list(LEVEL_SCORES)].agg(list(LEVEL_STATISTICS)).to_dict()
    return {
        "n_scores": len(block),
        **{
            level_column(score, statistic): float(levels[score][statistic])
            for score in LEVEL_SCORES
            for statistic in LEVEL_STATISTICS
        },
        "n_structure_binding": int(block["binding_term"].eq(STRUCTURE_TERM).sum()),
    }


def _null_row(block: pd.DataFrame) -> dict[str, Any]:
    above = block["above_null"].to_numpy(dtype=np.float64)
    return {
        "above_null_median": float(np.nanmedian(above)),
        "n_above_null": int(np.sum(above > 0.0)),
    }


def shift_row(block: pd.DataFrame) -> dict[str, Any]:
    if block.empty:
        return {}
    moved = block[delta_column("pfs")].dropna()
    return {
        "n_paired": len(block),
        **{
            change_column(score): float(block[delta_column(score)].mean()) for score in LEVEL_SCORES
        },
        "n_pfs_up": int(moved.gt(0.0).sum()),
        "n_pfs_down": int(moved.lt(0.0).sum()),
        "n_pfs_same": int(moved.eq(0.0).sum()),
        "n_adaptability_up": int(block[delta_column("score_dispersion")].gt(0.0).sum()),
        "n_center_up": int(block[delta_column("score_center")].gt(0.0).sum()),
    }


def objective_summary(
    groups: pd.DataFrame, paired: pd.DataFrame, baseline: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in SCOPES:
        levels, shifts, nulls = (
            _scoped(_first(frame), scope) for frame in (groups, paired, baseline)
        )
        for key in OBJECTIVE_MODELS:
            for arm in OBJECTIVE_ARMS:
                block = _variant(levels, key, arm)
                if block.empty:
                    continue
                rows.append(
                    {
                        "scope": scope,
                        "model_key": key,
                        "model_label": block["model_label"].iat[0],
                        "arm": arm,
                        "series": block["series"].iat[0],
                        **level_row(block),
                        **_null_row(_variant(nulls, key, arm)),
                        **shift_row(_variant(shifts, key, arm)),
                    }
                )
    return pd.DataFrame(rows)


def _published_rows(published: pd.DataFrame) -> pd.DataFrame:
    return published[
        published["model_key"].isin(OBJECTIVE_MODELS)
        & published["arm"].isin(MANUSCRIPT_ARMS)
        & published[REPLICATE].eq(FIRST_REPLICATE)
    ]


def _reconcile(frame: pd.DataFrame, published: pd.DataFrame, columns: Sequence[str]) -> int:
    keys = [*IDENTITY, "group", "level"]
    ours = _first(frame[frame["arm"].isin(MANUSCRIPT_ARMS)])
    theirs = _published_rows(published)
    merged = ours.merge(
        theirs[[*keys, "n_cells", *columns]],
        on=keys,
        how="inner",
        suffixes=("", "_published"),
        validate="one_to_one",
    )
    assert len(merged) == len(ours) == len(theirs), "the published variants were scored apart"
    assert merged["n_cells"].eq(merged["n_cells_published"]).all(), "the scored cells moved"
    for column in columns:
        assert np.allclose(
            merged[column].to_numpy(dtype=np.float64),
            merged[f"{column}_published"].to_numpy(dtype=np.float64),
            atol=RECONCILE_TOLERANCE,
            equal_nan=True,
        ), f"{column} drifted from the published table"
    return len(merged)


def check_objectives(
    groups: pd.DataFrame,
    paired: pd.DataFrame,
    published_groups: pd.DataFrame,
    published_paired: pd.DataFrame,
) -> int:
    scored = set(groups["arm"])
    assert set(MANUSCRIPT_ARMS) <= scored <= set(OBJECTIVE_ARMS), "an unexpected variant was scored"
    assert scored & set(DISTRIBUTION_ARMS), "a variant of the comparison is missing"
    assert set(paired["arm"]) == scored - {BASE_ARM}
    deltas = [delta_column(metric) for metric in PAIRED_METRICS]
    return _reconcile(groups, published_groups, SCORE_COLUMNS) + _reconcile(
        paired, published_paired, deltas
    )


def _plates(
    levels: pd.DataFrame, shifts: pd.DataFrame, mode: str, view: str, destination: Path
) -> list[Path]:
    if levels.empty:
        return []
    destination.mkdir(parents=True, exist_ok=True)
    produced = components_plate(levels, mode, view, destination)
    if not shifts.empty:
        produced += shift_plate(shifts, mode, view, destination)
    return produced


def block_arms(scope: str) -> tuple[str, ...]:
    if scope == POPULATION:
        return OBJECTIVE_ARMS
    return tuple(arm for arm in OBJECTIVE_ARMS if OBJECTIVE_HOMES.get(arm, scope) == scope)


def block_rows(levels: pd.DataFrame, scope: str) -> pd.DataFrame:
    order = {arm: index for index, arm in enumerate(block_arms(scope))}
    scoped = _scoped(levels, scope)
    kept = scoped[scoped["arm"].isin(order)].dropna(subset=["pfs"])
    return kept.iloc[kept["arm"].map(order).argsort(kind="stable")]


def scoped_blocks(levels: pd.DataFrame) -> list[Block]:
    scoped: list[Block] = [
        (BLOCK_LABELS[scope], block_rows(levels, scope)) for scope in BLOCK_SCOPES
    ]
    return [(header, rows) for header, rows in scoped if not rows.empty]


def scoped_stem(view: str, mode: str) -> str:
    return f"fig_fidelity_components_scoped_{view}_{mode}"


def scoped_components_plate(
    levels: pd.DataFrame, mode: str, view: str, destination: Path
) -> list[Path]:
    blocks = scoped_blocks(levels)
    if not blocks:
        return []
    destination.mkdir(parents=True, exist_ok=True)
    return component_blocks_plate(
        blocks, lambda row: arm_plot_name(str(row.arm)), destination / scoped_stem(view, mode)
    )


def scoped_models_plate(
    levels: pd.DataFrame, mode: str, view: str, destination: Path
) -> list[Path]:
    blocks = scoped_blocks(levels)
    if not blocks:
        return []
    destination.mkdir(parents=True, exist_ok=True)
    return condition_blocks_plate(blocks, arm_plot_name, destination / scoped_stem(view, mode))


def model_blocks(levels: pd.DataFrame) -> list[Block]:
    order = {arm: index for index, arm in enumerate(OBJECTIVE_ARMS)}
    pooled = _scoped(levels, POPULATION)
    kept = pooled[pooled["arm"].isin(order)].dropna(subset=["pfs", "score_center"])
    ranked = kept.iloc[kept["arm"].map(order).argsort(kind="stable")]
    blocks: list[Block] = []
    for key in OBJECTIVE_MODELS:
        rows = _backbone(ranked, key)
        if not rows.empty:
            blocks.append((str(rows["model_label"].iat[0]), rows))
    return blocks


def center_gap_models_plate(groups: pd.DataFrame, destination: Path) -> list[Path]:
    blocks = model_blocks(_first(groups))
    if not blocks:
        return []
    destination.mkdir(parents=True, exist_ok=True)
    return center_gap_plate(blocks, arm_plot_name, destination / CENTER_GAP_STEM)


ScopedPlate = Callable[[pd.DataFrame, str, str, Path], list[Path]]


def _plate_set(
    groups: pd.DataFrame, paired: pd.DataFrame, scoped: ScopedPlate, destination: Path
) -> list[Path]:
    if groups.empty:
        return []
    levels, shifts = (across_replicates(frame, spread=SPREAD) for frame in (groups, paired))
    produced: list[Path] = []
    for mode in MODES:
        for view, question in VIEWS.items():
            if question is None:
                continue
            produced += scoped(
                levels[levels["question"].eq(question) & levels["mode"].eq(mode)],
                mode,
                view,
                destination,
            )
            produced += _plates(
                view_frame(levels, levels, view, mode, POPULATION),
                view_frame(shifts, shifts, view, mode, POPULATION),
                mode,
                view,
                destination,
            )
    return produced


def objective_plates(
    groups: pd.DataFrame,
    paired: pd.DataFrame,
    destination: Path = OBJECTIVES_FIGURES,
    backbones: Path = OBJECTIVES_MODELS_FIGURES,
) -> list[Path]:
    with rc_context(cast(Any, PLATE_TEXT)):
        own = _plate_set(
            _backbone(groups, COMMON_BACKBONE),
            _backbone(paired, COMMON_BACKBONE),
            scoped_components_plate,
            destination,
        )
        every = _plate_set(groups, paired, scoped_models_plate, backbones)
        every += center_gap_models_plate(groups, backbones)
    return own + every
