from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from matplotlib import rc_context

from machine_bias_reproduction.config import FIGURES_ROOT, FIRST_REPLICATE, OUTPUTS_ROOT
from machine_bias_reproduction.figures import save_plate
from machine_bias_reproduction.questions import Question

from .fidelity import (
    IDENTITY,
    PAIRED_METRICS,
    POPULATION,
    SCORE_COLUMNS,
    build_fidelity,
    check_fidelity,
    delta_column,
)
from .fidelity_baseline import HOME, MANUSCRIPT_ARMS, build_baseline
from .fidelity_plates import (
    COMPONENT_ROW,
    ROW_LABEL_PAD,
    ROW_WIDTH,
    SPREAD,
    component_axis,
    component_handles,
    component_row,
    components_plate,
    row_figure,
    shift_plate,
    view_frame,
)
from .population import MODES, REPLICATE, RunSource, across_replicates, sources
from .population_plates import PLATE_TEXT, VIEWS, legend_below
from .registry import BASE_ARM, DISTRIBUTION_TRAINED_ARMS, arm_display

OBJECTIVE_MODELS: tuple[str, ...] = ("qwen3_vl_2b",)

DISTRIBUTION_ARMS: tuple[str, ...] = DISTRIBUTION_TRAINED_ARMS

OBJECTIVE_ARMS: tuple[str, ...] = (*MANUSCRIPT_ARMS, *DISTRIBUTION_ARMS)

SUBGROUPS = "subgroups"

COUNTRY_FAMILY = "country"

HOME_COUNTRIES: tuple[str, ...] = tuple(dict.fromkeys(HOME.values()))

SCOPES: tuple[str, ...] = (POPULATION, SUBGROUPS, *HOME_COUNTRIES)

BLOCK_SCOPES: tuple[str, ...] = (POPULATION, *HOME_COUNTRIES)

BLOCK_LABELS: dict[str, str] = {
    POPULATION: "All retained cells",
    "Germany": "German cells only",
    "Mexico": "Mexican cells only",
}

BLOCK_GAP = 1.4

BLOCK_RULE_INK = "#b8b8b8"

SPREAD_MODE = "fa"

EVERY_QUESTION = "all"

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

SPREAD_SCORES: tuple[str, ...] = (
    "score_accuracy",
    "score_dispersion",
    "score_structure",
    "pfs",
    "score_center",
)

RECONCILE_TOLERANCE = 1e-12

OBJECTIVES_FIGURES = FIGURES_ROOT / "fidelity" / "objectives"

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
    parts: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for replicate in sorted({run.replicate for run in selected}):
        drawn = [run for run in selected if run.replicate == replicate]
        built = build_fidelity(questions, drawn)
        check_fidelity(*built)
        parts.append(built)
    cells, groups, paired = (
        pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
        for frames in zip(*parts, strict=True)
    )
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


def level_column(score: str, statistic: str) -> str:
    return f"{score}_{statistic}"


def change_column(score: str) -> str:
    return f"{delta_column(score)}_{CHANGE_STATISTIC}"


def _level_row(block: pd.DataFrame) -> dict[str, Any]:
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


def _shift_row(block: pd.DataFrame) -> dict[str, Any]:
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
                        **_level_row(block),
                        **_null_row(_variant(nulls, key, arm)),
                        **_shift_row(_variant(shifts, key, arm)),
                    }
                )
    return pd.DataFrame(rows)


def pooled_deviation(deviations: pd.Series) -> float:
    squared = np.square(deviations.to_numpy(dtype=np.float64))
    return float(np.sqrt(squared.mean()))


def replicate_spread(groups: pd.DataFrame) -> pd.DataFrame:
    pooled = groups[groups["group"].eq(POPULATION) & groups["mode"].eq(SPREAD_MODE)]
    variant = ["model_key", "model_label", "arm", "series"]
    grouped = pooled.groupby([*variant, "question", "question_label"], sort=False)
    spread = grouped[list(SPREAD_SCORES)].std(ddof=1).add_suffix("_sd")
    spread.insert(0, "n_runs", grouped[REPLICATE].nunique())
    per_question = spread.reset_index()
    rows: list[dict[str, Any]] = []
    for identity, block in per_question.groupby(variant, sort=False):
        rows.append(
            {
                **dict(zip(variant, identity, strict=True)),
                "question": EVERY_QUESTION,
                "question_label": EVERY_QUESTION,
                "n_runs": int(block["n_runs"].min()),
                **{
                    f"{score}_sd": pooled_deviation(block[f"{score}_sd"]) for score in SPREAD_SCORES
                },
            }
        )
    every = pd.DataFrame(rows, columns=list(per_question.columns))
    return pd.concat([per_question, every], ignore_index=True)


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
    targeted = tuple(arm for arm, country in HOME.items() if country == scope)
    return (BASE_ARM, *targeted, *DISTRIBUTION_ARMS)


def block_rows(levels: pd.DataFrame, scope: str) -> pd.DataFrame:
    order = {arm: index for index, arm in enumerate(block_arms(scope))}
    scoped = _scoped(levels, scope)
    kept = scoped[scoped["arm"].isin(order)].dropna(subset=["pfs"])
    return kept.iloc[kept["arm"].map(order).argsort(kind="stable")]


def scoped_components_plate(
    levels: pd.DataFrame, mode: str, view: str, destination: Path
) -> list[Path]:
    blocks = [(scope, block_rows(levels, scope)) for scope in BLOCK_SCOPES]
    blocks = [(scope, rows) for scope, rows in blocks if not rows.empty]
    if not blocks:
        return []
    destination.mkdir(parents=True, exist_ok=True)
    slots = sum(len(rows) + 1 for _, rows in blocks) + BLOCK_GAP * (len(blocks) - 1)
    figure, axis = row_figure(int(np.ceil(slots)), ROW_WIDTH, COMPONENT_ROW)
    ticks: list[float] = []
    labels: list[str] = []
    headers: list[bool] = []
    position = 0.0
    for index, (scope, rows) in enumerate(blocks):
        if index:
            axis.axhline(position - BLOCK_GAP / 2.0, color=BLOCK_RULE_INK, linewidth=0.8)
            position -= BLOCK_GAP
        ticks.append(position)
        labels.append(BLOCK_LABELS[scope])
        headers.append(True)
        for row in rows.itertuples():
            position -= 1.0
            component_row(axis, row, position)
            ticks.append(position)
            labels.append(arm_display(str(row.arm)))
            headers.append(False)
    axis.set_yticks(ticks, labels)
    for label, header in zip(axis.get_yticklabels(), headers, strict=True):
        if header:
            label.set_fontweight("bold")
    axis.tick_params(axis="y", pad=ROW_LABEL_PAD)
    axis.set_ylim(position - 0.6, 0.6)
    component_axis(axis)
    drawn = pd.concat([rows for _, rows in blocks], ignore_index=True)
    legend_below(figure, component_handles(drawn), 2)
    return save_plate(figure, destination / f"fig_fidelity_components_scoped_{view}_{mode}")


def objective_plates(
    groups: pd.DataFrame, paired: pd.DataFrame, destination: Path = OBJECTIVES_FIGURES
) -> list[Path]:
    levels, shifts = (across_replicates(frame, spread=SPREAD) for frame in (groups, paired))
    produced: list[Path] = []
    with rc_context(cast(Any, PLATE_TEXT)):
        for mode in MODES:
            for view, question in VIEWS.items():
                if question is None:
                    continue
                produced += scoped_components_plate(
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
