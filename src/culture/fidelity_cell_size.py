from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from machine_bias_reproduction.config import (
    FIGURES_ROOT,
    FIRST_REPLICATE,
    GLOBAL_SEED,
    OUTPUTS_ROOT,
)
from machine_bias_reproduction.figures import save_plate
from machine_bias_reproduction.plates import GRID, INK, MUTED_INK, SURFACE, tinted
from machine_bias_reproduction.questions import Question, resolve_questions

from .fidelity import (
    BASE_PREFIX,
    EVERY_CELL,
    IDENTITY,
    MIN_CELLS_FOR_STRUCTURE,
    ORIENTATION,
    PAIRED_IDENTITY,
    POPULATION,
    TERMS,
    FloatArray,
    answer_array,
    binding_term,
    delta_column,
    group_fidelity,
    lead_with,
    loaded_runs,
    paired_fidelity,
    paired_runs,
)
from .fidelity_baseline import manuscript_runs
from .fidelity_bootstrap import TAILS, Cells, case_seed, redrawn, survey_cells
from .fidelity_plates import COMPONENT_INKS, COMPONENTS
from .fidelity_sensitivity import run_weights
from .matching import HOME
from .population import MODES, REPLICATE, RunSource, run_identity
from .population_plates import LEGEND_HANDLERS, MARKER_SIZE, PLATE_TEXT

MINIMUM = "min_respondents"

EVERY_SIZE = 0

MINIMUMS: tuple[int, ...] = (EVERY_SIZE, 20, 30, 40, 50, 75, 100)

SURVEY_RESAMPLES = 500

EVERY_QUESTION = "all"

LEVELS: tuple[str, ...] = (
    "score_accuracy",
    "score_dispersion",
    "adaptability_ratio",
    "rho_structure",
    ORIENTATION,
    "score_structure",
    "pfs",
    "score_center",
)

EXACT = "exact"

SAMPLED = "sampled"

SIMULATORS: tuple[str, ...] = (EXACT, SAMPLED)

SURVEY_READINGS: tuple[str, ...] = (
    "score_accuracy",
    "adaptability_ratio",
    "score_dispersion",
    "rho_structure",
    ORIENTATION,
    "score_structure",
    "pfs",
)

RUN_KEYS: tuple[str, ...] = ("question", "source", "mode")

SIZE_KEYS: tuple[str, ...] = (*IDENTITY, "group", "level", MINIMUM)

PAIRED_SIZE_KEYS: tuple[str, ...] = (*PAIRED_IDENTITY, "group", "level", MINIMUM)

RECONCILE_TOLERANCE = 1e-12

CELL_SIZE_FIGURES = FIGURES_ROOT / "fidelity" / "cell_size"

STRUCTURE = "score_structure"

STRUCTURE_INK = COMPONENT_INKS[[column for column, _, _ in COMPONENTS].index(STRUCTURE)]

STRUCTURE_LABEL = next(label for column, _, label in COMPONENTS if column == STRUCTURE)

SIMULATOR_MARKS: dict[str, dict[str, Any]] = {
    EXACT: {"color": INK, "marker": "o", "linestyle": "solid"},
    SAMPLED: {"color": MUTED_INK, "marker": "s", "linestyle": (0, (4, 2))},
}

SIMULATOR_LABELS: dict[str, str] = {
    EXACT: "exact simulator (distributions as given)",
    SAMPLED: "sampled simulator (one answer per respondent)",
}

MODELS_LABEL = "evaluated combinations ({count} per question)"

EVERY_SIZE_TICK = "all"

SIZE_XLABEL = "Minimum valid survey answers in a cell (cells kept)"

PANEL_WIDTH = 3.4

PANEL_HEIGHT = 4.4

BOX_WIDTH = 0.56

CELL_SIZE_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_cell_size.csv"

CELL_SIZE_PAIRED_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_cell_size_paired.csv"

CELL_SIZE_SURVEY_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_cell_size_survey.csv"

CELL_SIZE_SUMMARY_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_cell_size_summary.csv"


def sized_rows(
    identity: dict[str, Any],
    counts: FloatArray,
    minimums: Sequence[int],
    score: Callable[..., dict[str, Any]],
    *arrays: FloatArray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for minimum in minimums:
        keep = counts >= minimum
        row: dict[str, Any] = {
            **identity,
            "group": POPULATION,
            "level": EVERY_CELL,
            MINIMUM: minimum,
            "n_cells": int(keep.sum()),
            "n_respondents": float(counts[keep].sum()),
        }
        if int(keep.sum()) >= MIN_CELLS_FOR_STRUCTURE:
            row.update(score(*(array[keep] for array in arrays)))
        rows.append(row)
    return rows


def _bound(frame: pd.DataFrame, prefix: str = "") -> pd.Series:
    scored = frame[f"{prefix}score_accuracy"].notna()
    return binding_term(frame[scored], prefix).reindex(frame.index)


def build_cell_size(
    questions: list[Question] | None = None,
    runs: Sequence[RunSource] | None = None,
    minimums: Sequence[int] = MINIMUMS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = manuscript_runs(every_replicate=True) if runs is None else list(runs)
    rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for question in questions or resolve_questions(None):
        loaded = loaded_runs(question, selected)
        for run, prepared in loaded:
            names = prepared.names
            wvs = answer_array(prepared, prepared.wvs_props, names)
            counts = run_weights(prepared, names)
            for mode in prepared.modes():
                llm = answer_array(prepared, prepared.props(mode), names)
                identity = run_identity(run, question, mode)
                rows += sized_rows(identity, counts, minimums, group_fidelity, wvs, llm)
        for mode in MODES:
            for (run, tuned_run), (base_run, base_data), common in paired_runs(loaded, mode):
                identity = {**run_identity(run, question, mode), "base_source": base_run.source}
                paired_rows += sized_rows(
                    identity,
                    run_weights(tuned_run, common),
                    minimums,
                    paired_fidelity,
                    answer_array(tuned_run, tuned_run.wvs_props, common),
                    answer_array(tuned_run, tuned_run.props(mode), common),
                    answer_array(base_data, base_data.props(mode), common),
                )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["binding_term"] = _bound(frame)
        frame = lead_with(frame, list(SIZE_KEYS))
    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        paired["binding_term"] = _bound(paired)
        paired[f"{BASE_PREFIX}binding_term"] = _bound(paired, BASE_PREFIX)
        paired = lead_with(paired, list(PAIRED_SIZE_KEYS))
    return frame, paired


def survey_reference(
    questions: list[Question] | None = None,
    minimums: Sequence[int] = MINIMUMS,
    resamples: int = SURVEY_RESAMPLES,
    seed: int = GLOBAL_SEED,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for question in questions or resolve_questions(None):
        shares, counts = survey_cells(question)
        survey = Cells(shares.to_numpy(dtype=np.float64), counts.to_numpy(dtype=np.float64))
        sizes = counts.to_numpy(dtype=np.float64)
        kept = {
            minimum: sizes >= minimum
            for minimum in minimums
            if int((sizes >= minimum).sum()) >= MIN_CELLS_FOR_STRUCTURE
        }
        drawn: dict[str, dict[int, list[list[float]]]] = {
            simulator: {minimum: [] for minimum in kept} for simulator in SIMULATORS
        }
        rng = np.random.default_rng(case_seed(seed, question.var))
        for _ in range(resamples):
            surveyed, answered = redrawn(rng, survey), redrawn(rng, survey)
            for minimum, keep in kept.items():
                scored = {
                    EXACT: group_fidelity(surveyed[keep], survey.shares[keep]),
                    SAMPLED: group_fidelity(surveyed[keep], answered[keep]),
                }
                for simulator, row in scored.items():
                    drawn[simulator][minimum].append([row[name] for name in SURVEY_READINGS])
        for minimum in minimums:
            keep = sizes >= minimum
            for simulator in SIMULATORS:
                row = {
                    "question": question.var,
                    "question_label": question.label,
                    MINIMUM: minimum,
                    "simulator": simulator,
                    "n_cells": int(keep.sum()),
                    "n_respondents": float(sizes[keep].sum()),
                    "respondents_min": float(sizes[keep].min()) if keep.any() else float("nan"),
                    "respondents_median": (
                        float(np.median(sizes[keep])) if keep.any() else float("nan")
                    ),
                    "respondents_max": float(sizes[keep].max()) if keep.any() else float("nan"),
                    "n_resamples": resamples if minimum in kept else 0,
                }
                if minimum in kept:
                    values = np.asarray(drawn[simulator][minimum], dtype=np.float64)
                    for index, name in enumerate(SURVEY_READINGS):
                        low, high = np.nanquantile(values[:, index], TAILS)
                        row[f"{name}_mean"] = float(np.nanmean(values[:, index]))
                        row[f"{name}_low"] = float(low)
                        row[f"{name}_high"] = float(high)
                rows.append(row)
    return pd.DataFrame(rows)


def _first(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[frame[REPLICATE].eq(FIRST_REPLICATE)]


def _level_row(block: pd.DataFrame, whole: pd.DataFrame) -> dict[str, Any]:
    scored = block[block["score_accuracy"].notna()]
    share = block["n_respondents"].to_numpy(dtype=np.float64) / whole["n_respondents"].to_numpy(
        dtype=np.float64
    )
    row: dict[str, Any] = {
        "n_combinations": len(block),
        "n_scored": len(scored),
        "n_cells_min": int(block["n_cells"].min()),
        "n_cells_max": int(block["n_cells"].max()),
        "share_respondents_mean": float(share.mean()),
    }
    for name in LEVELS:
        values = scored[name].to_numpy(dtype=np.float64)
        row[f"{name}_median"] = float(np.nanmedian(values)) if len(scored) else float("nan")
        row[f"{name}_mean"] = float(np.nanmean(values)) if len(scored) else float("nan")
    for _, term in TERMS:
        row[f"n_{term}_binding"] = int(scored["binding_term"].eq(term).sum())
    row["share_structure_binding"] = (
        row["n_structure_binding"] / len(scored) if len(scored) else float("nan")
    )
    row["n_zero_pfs"] = int(scored["pfs"].eq(0.0).sum())
    return row


def _survey_row(survey: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for simulator in SIMULATORS:
        measured = survey[survey["simulator"].eq(simulator)]
        for name in ("score_structure", "pfs"):
            column = f"{name}_mean"
            values = (
                measured[column].to_numpy(dtype=np.float64)
                if column in measured.columns
                else np.empty(0, dtype=np.float64)
            )
            defined = values[np.isfinite(values)]
            row[f"{simulator}_{name}_min"] = float(defined.min()) if defined.size else float("nan")
            row[f"{simulator}_{name}_max"] = float(defined.max()) if defined.size else float("nan")
    return row


def _shift_row(block: pd.DataFrame, culture: str) -> dict[str, Any]:
    slug = culture.replace("-", "_")
    scored = block[block["score_accuracy"].notna()] if not block.empty else block
    moved = (
        scored[delta_column("pfs")].to_numpy(dtype=np.float64)
        if not scored.empty
        else np.empty(0, dtype=np.float64)
    )
    centered = (
        scored[delta_column("score_center")].to_numpy(dtype=np.float64)
        if not scored.empty
        else np.empty(0, dtype=np.float64)
    )
    return {
        f"{slug}_n_paired": len(scored),
        f"{slug}_delta_pfs_mean": float(np.nanmean(moved)) if moved.size else float("nan"),
        f"{slug}_n_pfs_up": int((moved > 0.0).sum()),
        f"{slug}_n_pfs_down": int((moved < 0.0).sum()),
        f"{slug}_n_pfs_unchanged": int((moved == 0.0).sum()),
        f"{slug}_delta_center_mean": (
            float(np.nanmean(centered)) if centered.size else float("nan")
        ),
    }


def cell_size_summary(
    frame: pd.DataFrame, paired: pd.DataFrame, survey: pd.DataFrame
) -> pd.DataFrame:
    levels, shifts = _first(frame), _first(paired)
    keys = list(RUN_KEYS)
    whole = levels[levels[MINIMUM].eq(EVERY_SIZE)].set_index(keys)
    questions = list(pd.unique(levels["question"]))
    rows: list[dict[str, Any]] = []
    for minimum in sorted(pd.unique(levels[MINIMUM])):
        sized = levels[levels[MINIMUM].eq(minimum)]
        moved = shifts[shifts[MINIMUM].eq(minimum)] if not shifts.empty else shifts
        measured = survey[survey[MINIMUM].eq(minimum)]
        for question in (*questions, EVERY_QUESTION):
            every = question == EVERY_QUESTION
            block = sized if every else sized[sized["question"].eq(question)]
            row: dict[str, Any] = {
                MINIMUM: int(minimum),
                "question": question,
                **_level_row(block, whole.reindex(pd.MultiIndex.from_frame(block[keys]))),
                **_survey_row(measured if every else measured[measured["question"].eq(question)]),
            }
            for culture in HOME:
                tuned = moved if moved.empty else moved[moved["arm"].eq(culture)]
                if not every and not tuned.empty:
                    tuned = tuned[tuned["question"].eq(question)]
                row.update(_shift_row(tuned, culture))
            rows.append(row)
    return pd.DataFrame(rows)


def _reconcile(frame: pd.DataFrame, published: pd.DataFrame, keys: Sequence[str]) -> int:
    whole = frame[frame[MINIMUM].eq(EVERY_SIZE)].drop(columns=[MINIMUM, "n_respondents"])
    reference = published[
        published["group"].eq(POPULATION) & published["source"].isin(set(whole["source"]))
    ]
    columns = [column for column in reference.columns if column not in keys]
    missing = [column for column in columns if column not in whole.columns]
    assert not missing, f"the cell-size table lost {missing}"
    merged = whole.merge(
        reference, on=list(keys), how="inner", suffixes=("", "_published"), validate="one_to_one"
    )
    assert len(merged) == len(whole) == len(reference), "the cell-size table scores other rows"
    for column in columns:
        ours, theirs = merged[column], merged[f"{column}_published"]
        if pd.api.types.is_numeric_dtype(ours) and pd.api.types.is_numeric_dtype(theirs):
            assert np.allclose(
                ours.to_numpy(dtype=np.float64),
                theirs.to_numpy(dtype=np.float64),
                atol=RECONCILE_TOLERANCE,
                equal_nan=True,
            ), f"{column} drifted from the published table"
        else:
            same = ours.astype("string").fillna("").eq(theirs.astype("string").fillna(""))
            assert same.all(), f"{column} drifted from the published table"
    return len(merged)


def _check_sizes(frame: pd.DataFrame, keys: Sequence[str]) -> None:
    runs = [key for key in keys if key != MINIMUM]
    ordered = frame.sort_values([*runs, MINIMUM])
    steps = ordered.groupby(runs, dropna=False, sort=False)["n_cells"].diff()
    assert not steps.gt(0).any(), "a larger minimum kept more cells"
    unscored = frame["n_cells"].lt(MIN_CELLS_FOR_STRUCTURE)
    assert frame.loc[unscored, "score_accuracy"].isna().all(), "too few cells were scored"
    assert frame.loc[~unscored, "score_accuracy"].notna().all(), "a scorable set went unscored"


def check_cell_size(
    frame: pd.DataFrame,
    paired: pd.DataFrame,
    groups: pd.DataFrame,
    published_paired: pd.DataFrame,
) -> int:
    assert not frame.empty
    _check_sizes(frame, SIZE_KEYS)
    matched = _reconcile(frame, groups, [*IDENTITY, "group", "level"])
    if paired.empty:
        return matched
    _check_sizes(paired, PAIRED_SIZE_KEYS)
    return matched + _reconcile(paired, published_paired, [*PAIRED_IDENTITY, "group", "level"])


def size_tick(minimum: int) -> str:
    return EVERY_SIZE_TICK if minimum == EVERY_SIZE else str(minimum)


def plotted_minimums(frame: pd.DataFrame) -> list[int]:
    scored = frame.assign(scored=frame["score_accuracy"].notna())
    whole = scored.groupby(MINIMUM)["scored"].all()
    return [int(minimum) for minimum in whole.index[whole.to_numpy()]]


def cell_size_plate(
    frame: pd.DataFrame, survey: pd.DataFrame, destination: Path = CELL_SIZE_FIGURES
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    levels = _first(frame)
    minimums = plotted_minimums(levels)
    questions = list(pd.unique(levels["question"]))
    positions = np.arange(len(minimums), dtype=np.float64)
    with rc_context(cast(Any, PLATE_TEXT)):
        figure = Figure(
            figsize=(PANEL_WIDTH * len(questions), PANEL_HEIGHT + 1.3), layout="constrained"
        )
        axes = figure.subplots(1, len(questions), sharey=True, squeeze=False)[0]
        for axis, question in zip(axes, questions, strict=True):
            asked = levels[levels["question"].eq(question)]
            measured = survey[survey["question"].eq(question)]
            values = [
                asked.loc[asked[MINIMUM].eq(minimum), STRUCTURE].dropna().to_numpy(dtype=np.float64)
                for minimum in minimums
            ]
            axis.boxplot(
                values,
                positions=positions,
                widths=BOX_WIDTH,
                patch_artist=True,
                manage_ticks=False,
                boxprops={
                    "facecolor": tinted(STRUCTURE_INK),
                    "edgecolor": STRUCTURE_INK,
                    "linewidth": 1.0,
                },
                medianprops={"color": STRUCTURE_INK, "linewidth": 1.7},
                whiskerprops={"color": STRUCTURE_INK, "linewidth": 0.9},
                capprops={"color": STRUCTURE_INK, "linewidth": 0.9},
                flierprops={
                    "marker": "o",
                    "markersize": 3.4,
                    "markerfacecolor": SURFACE,
                    "markeredgecolor": STRUCTURE_INK,
                    "markeredgewidth": 0.8,
                },
                zorder=3,
            )
            for simulator, mark in SIMULATOR_MARKS.items():
                reference = measured[measured["simulator"].eq(simulator)].set_index(MINIMUM)
                reference = reference.reindex(minimums)
                center = reference[f"{STRUCTURE}_mean"].to_numpy(dtype=np.float64)
                low = reference[f"{STRUCTURE}_low"].to_numpy(dtype=np.float64)
                high = reference[f"{STRUCTURE}_high"].to_numpy(dtype=np.float64)
                axis.fill_between(
                    positions, low, high, color=mark["color"], alpha=0.14, linewidth=0
                )
                axis.plot(positions, center, markersize=MARKER_SIZE - 2.0, linewidth=1.6, **mark)
            kept = measured[measured["simulator"].eq(EXACT)].set_index(MINIMUM)["n_cells"]
            axis.set_xticks(
                positions, [f"{size_tick(minimum)}\n({int(kept[minimum])})" for minimum in minimums]
            )
            axis.set_title(str(asked["question_label"].iloc[0]), fontweight="bold")
            axis.set_ylim(-0.03, 1.03)
            axis.grid(axis="y", color=GRID, linewidth=0.6)
        axes[0].set_ylabel(STRUCTURE_LABEL)
        count = int(levels.groupby([MINIMUM, "question"]).size().max())
        handles: list[Any] = [
            Patch(
                facecolor=tinted(STRUCTURE_INK),
                edgecolor=STRUCTURE_INK,
                label=MODELS_LABEL.format(count=count),
            ),
            *(
                Line2D(
                    [],
                    [],
                    markersize=MARKER_SIZE - 2.0,
                    linewidth=1.6,
                    label=SIMULATOR_LABELS[simulator],
                    **mark,
                )
                for simulator, mark in SIMULATOR_MARKS.items()
            ),
        ]
        legend = figure.legend(
            handles=handles,
            loc="outside lower center",
            ncols=len(handles),
            handler_map=LEGEND_HANDLERS,
            title=SIZE_XLABEL,
        )
        legend.get_title().set_fontsize(PLATE_TEXT["axes.labelsize"])
        return save_plate(figure, destination / "fig_fidelity_cell_size_structure")
