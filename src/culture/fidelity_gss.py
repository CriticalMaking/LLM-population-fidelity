from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from matplotlib import rc_context

from machine_bias_reproduction.config import FIGURES_ROOT, FIRST_REPLICATE, OUTPUTS_ROOT
from machine_bias_reproduction.questions import LETTERS

from .fidelity import (
    BASE_PREFIX,
    IDENTITY,
    MIN_CELLS_FOR_STRUCTURE,
    PAIRED_IDENTITY,
    PAIRED_METRICS,
    POPULATION,
    SCORE_COLUMNS,
    FloatArray,
    binding_term,
    delta_column,
    group_fidelity,
    lead_with,
    paired_fidelity,
)
from .fidelity_objectives import (
    OBJECTIVE_MODELS,
    STRUCTURE_TERM,
    level_row,
    shift_row,
    within_country,
)
from .fidelity_plates import Block, condition_blocks_plate
from .gss import (
    EXPECTED_SUBGROUPS,
    GSS_ARMS,
    GSS_MODE,
    RUN_KEYS,
    GssRecord,
    ordered_shares,
    run_table,
)
from .population import REPLICATE
from .population_plates import PLATE_TEXT, arm_plot_name
from .registry import BASE_ARM, CULTURE_MODELS, SUBPOP_ARM, arm_display, run_slug
from .tables import read_csv

GSS_MODELS: tuple[str, ...] = OBJECTIVE_MODELS

EVERY_SUBGROUP = "every subgroup"

EVERY_TOPIC = "all"

MASS = "valid_answer_mass"

WVS = "WVS"

GSS_SURVEY = "GSS"

SUBGROUP_CELLS = "SubPOP subgroups"

TOPICS: dict[str, tuple[str, str]] = {
    "Happiness": ("d_happy", "happy"),
    "Religious attendance": ("d_religiousp", "attend"),
    "Social trust": ("d_trust", "trust"),
}

WVS_HOMES: tuple[tuple[str, str], ...] = (
    ("german", "Germany"),
    ("spanish-mx", "Mexico"),
    (SUBPOP_ARM, "United States"),
)

GSS_FIGURES = FIGURES_ROOT / "fidelity" / "gss"

GSS_STEM = "fig_fidelity_components_gss_ntp"

GSS_GROUPS_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_gss_groups.csv"

GSS_PAIRED_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_gss_paired.csv"

GSS_SUMMARY_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_gss_summary.csv"

GSS_COMPARISON_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_gss_comparison.csv"

RunKey = tuple[str, str, str]

RunRows = dict[RunKey, FloatArray]


def question_records(records: Sequence[GssRecord]) -> dict[str, list[GssRecord]]:
    blocks: dict[str, list[GssRecord]] = {}
    for record in records:
        blocks.setdefault(record.qkey, []).append(record)
    return blocks


def survey_array(block: Sequence[GssRecord]) -> FloatArray:
    return np.vstack([ordered_shares(record.responses, record.ordinal) for record in block])


def run_rows(frame: pd.DataFrame) -> RunRows:
    letters = frame[list(LETTERS)].to_numpy(dtype=np.float64)
    keys = frame[list(RUN_KEYS)].astype(str).itertuples(index=False, name=None)
    return {cast(RunKey, key): row for key, row in zip(keys, letters, strict=True)}


def model_array(block: Sequence[GssRecord], rows: Mapping[RunKey, FloatArray]) -> FloatArray:
    shares = []
    for record in block:
        raw = rows.get((record.qkey, record.attribute, record.group))
        width = len(ordered_shares(record.responses, record.ordinal))
        shares.append(
            np.full(width, np.nan) if raw is None else ordered_shares(list(raw), record.ordinal)
        )
    return np.vstack(shares)


def answer_mass(block: Sequence[GssRecord], rows: Mapping[RunKey, FloatArray]) -> float:
    found = [rows.get((record.qkey, record.attribute, record.group)) for record in block]
    masses = [float(np.nansum(raw)) for raw in found if raw is not None]
    return float(np.mean(masses)) if masses else float("nan")


def gss_runs(root: Path | None = None) -> list[tuple[str, str]]:
    return [
        (key, arm) for key in GSS_MODELS for arm in GSS_ARMS if run_table(key, arm, root).is_file()
    ]


def load_runs(root: Path | None = None) -> dict[tuple[str, str], RunRows]:
    loaded: dict[tuple[str, str], RunRows] = {}
    for key, arm in gss_runs(root):
        frame = read_csv(run_table(key, arm, root))
        if frame is not None:
            loaded[(key, arm)] = run_rows(frame)
    return loaded


def gss_identity(model_key: str, arm: str, qkey: str) -> dict[str, Any]:
    model = CULTURE_MODELS[model_key]
    return {
        "question": qkey,
        "question_label": qkey,
        "model_key": model_key,
        "model_label": model.label,
        "arm": arm,
        "series": f"{model.label} ({arm_display(arm)})",
        "source": run_slug(model_key, arm),
        "mode": GSS_MODE,
        REPLICATE: FIRST_REPLICATE,
    }


def _retained(*arrays: FloatArray) -> np.ndarray:
    kept = np.ones(arrays[0].shape[0], dtype=bool)
    for array in arrays:
        kept &= np.isfinite(array).all(axis=1)
    return kept


def build_gss(
    records: Sequence[GssRecord],
    runs: Mapping[tuple[str, str], RunRows] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    loaded = load_runs() if runs is None else runs
    scope = {"group": POPULATION, "level": EVERY_SUBGROUP}
    rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for qkey, block in question_records(records).items():
        survey = survey_array(block)
        shares = {run: model_array(block, table) for run, table in loaded.items()}
        for (key, arm), llm in shares.items():
            kept = _retained(llm)
            if int(kept.sum()) < MIN_CELLS_FOR_STRUCTURE:
                continue
            rows.append(
                {
                    **gss_identity(key, arm, qkey),
                    **scope,
                    **group_fidelity(survey[kept], llm[kept]),
                    MASS: answer_mass(block, loaded[(key, arm)]),
                }
            )
        for key in GSS_MODELS:
            tuned, base = shares.get((key, SUBPOP_ARM)), shares.get((key, BASE_ARM))
            if tuned is None or base is None:
                continue
            kept = _retained(tuned, base)
            if int(kept.sum()) < MIN_CELLS_FOR_STRUCTURE:
                continue
            paired_rows.append(
                {
                    **gss_identity(key, SUBPOP_ARM, qkey),
                    "base_source": run_slug(key, BASE_ARM),
                    **scope,
                    **paired_fidelity(survey[kept], tuned[kept], base[kept]),
                }
            )
    groups = pd.DataFrame(rows)
    paired = pd.DataFrame(paired_rows)
    if not groups.empty:
        groups["binding_term"] = binding_term(groups)
        groups = lead_with(groups, [*IDENTITY, "group", "level"])
    if not paired.empty:
        paired["binding_term"] = binding_term(paired)
        paired[f"{BASE_PREFIX}binding_term"] = binding_term(paired, BASE_PREFIX)
        paired = lead_with(paired, [*PAIRED_IDENTITY, "group", "level"])
    return groups, paired


def check_gss(groups: pd.DataFrame, paired: pd.DataFrame) -> None:
    assert not groups.empty
    assert set(groups["arm"]) <= set(GSS_ARMS)
    assert groups["n_cells"].le(EXPECTED_SUBGROUPS).all()
    assert not groups.duplicated(["question", "model_key", "arm"]).any()
    scored = groups[list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)
    finite = scored[np.isfinite(scored)]
    assert np.all((finite >= 0.0) & (finite <= 1.0))
    for question, block in groups.groupby("question"):
        complete = block[block["n_cells"].eq(block["n_cells"].max())]
        spread = complete["d_wvs"].to_numpy(dtype=np.float64)
        assert np.allclose(spread, spread[0]), f"survey dispersion moved for {question}"
    if paired.empty:
        return
    assert set(paired["arm"]) == {SUBPOP_ARM}
    assert not paired.duplicated(["question", "model_key"]).any()
    for metric in PAIRED_METRICS:
        expected = paired[metric] - paired[f"{BASE_PREFIX}{metric}"]
        assert np.allclose(paired[delta_column(metric)], expected, equal_nan=True)


def _topic(frame: pd.DataFrame, topic: str, survey: str) -> pd.DataFrame:
    if topic == EVERY_TOPIC or frame.empty:
        return frame
    wvs_question, gss_question = TOPICS[topic]
    return frame[frame["question"].eq(wvs_question if survey == WVS else gss_question)]


def _condition(frame: pd.DataFrame, key: str, arm: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[frame["model_key"].eq(key) & frame["arm"].eq(arm)]


def gss_summary(groups: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for topic in (EVERY_TOPIC, *TOPICS):
        levels, shifts = (_topic(frame, topic, GSS_SURVEY) for frame in (groups, paired))
        for key in GSS_MODELS:
            for arm in GSS_ARMS:
                block = _condition(levels, key, arm)
                if block.empty:
                    continue
                rows.append(
                    {
                        "topic": topic,
                        "model_key": key,
                        "model_label": block["model_label"].iat[0],
                        "arm": arm,
                        "series": block["series"].iat[0],
                        **level_row(block),
                        f"{MASS}_mean": float(block[MASS].mean()),
                        **shift_row(_condition(shifts, key, arm)),
                    }
                )
    return pd.DataFrame(rows)


def change_row(paired: pd.DataFrame, groups: pd.DataFrame) -> dict[str, Any]:
    if paired.empty:
        return {}
    pfs = paired[delta_column("pfs")]
    center = paired[delta_column("score_center")]
    return {
        **shift_row(paired),
        f"{BASE_PREFIX}adaptability_ratio_median": float(
            paired[f"{BASE_PREFIX}adaptability_ratio"].median()
        ),
        "adaptability_ratio_median": float(paired["adaptability_ratio"].median()),
        "n_accuracy_up": int(paired[delta_column("score_accuracy")].gt(0.0).sum()),
        "n_structure_up": int(paired[delta_column("score_structure")].gt(0.0).sum()),
        "n_center_up_pfs_down": int((center.gt(0.0) & pfs.lt(0.0)).sum()),
        "n_structure_binding": int(groups["binding_term"].eq(STRUCTURE_TERM).sum()),
    }


def _wvs_rows(frame: pd.DataFrame, arm: str, country: str) -> pd.DataFrame:
    kept = frame[
        frame[REPLICATE].eq(FIRST_REPLICATE)
        & frame["mode"].eq(GSS_MODE)
        & frame["model_key"].isin(GSS_MODELS)
        & frame["arm"].eq(arm)
    ]
    return within_country(kept, country)


def survey_comparison(
    wvs_groups: pd.DataFrame,
    wvs_paired: pd.DataFrame,
    gss_groups: pd.DataFrame,
    gss_paired: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for topic in (EVERY_TOPIC, *TOPICS):
        for arm, country in WVS_HOMES:
            levels, shifts = (
                _topic(_wvs_rows(frame, arm, country), topic, WVS)
                for frame in (wvs_groups, wvs_paired)
            )
            change = change_row(shifts, levels)
            if change:
                rows.append({"survey": WVS, "cells": country, "arm": arm, "topic": topic, **change})
        tuned = gss_groups if gss_groups.empty else gss_groups[gss_groups["arm"].eq(SUBPOP_ARM)]
        change = change_row(_topic(gss_paired, topic, GSS_SURVEY), _topic(tuned, topic, GSS_SURVEY))
        if change:
            rows.append(
                {
                    "survey": GSS_SURVEY,
                    "cells": SUBGROUP_CELLS,
                    "arm": SUBPOP_ARM,
                    "topic": topic,
                    **change,
                }
            )
    return pd.DataFrame(rows)


def median_rows(groups: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_key", "model_label", "arm", "series"]
    return groups.groupby(keys, sort=False)[list(SCORE_COLUMNS)].median().reset_index()


def _released_first(rows: pd.DataFrame) -> pd.DataFrame:
    order = {arm: index for index, arm in enumerate(GSS_ARMS)}
    return rows.iloc[rows["arm"].map(order).argsort(kind="stable")]


def gss_blocks(groups: pd.DataFrame) -> list[Block]:
    count = groups["question"].nunique()
    blocks: list[Block] = [(f"All {count} GSS questions (median)", median_rows(groups))]
    for topic, (_, question) in TOPICS.items():
        rows = groups[groups["question"].eq(question)]
        if not rows.empty:
            blocks.append((f"GSS: {topic.lower()}", rows))
    return [(header, _released_first(rows)) for header, rows in blocks]


def gss_plate(groups: pd.DataFrame, destination: Path = GSS_FIGURES) -> list[Path]:
    if groups.empty:
        return []
    destination.mkdir(parents=True, exist_ok=True)
    with rc_context(cast(Any, PLATE_TEXT)):
        return condition_blocks_plate(gss_blocks(groups), arm_plot_name, destination / GSS_STEM)
