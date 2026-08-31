from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from machine_bias_reproduction.config import FIGURES_ROOT, OUTPUTS_ROOT
from machine_bias_reproduction.data import PreparedData, canonical_run_paths, prepare_data
from machine_bias_reproduction.io_utils import write_csv
from machine_bias_reproduction.metrics import nemd
from machine_bias_reproduction.questions import Question, resolve_question

from .cross_culture_panels import culture_map, head_to_head_figure, league_figure, pairwise_heatmap
from .matching import countries_for, restrict_to
from .palette import MODES, arm_order
from .registry import ARMS, BASE_ARM, CultureModel, csv_stems
from .tables import read_csv

SURVEY = "WVS"
DEFAULT_PAIR: tuple[str, str] = ("spanish", "spanish-mx")
POOLED_LABEL = "All models"
MIN_CELLS_FOR_TEST = 10


def run_outputs(outputs_root: Path, model_key: str, arm: str, question: Question) -> Path:
    return outputs_root / "culture" / model_key / arm / question.var


def _write(frame: pd.DataFrame, destination: Path) -> None:
    if not frame.empty:
        write_csv(frame, destination)


def load_arm(
    model: CultureModel,
    arm: str,
    question: Question,
    outputs_root: Path = OUTPUTS_ROOT,
) -> PreparedData | None:
    ntp_path, fa_path = canonical_run_paths(
        run_outputs(outputs_root, model.key, arm, question),
        question,
        csv_stems(model.key, arm, question),
    )
    ntp = pd.read_csv(ntp_path) if ntp_path.is_file() else None
    fa = pd.read_csv(fa_path) if fa_path.is_file() else None
    if ntp is None and fa is None:
        return None
    return prepare_data(ntp, fa, question)


def load_arms(
    model: CultureModel,
    arms: Sequence[str],
    question: Question,
    outputs_root: Path = OUTPUTS_ROOT,
) -> dict[str, PreparedData]:
    loaded: dict[str, PreparedData] = {}
    for arm in arm_order(arms):
        prepared = load_arm(model, arm, question, outputs_root)
        if prepared is not None:
            loaded[arm] = prepared
    return loaded


def entity_order(names: Sequence[str]) -> list[str]:
    return [SURVEY] * (SURVEY in names) + arm_order(name for name in names if name != SURVEY)


def survey_props(loaded: Mapping[str, PreparedData], question: Question) -> pd.DataFrame:
    columns = list(question.answer_columns)
    stacked = pd.concat(
        [prepared.wvs_props.loc[prepared.names, columns] for prepared in loaded.values()]
    )
    return stacked[~stacked.index.duplicated()]


def arm_props(
    loaded: Mapping[str, PreparedData],
    question: Question,
    mode: str,
) -> dict[str, pd.DataFrame]:
    columns = list(question.answer_columns)
    frames: dict[str, pd.DataFrame] = {}
    for arm, prepared in loaded.items():
        if mode.lower() not in prepared.modes():
            continue
        frames[arm] = prepared.props(mode.lower()).loc[prepared.names, columns]
    return frames


def pairwise_matrix(props: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    entities = list(props)
    means = pd.DataFrame(np.nan, index=entities, columns=entities, dtype=np.float64)
    counts = pd.DataFrame(0, index=entities, columns=entities, dtype=int)
    for entity in entities:
        means.loc[entity, entity] = 0.0
        counts.loc[entity, entity] = len(props[entity].index)
    for first, second in combinations(entities, 2):
        shared = props[first].index.intersection(props[second].index)
        if shared.empty:
            continue
        value = float(
            np.mean(
                nemd(
                    props[first].loc[shared].to_numpy(np.float64),
                    props[second].loc[shared].to_numpy(np.float64),
                )
            )
        )
        means.loc[first, second] = value
        means.loc[second, first] = value
        counts.loc[first, second] = len(shared)
        counts.loc[second, first] = len(shared)
    return means, counts


def pairwise_rows(
    model_key: str,
    model_label: str,
    question: Question,
    mode: str,
    matrix: pd.DataFrame,
    counts: pd.DataFrame,
) -> pd.DataFrame:
    entities = list(matrix.index)
    rows: list[dict[str, Any]] = []
    for position, first in enumerate(entities):
        for second in entities[position + 1 :]:
            count = int(counts.loc[first, second])
            if not count:
                continue
            rows.append(
                {
                    "model_key": model_key,
                    "model_label": model_label,
                    "question": question.var,
                    "mode": mode,
                    "culture_a": first,
                    "culture_b": second,
                    "subpopulations": count,
                    "mean_nEMD": float(matrix.loc[first, second]),
                }
            )
    return pd.DataFrame(rows)


def pooled_matrix(matrices: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not matrices:
        return pd.DataFrame()
    names: set[str] = set()
    for matrix in matrices:
        names.update(matrix.index)
    entities = entity_order(sorted(names))
    stack = np.stack(
        [
            matrix.reindex(index=entities, columns=entities).to_numpy(np.float64)
            for matrix in matrices
        ]
    )
    observed = np.isfinite(stack)
    counts = observed.sum(axis=0)
    totals = np.where(observed, stack, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore"):
        values = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
    pooled = pd.DataFrame(values, index=entities, columns=entities)
    for entity in entities:
        pooled.loc[entity, entity] = 0.0
    return pooled


def complete_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    current = matrix
    while True:
        values = current.to_numpy(np.float64) if len(current) else np.empty((0, 0))
        missing = ~np.isfinite(values) & ~np.eye(len(current), dtype=bool)
        if not missing.any():
            return current
        gaps = missing.sum(axis=0) + missing.sum(axis=1)
        position = len(gaps) - 1 - int(np.argmax(gaps[::-1]))
        keep = [name for index, name in enumerate(current.index) if index != position]
        current = current.loc[keep, keep]


def restricted_mean(
    distances: pd.DataFrame | None,
    countries: tuple[str, ...],
    mode: str,
    cells: pd.Series | None = None,
) -> float:
    if distances is None:
        return float("nan")
    restricted = restrict_to(distances, countries)
    restricted = restricted[restricted["method"] == mode]
    if cells is not None:
        restricted = restricted[restricted["subpopulation"].isin(cells)]
    values = restricted["nEMD"]
    return float(values.mean()) if not values.empty else float("nan")


def arm_cells(
    distances: pd.DataFrame | None,
    countries: tuple[str, ...],
    mode: str,
) -> pd.Series | None:
    if distances is None:
        return None
    restricted = restrict_to(distances, countries)
    return restricted.loc[restricted["method"] == mode, "subpopulation"]


def league_table(
    models: Sequence[CultureModel],
    question: Question,
    outputs_root: Path = OUTPUTS_ROOT,
) -> pd.DataFrame:
    archived = read_csv(outputs_root / "archived" / question.var / "subpopulation_distances.csv")
    frames: list[pd.DataFrame] = []
    for model in models:
        matched = read_csv(
            outputs_root / "culture" / model.key / question.var / "culture_matched_comparison.csv"
        )
        if matched is None or matched.empty:
            continue
        base = read_csv(
            run_outputs(outputs_root, model.key, BASE_ARM, question) / "subpopulation_distances.csv"
        )
        arm_distances = {
            culture: read_csv(
                run_outputs(outputs_root, model.key, culture, question)
                / "subpopulation_distances.csv"
            )
            for culture in matched["culture"].unique()
        }
        frame = matched.copy()
        frame.insert(frame.columns.get_loc("model_label") + 1, "question", question.var)
        base_means: list[float] = []
        mixtral_means: list[float] = []
        for _, row in frame.iterrows():
            countries = countries_for(row["culture"])
            cells = arm_cells(arm_distances[row["culture"]], countries, row["mode"])
            base_means.append(restricted_mean(base, countries, row["mode"], cells))
            mixtral_means.append(restricted_mean(archived, countries, row["mode"], cells))
        frame["base_mean_nEMD"] = base_means
        frame["delta_vs_base"] = frame["mean_nEMD"] - frame["base_mean_nEMD"]
        frame["mixtral_mean_nEMD"] = mixtral_means
        frame["delta_vs_mixtral"] = frame["mean_nEMD"] - frame["mixtral_mean_nEMD"]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["rank"] = (
        combined.groupby(["mode", "model_key"])["mean_nEMD"].rank(method="min").astype(int)
    )
    return combined.sort_values(["mode", "mean_nEMD"]).reset_index(drop=True)


def shared_countries(pair: tuple[str, str]) -> tuple[str, ...]:
    unknown = [arm for arm in pair if arm not in ARMS]
    if unknown:
        raise ValueError(f"unknown cultures: {', '.join(unknown)}")
    return tuple(country for country in countries_for(pair[0]) if country in countries_for(pair[1]))


def head_to_head_cells(
    model: CultureModel,
    pair: tuple[str, str],
    question: Question,
    outputs_root: Path = OUTPUTS_ROOT,
) -> pd.DataFrame:
    shared = shared_countries(pair)
    if not shared:
        return pd.DataFrame()
    arm_a, arm_b = pair
    frame_a = read_csv(
        run_outputs(outputs_root, model.key, arm_a, question) / "subpopulation_distances.csv"
    )
    frame_b = read_csv(
        run_outputs(outputs_root, model.key, arm_b, question) / "subpopulation_distances.csv"
    )
    if frame_a is None or frame_b is None:
        return pd.DataFrame()
    frame_a = frame_a[frame_a["method"].isin(MODES)]
    frame_b = frame_b[frame_b["method"].isin(MODES)]
    kept = ["method", "subpopulation", "nEMD"]
    merged = restrict_to(frame_a, shared)[kept].merge(
        restrict_to(frame_b, shared)[kept],
        on=["method", "subpopulation"],
        suffixes=("_a", "_b"),
    )
    merged = merged.rename(columns={"method": "mode"})
    merged["delta_nEMD"] = merged["nEMD_b"] - merged["nEMD_a"]
    merged.insert(0, "model_key", model.key)
    merged.insert(1, "model_label", model.label)
    merged.insert(2, "question", question.var)
    merged.insert(3, "culture_a", arm_a)
    merged.insert(4, "culture_b", arm_b)
    merged.insert(5, "countries", "; ".join(shared))
    return merged


def head_to_head_summary(cells: pd.DataFrame) -> pd.DataFrame:
    if cells.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    grouped = cells.groupby(["model_key", "model_label", "mode"], sort=False)
    for (model_key, model_label, mode), group in grouped:
        deltas = group["delta_nEMD"].to_numpy(np.float64)
        testable = len(group) >= MIN_CELLS_FOR_TEST and bool(np.any(deltas != 0))
        rows.append(
            {
                "model_key": model_key,
                "model_label": model_label,
                "question": group["question"].iloc[0],
                "culture_a": group["culture_a"].iloc[0],
                "culture_b": group["culture_b"].iloc[0],
                "countries": group["countries"].iloc[0],
                "mode": mode,
                "subpopulations": len(group),
                "mean_nEMD_a": float(group["nEMD_a"].mean()),
                "mean_nEMD_b": float(group["nEMD_b"].mean()),
                "median_nEMD_a": float(group["nEMD_a"].median()),
                "median_nEMD_b": float(group["nEMD_b"].median()),
                "mean_delta_nEMD": float(group["delta_nEMD"].mean()),
                "median_delta_nEMD": float(group["delta_nEMD"].median()),
                "share_b_closer": float((group["delta_nEMD"] < 0).mean()),
                "wilcoxon_p": (
                    float(wilcoxon(group["nEMD_a"], group["nEMD_b"]).pvalue)
                    if testable
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["mode", "model_key"]).reset_index(drop=True)


def culture_cross(
    models: list[CultureModel],
    cultures: list[str],
    question: str | Question = "d_happy",
    *,
    pair: tuple[str, str] = DEFAULT_PAIR,
    outputs_root: Path = OUTPUTS_ROOT,
    tables_root: Path | None = None,
    figures_root: Path = FIGURES_ROOT,
) -> dict[str, Any]:
    outcome = resolve_question(question)
    shared = shared_countries(pair)
    destination = figures_root / "culture" / outcome.var
    tables_dir = (tables_root or outputs_root) / "culture" / outcome.var
    per_model: dict[str, list[str]] = {}
    figure_paths: list[Path] = []
    table_paths: list[Path] = []
    all_rows: list[pd.DataFrame] = []
    model_matrices: dict[str, dict[str, pd.DataFrame]] = {}

    for model in models:
        loaded = load_arms(model, cultures, outcome, outputs_root)
        per_model[model.key] = sorted(loaded)
        if not loaded:
            continue
        matrices: dict[str, pd.DataFrame] = {}
        for mode in MODES:
            frames = arm_props(loaded, outcome, mode)
            if len(frames) < 2:
                matrices[mode] = pd.DataFrame()
                continue
            props = {SURVEY: survey_props(loaded, outcome), **frames}
            means, counts = pairwise_matrix(props)
            rows = pairwise_rows(model.key, model.label, outcome, mode, means, counts)
            if not rows.empty:
                all_rows.append(rows)
            matrices[mode] = means
        if all(matrix.empty for matrix in matrices.values()):
            continue
        model_matrices[model.key] = matrices
        destination.mkdir(parents=True, exist_ok=True)
        figure_paths.extend(
            pairwise_heatmap(
                matrices, model.label, destination, f"fig_culture_pairwise_nemd_{model.key}"
            )
        )
        figure_paths.extend(
            culture_map(
                {mode: complete_matrix(matrix) for mode, matrix in matrices.items()},
                model.label,
                destination,
                f"fig_culture_map_{model.key}",
            )
        )

    if len(model_matrices) > 1:
        pooled = {
            mode: pooled_matrix(
                [matrices[mode] for matrices in model_matrices.values() if not matrices[mode].empty]
            )
            for mode in MODES
        }
        figure_paths.extend(
            pairwise_heatmap(pooled, POOLED_LABEL, destination, "fig_culture_pairwise_nemd_pooled")
        )
        figure_paths.extend(
            culture_map(
                {mode: complete_matrix(matrix) for mode, matrix in pooled.items()},
                POOLED_LABEL,
                destination,
                "fig_culture_map_pooled",
            )
        )

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        _write(combined, tables_dir / "culture_pairwise_nemd.csv")
        table_paths.append(tables_dir / "culture_pairwise_nemd.csv")

    league = league_table(models, outcome, outputs_root)
    _write(league, tables_dir / "culture_league.csv")
    if not league.empty:
        table_paths.append(tables_dir / "culture_league.csv")
        destination.mkdir(parents=True, exist_ok=True)
        figure_paths.extend(league_figure(league, destination))

    collected: list[pd.DataFrame] = []
    for model in models:
        cells = head_to_head_cells(model, pair, outcome, outputs_root)
        if not cells.empty:
            collected.append(cells)
    if collected:
        cells = pd.concat(collected, ignore_index=True)
        _write(cells, tables_dir / "culture_head_to_head_cells.csv")
        table_paths.append(tables_dir / "culture_head_to_head_cells.csv")
        summary = head_to_head_summary(cells)
        _write(summary, tables_dir / "culture_head_to_head.csv")
        if not summary.empty:
            table_paths.append(tables_dir / "culture_head_to_head.csv")
        destination.mkdir(parents=True, exist_ok=True)
        figure_paths.extend(head_to_head_figure(cells, destination))

    return {
        "question": outcome.var,
        "pair": {"cultures": list(pair), "countries": list(shared)},
        "models": per_model,
        "figures": [str(path) for path in figure_paths],
        "tables": [str(path) for path in table_paths],
    }
