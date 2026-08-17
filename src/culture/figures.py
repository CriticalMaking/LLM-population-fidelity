from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from machine_bias_reproduction.questions import Question, resolve_question

from .mds import degenerate_ntp_plate
from .panels import (
    capacity_figure,
    country_heatmap,
    cross_model_figure,
    density_figure,
    distance_density_figure,
    matched_figures,
    quality_figure,
    ranking_figure,
    response_shift,
    run_capacity_figure,
)
from .registry import CULTURE_FIGURES, CULTURE_ROOT, CultureModel, run_paths
from .tables import comparison_table, has_distances, load_model_frames


def _model_figures(
    model: CultureModel,
    frames: dict[str, dict[str, Any]],
    question: Question,
    destination: Path,
) -> tuple[list[Path], pd.DataFrame]:
    destination.mkdir(parents=True, exist_ok=True)
    figures = list(capacity_figure(frames, destination))
    if any(has_distances(tables) for tables in frames.values()):
        figures.extend(density_figure(frames, question, destination))
        figures.extend(distance_density_figure(frames, destination))
        figures.extend(quality_figure(frames, destination))
        figures.extend(ranking_figure(frames, question, destination))
        figures.extend(country_heatmap(frames, destination))
        figures.extend(response_shift(frames, question, destination))
    matched_paths, matched = matched_figures(model, frames, question, destination / "matched")
    figures.extend(matched_paths)
    return figures, matched


def _write(frame: pd.DataFrame, destination: Path) -> None:
    if frame.empty:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)


def _unplaced_figures(
    model: CultureModel,
    culture: str,
    tables: dict[str, Any],
    question: Question,
) -> list[Path]:
    coverage = tables.get("coverage") or {}
    if coverage.get("subpopulations_retained"):
        return []
    capacity = tables.get("capacity")
    if capacity is None or capacity.empty:
        return []
    destination = run_paths(model.key, culture, question).figures
    destination.mkdir(parents=True, exist_ok=True)
    produced = list(run_capacity_figure(capacity, coverage, destination))

    ntp = capacity[capacity["mode"] == "NTP"]
    row = ntp.iloc[0] if not ntp.empty else None
    distinct = (
        None
        if row is None or pd.isna(row["distinct_distributions"])
        else int(row["distinct_distributions"])
    )
    mass = None if row is None or pd.isna(row["mean_valid_mass"]) else float(row["mean_valid_mass"])
    produced.extend(
        degenerate_ntp_plate(
            model,
            culture,
            question,
            valid_answer_mass=mass,
            distinct_answers=distinct,
        )
    )
    return produced


def compare_cultures(
    models: list[CultureModel],
    cultures: list[str],
    question: str | Question = "d_happy",
) -> dict[str, Any]:
    outcome = resolve_question(question)
    tables: dict[str, pd.DataFrame] = {}
    produced: list[Path] = []
    per_model: dict[str, list[str]] = {}

    for model in models:
        frames = load_model_frames(model, cultures, outcome)
        per_model[model.key] = sorted(frames)
        if not frames:
            continue
        figures, matched = _model_figures(
            model, frames, outcome, CULTURE_FIGURES / model.key / outcome.var
        )
        produced.extend(figures)
        for culture, culture_tables in frames.items():
            produced.extend(_unplaced_figures(model, culture, culture_tables, outcome))

        table = comparison_table(model, frames)
        tables[model.key] = table
        output_directory = CULTURE_ROOT / model.key / outcome.var
        _write(table, output_directory / "culture_comparison.csv")
        if not matched.empty:
            matched["model_key"] = model.key
            matched["model_label"] = model.label
        _write(matched, output_directory / "culture_matched_comparison.csv")

    if len(tables) > 1:
        cross_model = CULTURE_FIGURES / outcome.var
        cross_model.mkdir(parents=True, exist_ok=True)
        produced.extend(cross_model_figure(tables, outcome, cross_model))
    if tables:
        combined = pd.concat(tables.values(), ignore_index=True)
        _write(combined, CULTURE_ROOT / outcome.var / "culture_comparison.csv")

    return {
        "question": outcome.var,
        "models": per_model,
        "figures": len(produced),
    }
