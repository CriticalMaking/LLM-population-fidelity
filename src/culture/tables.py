from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from machine_bias_reproduction.config import paths_for
from machine_bias_reproduction.metrics import DISTANCES
from machine_bias_reproduction.questions import Question

from . import capacity as capacity_module
from .matching import MATCHED_CULTURES, countries_for, restrict
from .palette import MODES
from .registry import PREFLIGHT_MIN_VALID_ANSWER_MASS, CultureModel


def read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.is_file() else None


def read_csv_tsv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, sep="\t") if path.is_file() else None


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as stream:
        payload: dict[str, Any] = json.load(stream)
    return payload


def _preflight(path: Path) -> dict[str, Any]:
    probe = read_json(path).get("preflight") or {}
    return {"mean_mass": probe.get("mean_mass"), "informative": probe.get("informative")}


def load_model_frames(
    model: CultureModel,
    cultures: list[str],
    question: Question,
) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for culture in cultures:
        paths = paths_for(f"culture/{model.key}/{culture}", question.var)
        distances = read_csv(paths.outputs / "subpopulation_distances.csv")
        capacity = capacity_module.read(paths)
        if distances is None and capacity is None:
            continue
        manifest = read_json(paths.outputs / "run_manifest.json")
        loaded[culture] = {
            "distances": distances,
            "capacity": capacity,
            "quality": read_csv(paths.outputs / "quality_bands.csv"),
            "responses": read_csv(paths.outputs / "response_distributions.csv"),
            "summary": read_csv(paths.outputs / "summary_metrics.csv"),
            "fit": read_csv(paths.outputs / "regression_fit.csv"),
            "coverage": manifest.get("coverage", {}),
            "preflight": _preflight(paths.outputs / "inference_manifest.json"),
        }
    return loaded


def valid_rate(tables: dict[str, Any]) -> float | None:
    capacity = tables.get("capacity")
    if capacity is None or capacity.empty:
        return None
    prompts = float(capacity["prompts"].sum())
    return float(capacity["valid"].sum()) / prompts if prompts else 0.0


def is_flagged(tables: dict[str, Any]) -> bool:
    rate = valid_rate(tables)
    if rate is not None:
        return rate < PREFLIGHT_MIN_VALID_ANSWER_MASS
    return tables.get("preflight", {}).get("informative") is False


def has_distances(tables: dict[str, Any]) -> bool:
    return tables.get("distances") is not None


def reference_distances(question: Question) -> pd.DataFrame | None:
    return read_csv(paths_for("archived", question.var).outputs / "subpopulation_distances.csv")


def series(
    frames: dict[str, dict[str, Any]],
    mode: str,
    column: str = "nEMD",
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for culture, tables in frames.items():
        if not has_distances(tables):
            continue
        distances = tables["distances"]
        selected = distances.loc[distances["method"] == mode, column]
        if not selected.empty:
            values[culture] = selected.to_numpy(dtype=np.float64)
    return values


def matched_frames(frames: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    matched: dict[str, dict[str, Any]] = {}
    for culture in MATCHED_CULTURES:
        tables = frames.get(culture)
        if tables is None or not has_distances(tables):
            continue
        restricted = restrict(tables["distances"], culture)
        if restricted.empty:
            continue
        matched[culture] = {**tables, "distances": restricted}
    return matched


def matched_table(matched: dict[str, dict[str, Any]], model: CultureModel) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for culture, tables in matched.items():
        distances = tables["distances"]
        for mode in MODES:
            values = distances.loc[distances["method"] == mode, "nEMD"].to_numpy(np.float64)
            if not values.size:
                continue
            record: dict[str, Any] = {
                "model_key": model.key,
                "model_label": model.label,
                "culture": culture,
                "countries": "; ".join(countries_for(culture)),
                "mode": mode,
                "subpopulations": int(values.size),
                "mean_nEMD": float(values.mean()),
                "median_nEMD": float(np.median(values)),
            }
            for name in DISTANCES:
                if name == "nEMD" or name not in distances.columns:
                    continue
                column = distances.loc[distances["method"] == mode, name].to_numpy(np.float64)
                record[f"mean_{name}"] = float(column.mean())
            rows.append(record)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["mode", "mean_nEMD"]).reset_index(drop=True)


def comparison_table(model: CultureModel, frames: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for culture, tables in frames.items():
        coverage = tables.get("coverage") or {}
        rate = valid_rate(tables)
        base: dict[str, Any] = {
            "model_key": model.key,
            "model_label": model.label,
            "culture": culture,
            "valid_answer_rate": rate,
            "valid_answer_mass": tables.get("preflight", {}).get("mean_mass"),
            "informative": (not is_flagged(tables)) if rate is not None else None,
            "subpopulations_retained": coverage.get("subpopulations_retained"),
            "subpopulations_dropped": coverage.get("subpopulations_dropped"),
        }
        if not has_distances(tables):
            rows.append({**base, "mode": None})
            continue
        distances = tables["distances"]
        for mode in MODES:
            selected = distances[distances["method"] == mode]
            values = selected["nEMD"].to_numpy(np.float64)
            if not values.size:
                continue
            record: dict[str, Any] = {
                **base,
                "mode": mode,
                "subpopulations": int(values.size),
                "mean_nEMD": float(values.mean()),
                "median_nEMD": float(np.median(values)),
                "p90_nEMD": float(np.percentile(values, 90)),
            }
            for name in DISTANCES:
                if name != "nEMD" and name in selected.columns:
                    record[f"mean_{name}"] = float(selected[name].to_numpy(np.float64).mean())
            summary = tables.get("summary")
            if summary is not None:
                overall = summary[
                    (summary["metric"] == "overall_nEMD") & (summary["method"] == mode)
                ]
                if not overall.empty:
                    record["overall_nEMD"] = float(overall.iloc[0]["value"])
            fit = tables.get("fit")
            if fit is not None:
                selected_fit = fit[(fit["mode"] == mode.lower()) & (fit["model"] == "social")]
                if not selected_fit.empty:
                    record["social_adj_r_squared"] = float(
                        selected_fit.iloc[0]["adjusted_r_squared"]
                    )
            rows.append(record)
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["mode", "mean_nEMD"], na_position="last")
        .reset_index(drop=True)
    )
