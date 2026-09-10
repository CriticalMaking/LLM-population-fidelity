from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from machine_bias_reproduction.config import FIRST_REPLICATE, RunPaths, paths_for
from machine_bias_reproduction.figures import label_fit, save_plate
from machine_bias_reproduction.inference import EVENT_LOG_NAME
from machine_bias_reproduction.plates import GRID, INK, MUTED_INK, bar_layout
from machine_bias_reproduction.questions import Question, resolve_questions

from .adapters import health_table
from .matching import WVS_COUNTRIES, country_of, home_splits
from .palette import MDS_REFERENCES as REFERENCES
from .palette import MODES, arm_tone, model_tone
from .registry import (
    ADAPTER_MIN_EVAL_TOKEN_ACCURACY,
    BASE_ARM,
    CULTURE_FIGURES,
    CULTURE_ROOT,
    CultureModel,
    arm_display,
    is_base,
)
from .tables import read_csv, read_csv_tsv


def _summary_metrics(source: str, question: Question) -> dict[tuple[str, str], float] | None:
    frame = read_csv(paths_for(source, question.var).outputs / "summary_metrics.csv")
    if frame is None:
        return None
    return {
        (str(metric), str(method)): float(value)
        for metric, method, value in zip(
            frame["metric"], frame["method"], frame["value"], strict=True
        )
    }


def _distances(source: str, question: Question) -> pd.DataFrame | None:
    frame = read_csv(paths_for(source, question.var).outputs / "subpopulation_distances.csv")
    if frame is None:
        return None
    frame = frame.copy()
    frame["country"] = country_of(frame["subpopulation"])
    return frame


def _sources(model: CultureModel, cultures: list[str]) -> list[tuple[str, str]]:
    return [
        *REFERENCES,
        *((culture, f"culture/{model.key}/{culture}") for culture in cultures),
    ]


def distance_table(
    model: CultureModel,
    cultures: list[str],
    questions: list[Question],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for question in questions:
        for label, source in _sources(model, cultures):
            metrics = _summary_metrics(source, question)
            if metrics is None:
                continue
            survey = metrics.get(("median_pairwise_nEMD", "WVS"))
            for mode in MODES:
                overall = metrics.get(("overall_nEMD", mode))
                spread = metrics.get(("median_pairwise_nEMD", mode))
                if overall is None:
                    continue
                rows.append(
                    {
                        "model_key": model.key,
                        "model_label": model.label,
                        "question": question.var,
                        "question_label": question.label,
                        "series": label,
                        "mode": mode,
                        "overall_nEMD": overall,
                        "survey_median_pairwise_nEMD": survey,
                        "model_median_pairwise_nEMD": spread,
                        "compression": (survey / spread if survey is not None and spread else None),
                    }
                )
    return pd.DataFrame(rows)


MIXTRAL_REFERENCE = "Mixtral archived"
BASE_REFERENCE = "as released (same model)"


def home_advantage_table(
    model: CultureModel,
    cultures: list[str],
    questions: list[Question],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for question in questions:
        references = {
            MIXTRAL_REFERENCE: _distances("archived", question),
            BASE_REFERENCE: _distances(f"culture/{model.key}/{BASE_ARM}", question),
        }
        for culture in cultures:
            if is_base(culture):
                continue
            tuned = _distances(f"culture/{model.key}/{culture}", question)
            if tuned is None:
                continue
            for name, reference in references.items():
                if reference is None:
                    continue
                for label, countries in home_splits(culture):
                    for mode in MODES:
                        row = _home_advantage_row(tuned, reference, mode, countries)
                        if row is None:
                            continue
                        rows.append(
                            {
                                "model_key": model.key,
                                "model_label": model.label,
                                "question": question.var,
                                "question_label": question.label,
                                "culture": culture,
                                "reference": name,
                                "home_country": label,
                                "mode": mode,
                                **row,
                            }
                        )
    return pd.DataFrame(rows)


def _home_advantage_row(
    tuned: pd.DataFrame,
    reference: pd.DataFrame,
    mode: str,
    countries: tuple[str, ...],
) -> dict[str, Any] | None:
    model_rows = tuned[tuned["method"] == mode]
    shared = reference[
        (reference["method"] == mode) & reference["subpopulation"].isin(model_rows["subpopulation"])
    ]
    if model_rows.empty or shared.empty:
        return None
    model_home = model_rows["country"].isin(countries)
    shared_home = shared["country"].isin(countries)
    if not model_home.any() or not (~model_home).any():
        return None
    tuned_home = float(model_rows.loc[model_home, "nEMD"].mean())
    tuned_away = float(model_rows.loc[~model_home, "nEMD"].mean())
    reference_home = float(shared.loc[shared_home, "nEMD"].mean())
    reference_away = float(shared.loc[~shared_home, "nEMD"].mean())
    return {
        "subpopulations_home": int(model_home.sum()),
        "subpopulations_away": int((~model_home).sum()),
        "tuned_home_nEMD": tuned_home,
        "tuned_away_nEMD": tuned_away,
        "reference_home_nEMD": reference_home,
        "reference_away_nEMD": reference_away,
        "difference_in_differences": (tuned_home - tuned_away) - (reference_home - reference_away),
    }


ALL_COUNTRIES = "(all)"


def _full_coefficients(source: str, question: Question) -> pd.DataFrame | None:
    frame = read_csv(paths_for(source, question.var).outputs / "full_coefficients.csv")
    if frame is None:
        return None
    return frame[frame["model"] == "full"]


def _coefficient(frame: pd.DataFrame | None, mode: str, predictor: str) -> float | None:
    if frame is None:
        return None
    cell = frame[(frame["mode"] == mode.lower()) & (frame["predictor"] == predictor)]
    return float(cell.iloc[0]["estimate"]) if not cell.empty else None


def _paired_means(
    base: pd.DataFrame,
    arm: pd.DataFrame,
    mode: str,
    country: str | None,
) -> tuple[int, float, float] | None:
    base_rows = base[base["method"] == mode]
    arm_rows = arm[arm["method"] == mode]
    if country is not None:
        base_rows = base_rows[base_rows["country"] == country]
        arm_rows = arm_rows[arm_rows["country"] == country]
    shared = set(base_rows["subpopulation"]) & set(arm_rows["subpopulation"])
    if not shared:
        return None
    base_mean = float(base_rows[base_rows["subpopulation"].isin(shared)]["nEMD"].mean())
    arm_mean = float(arm_rows[arm_rows["subpopulation"].isin(shared)]["nEMD"].mean())
    return len(shared), base_mean, arm_mean


def base_delta_table(
    model: CultureModel,
    cultures: list[str],
    questions: list[Question],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for question in questions:
        base = _distances(f"culture/{model.key}/{BASE_ARM}", question)
        if base is None:
            continue
        base_fit = _full_coefficients(f"culture/{model.key}/{BASE_ARM}", question)
        for culture in cultures:
            if is_base(culture):
                continue
            arm = _distances(f"culture/{model.key}/{culture}", question)
            if arm is None:
                continue
            arm_fit = _full_coefficients(f"culture/{model.key}/{culture}", question)
            for mode in MODES:
                for country in (ALL_COUNTRIES, *WVS_COUNTRIES):
                    paired = _paired_means(
                        base, arm, mode, None if country == ALL_COUNTRIES else country
                    )
                    if paired is None:
                        continue
                    subpopulations, base_mean, arm_mean = paired
                    predictor = None if country == ALL_COUNTRIES else f"country{country}"
                    rows.append(
                        {
                            "model_key": model.key,
                            "model_label": model.label,
                            "question": question.var,
                            "question_label": question.label,
                            "arm": culture,
                            "mode": mode,
                            "country": country,
                            "subpopulations": subpopulations,
                            "base_mean_nEMD": base_mean,
                            "arm_mean_nEMD": arm_mean,
                            "delta_nEMD": arm_mean - base_mean,
                            **_coefficient_columns(base_fit, arm_fit, mode, predictor),
                        }
                    )
    return pd.DataFrame(rows)


def _coefficient_columns(
    base_fit: pd.DataFrame | None,
    arm_fit: pd.DataFrame | None,
    mode: str,
    predictor: str | None,
) -> dict[str, float | None]:
    center_base = _coefficient(base_fit, mode, "nEMD_center")
    center_arm = _coefficient(arm_fit, mode, "nEMD_center")
    country_base = _coefficient(base_fit, mode, predictor) if predictor else None
    country_arm = _coefficient(arm_fit, mode, predictor) if predictor else None
    return {
        "nEMD_center_base": center_base,
        "nEMD_center_arm": center_arm,
        "nEMD_center_delta": _difference(center_base, center_arm),
        "beta_country_base": country_base,
        "beta_country_arm": country_arm,
        "beta_country_delta": _difference(country_base, country_arm),
    }


def _difference(base: float | None, arm: float | None) -> float | None:
    return None if base is None or arm is None else arm - base


def session_hours(paths: RunPaths) -> tuple[float, int] | None:
    path = paths.logs / EVENT_LOG_NAME
    if not path.is_file():
        return None
    sessions: dict[str, list[datetime]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            stamp = datetime.fromisoformat(event["created_at"])
            span = sessions.setdefault(event["run_id"], [stamp, stamp])
            span[0] = min(span[0], stamp)
            span[1] = max(span[1], stamp)
    if not sessions:
        return None
    seconds = sum((last - first).total_seconds() for first, last in sessions.values())
    return seconds / 3600.0, len(sessions)


def sweep_cost_table(
    model: CultureModel,
    cultures: list[str],
    questions: list[Question],
) -> pd.DataFrame:
    logged = [
        read_csv_tsv(CULTURE_ROOT / "logs" / name)
        for name in ("sweep_summary.tsv", "base_summary.tsv")
    ]
    present = [frame for frame in logged if frame is not None]
    summary = pd.concat(present, ignore_index=True) if present else None
    if summary is not None and "replicate" in summary.columns:
        first = pd.to_numeric(summary["replicate"], errors="coerce").fillna(FIRST_REPLICATE)
        summary = summary[first.eq(FIRST_REPLICATE)]
    rows: list[dict[str, Any]] = []
    for culture in cultures:
        for question in questions:
            paths = paths_for(f"culture/{model.key}/{culture}", question.var)
            timing = session_hours(paths)
            if timing is None:
                continue
            hours, sessions = timing
            recorded = None
            if summary is not None:
                cell = summary[
                    (summary["model"] == model.key)
                    & (summary["culture"] == culture)
                    & (summary["question"] == question.var)
                ]
                recorded = str(cell.iloc[0]["status"]) if not cell.empty else None
            rows.append(
                {
                    "model_key": model.key,
                    "model_label": model.label,
                    "culture": culture,
                    "question": question.var,
                    "question_label": question.label,
                    "hours": hours,
                    "sessions": sessions,
                    "sweep_status": recorded,
                }
            )
    return pd.DataFrame(rows)


def _grouped_bars(
    axis: Any,
    frame: pd.DataFrame,
    value: str,
    questions: list[Question],
    series_order: list[str],
) -> None:
    width, offsets = bar_layout(len(series_order))
    positions = np.arange(len(questions))
    for index, series in enumerate(series_order):
        offset = offsets[index]
        heights: list[float] = []
        missing: list[float] = []
        for slot, question in enumerate(questions):
            cell = frame[(frame["series"] == series) & (frame["question"] == question.var)]
            if cell.empty or pd.isna(cell.iloc[0][value]):
                heights.append(0.0)
                missing.append(positions[slot] + offset)
            else:
                heights.append(float(cell.iloc[0][value]))
        tone = arm_tone(series)
        axis.bar(
            positions + offset,
            heights,
            width,
            label=arm_display(series),
            color=tone.fill,
            edgecolor=tone.ink,
            linewidth=0.7,
        )
        for position in missing:
            axis.text(position, 0.0, "n/a", ha="center", va="bottom", fontsize=8, color=MUTED_INK)
    axis.set_xticks(positions, [question.label for question in questions])
    axis.yaxis.grid(True, color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)


def _series_order(frame: pd.DataFrame, cultures: list[str]) -> list[str]:
    present = set(frame["series"])
    return [label for label, _ in REFERENCES if label in present] + [
        culture for culture in cultures if culture in present
    ]


def overall_figure(
    frame: pd.DataFrame,
    questions: list[Question],
    cultures: list[str],
    destination: Path,
) -> list[Path]:
    ntp = frame[frame["mode"] == "NTP"]
    figure, axis = plt.subplots(figsize=(11, 5.0))
    _grouped_bars(axis, ntp, "overall_nEMD", questions, _series_order(ntp, cultures))
    axis.set_ylabel("Overall nEMD (lower is closer to the WVS)")
    axis.legend(ncol=5, loc="upper left")
    axis.set_ylim(0, max(0.42, float(ntp["overall_nEMD"].max()) * 1.35))
    figure.tight_layout()
    return save_plate(figure, destination / "fig_overall_nemd")


def compression_figure(
    frame: pd.DataFrame,
    questions: list[Question],
    cultures: list[str],
    destination: Path,
) -> list[Path]:
    ntp = frame[frame["mode"] == "NTP"]
    figure, axis = plt.subplots(figsize=(11, 5.0))
    _grouped_bars(axis, ntp, "compression", questions, _series_order(ntp, cultures))
    axis.axhline(1.0, color=INK, linestyle=(0, (4, 2)), linewidth=1.0, zorder=0)
    axis.set_ylabel("Compression ratio (survey spread / model spread)")
    axis.legend(ncol=5, loc="upper left")
    axis.set_ylim(0, max(7.6, float(ntp["compression"].max()) * 1.25))
    figure.tight_layout()
    return save_plate(figure, destination / "fig_compression")


def base_delta_figure(
    frame: pd.DataFrame,
    questions: list[Question],
    cultures: list[str],
    destination: Path,
) -> list[Path]:
    ntp = frame[(frame["mode"] == "NTP") & (frame["country"] != ALL_COUNTRIES)]
    if ntp.empty:
        return []
    arms = [culture for culture in cultures if culture in set(ntp["arm"])]
    countries = [name for name in WVS_COUNTRIES if name in set(ntp["country"])]
    width, offsets = bar_layout(len(arms))
    positions = np.arange(len(countries))
    span = float(ntp["delta_nEMD"].abs().max()) or 1.0
    figure, axes = plt.subplots(
        1, len(questions), figsize=(5.0 * len(questions), 5.0), sharey=True, squeeze=False
    )
    for axis, question in zip(axes[0], questions, strict=True):
        rows = ntp[ntp["question"] == question.var]
        for index, arm in enumerate(arms):
            offset = offsets[index]
            cells = rows[rows["arm"] == arm].set_index("country")["delta_nEMD"]
            heights = [float(cells[name]) if name in cells.index else 0.0 for name in countries]
            tone = arm_tone(arm)
            axis.bar(
                positions + offset,
                heights,
                width,
                label=arm_display(arm) if question is questions[0] else None,
                color=tone.fill,
                edgecolor=tone.ink,
                linewidth=0.7,
            )
        axis.axhline(0, color=INK, linewidth=1.0)
        axis.set_xticks(positions, countries, rotation=30, ha="right")
        axis.set_ylim(-span * 1.25, span * 1.25)
        axis.yaxis.grid(True, color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        label_fit(axis, question.label)
    axes[0][0].set_ylabel("nEMD after finetuning minus before")
    axes[0][0].legend(ncol=3, loc="upper left")
    figure.tight_layout()
    return save_plate(figure, destination / "fig_base_delta")


def home_advantage_figure(
    frame: pd.DataFrame,
    questions: list[Question],
    cultures: list[str],
    destination: Path,
    stem: str = "fig_home_advantage",
) -> list[Path]:
    ntp = frame[frame["mode"] == "NTP"]
    if ntp.empty:
        return []
    present = [culture for culture in cultures if culture in set(ntp["culture"])]
    width, offsets = bar_layout(len(present))
    positions = np.arange(len(questions))
    span = ntp["difference_in_differences"]
    figure, axis = plt.subplots(figsize=(11, 5.0))
    axis.set_ylim(float(span.min()) * 1.30, float(span.max()) * 1.30)
    for index, culture in enumerate(present):
        offset = offsets[index]
        rows = ntp[ntp["culture"] == culture]
        tone = arm_tone(culture)
        for slot, question in enumerate(questions):
            cell = rows[rows["question"] == question.var]
            if cell.empty:
                axis.text(
                    positions[slot] + offset,
                    0.004,
                    "n/a",
                    ha="center",
                    fontsize=8,
                    color=MUTED_INK,
                )
                continue
            splits = cell.set_index("home_country")["difference_in_differences"]
            combined = [name for name in splits.index if "; " in name] or list(splits.index)
            axis.bar(
                positions[slot] + offset,
                float(splits[combined[0]]),
                width,
                label=arm_display(culture) if slot == 0 else None,
                color=tone.fill,
                edgecolor=tone.ink,
                linewidth=0.7,
            )
            for name, value in splits.items():
                if name in combined:
                    continue
                axis.hlines(
                    float(value),
                    positions[slot] + offset - width / 2,
                    positions[slot] + offset + width / 2,
                    color=INK,
                    linewidth=1.2,
                    zorder=3,
                )
    axis.axhline(0, color=INK, linewidth=1.0)
    axis.set_xticks(positions, [question.label for question in questions])
    axis.set_ylabel("Difference-in-differences (nEMD)")
    axis.yaxis.grid(True, color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(ncol=3, loc="upper left")
    figure.tight_layout()
    return save_plate(figure, destination / stem)


def sweep_cost_figure(runs: pd.DataFrame, destination: Path) -> list[Path]:
    if runs.empty:
        return []
    labels = {question.var: question.label for question in resolve_questions(None)}
    order = [var for var in labels if var in set(runs["question"])]
    cultures = sorted(set(runs["culture"]))
    width, offsets = bar_layout(len(cultures))
    positions = np.arange(len(order))
    ceiling = float(runs["hours"].max())
    figure, axis = plt.subplots(figsize=(11, 4.6))
    for index, culture in enumerate(cultures):
        offset = offsets[index]
        rows = runs[runs["culture"] == culture].set_index("question")
        tone = arm_tone(culture)
        for slot, var in enumerate(order):
            if var not in rows.index:
                continue
            height = float(rows.loc[var, "hours"])
            axis.bar(
                positions[slot] + offset,
                height,
                width,
                label=arm_display(culture) if slot == 0 else None,
                color=tone.fill,
                edgecolor=tone.ink,
                linewidth=0.7,
            )
            axis.text(
                positions[slot] + offset,
                height + ceiling * 0.025,
                f"{height:.0f}h",
                ha="center",
                fontsize=8.5,
                color=MUTED_INK,
            )
    axis.set_xticks(positions, [labels[var] for var in order])
    axis.set_ylabel("GPU hours per run")
    axis.set_ylim(0, ceiling * 1.18)
    axis.yaxis.grid(True, color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(ncol=3, loc="upper left")
    figure.tight_layout()
    return save_plate(figure, destination / "fig_sweep_cost")


def adapter_health_figure(frame: pd.DataFrame, destination: Path) -> list[Path]:
    if frame.empty:
        return []
    cultures = sorted(set(frame["culture"]))
    models = list(dict.fromkeys(frame["model_key"]))
    positions = np.arange(len(cultures))
    height, offsets = bar_layout(len(models))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for index, model_key in enumerate(models):
        rows = frame[frame["model_key"] == model_key].set_index("culture")
        label = str(rows["model_label"].iloc[0])
        tone = model_tone(model_key, index)
        offset = offsets[index]
        present = [culture for culture in cultures if culture in rows.index]
        slots = [positions[cultures.index(culture)] + offset for culture in present]
        axes[0].scatter(
            [float(rows.loc[culture, "update_norm_mean"]) for culture in present],
            slots,
            label=label,
            color=tone.fill,
            edgecolors=tone.ink,
            linewidths=0.8,
            s=64,
            zorder=3,
        )
        axes[1].barh(
            slots,
            [float(rows.loc[culture, "eval_token_accuracy"]) for culture in present],
            height,
            label=label,
            color=tone.fill,
            edgecolor=tone.ink,
            linewidth=0.7,
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Mean ‖ΔW‖ added to each attention projection")
    axes[0].xaxis.grid(True, color=GRID, linewidth=0.6)

    axes[1].axvline(
        ADAPTER_MIN_EVAL_TOKEN_ACCURACY,
        color=INK,
        linestyle=(0, (4, 2)),
        linewidth=1.0,
        label=f"usable threshold ({ADAPTER_MIN_EVAL_TOKEN_ACCURACY:.0%})",
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Token accuracy on held-out training data")
    axes[1].xaxis.grid(True, color=GRID, linewidth=0.6)

    for axis in axes:
        axis.set_yticks(positions, cultures)
        axis.set_ylim(len(cultures) - 0.5, -0.5)
        axis.set_axisbelow(True)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=len(labels), loc="lower center")
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    return save_plate(figure, destination / "fig_adapter_health")


def _has_own_series(distances: pd.DataFrame) -> bool:
    if distances.empty:
        return False
    return bool(set(distances["series"]) - {label for label, _ in REFERENCES})


def _write(frame: pd.DataFrame, destination: Path) -> Path | None:
    if frame.empty:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def culture_summary(
    models: list[CultureModel],
    cultures: list[str],
    questions: list[str] | None = None,
) -> dict[str, Any]:
    outcomes = resolve_questions(questions)
    produced: list[Path] = []
    written: list[Path] = []

    for model in models:
        figures = CULTURE_FIGURES / model.key / "summary"
        outputs = CULTURE_ROOT / model.key / "summary"
        distances = distance_table(model, cultures, outcomes)
        advantage = home_advantage_table(model, cultures, outcomes)
        deltas = base_delta_table(model, cultures, outcomes)
        cost = sweep_cost_table(model, cultures, outcomes)
        if not _has_own_series(distances):
            continue
        drawn = [question for question in outcomes if question.var in set(distances["question"])]
        if not drawn:
            continue
        figures.mkdir(parents=True, exist_ok=True)
        produced.extend(overall_figure(distances, drawn, cultures, figures))
        produced.extend(compression_figure(distances, drawn, cultures, figures))
        for reference, stem in (
            (MIXTRAL_REFERENCE, "fig_home_advantage"),
            (BASE_REFERENCE, "fig_home_advantage_base"),
        ):
            against = (
                advantage[advantage["reference"] == reference] if not advantage.empty else advantage
            )
            if against.empty:
                continue
            shown = [question for question in outcomes if question.var in set(against["question"])]
            produced.extend(home_advantage_figure(against, shown, cultures, figures, stem))
        if not deltas.empty:
            shown = [question for question in outcomes if question.var in set(deltas["question"])]
            produced.extend(base_delta_figure(deltas, shown, cultures, figures))
        produced.extend(sweep_cost_figure(cost, figures))

        for frame, name in (
            (distances, "culture_summary.csv"),
            (advantage, "culture_home_advantage.csv"),
            (deltas, "base_deltas.csv"),
            (cost, "sweep_cost.csv"),
        ):
            path = _write(frame, outputs / name)
            if path is not None:
                written.append(path)

    health = health_table(models, [culture for culture in cultures if not is_base(culture)])
    if not health.empty:
        shared = CULTURE_FIGURES / "summary"
        shared.mkdir(parents=True, exist_ok=True)
        produced.extend(adapter_health_figure(health, shared))
        path = _write(health, CULTURE_ROOT / "adapter_health.csv")
        if path is not None:
            written.append(path)

    return {
        "questions": [question.var for question in outcomes],
        "figures": [str(path) for path in produced],
        "tables": [str(path) for path in written],
    }
