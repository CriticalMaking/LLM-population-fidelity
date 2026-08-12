"""Cross-question summaries: the three claims the culture extension rests on.

Every other culture figure answers one question at a time. These read across
happiness, politics and religion together, because each of the three claims is
about the *pattern* over outcomes rather than about any one of them:

``fig_overall_nemd``
    How far each model sits from the survey overall. Lower is closer.
``fig_compression``
    How much each model squashes the spread between subpopulations. Above 1.0 the
    model makes groups more alike than they really are -- the original paper's
    failure mode; below 1.0 it spreads them wider than the survey does.
``fig_home_advantage``
    Whether a finetuned culture MLLM is closer to its *own* respondents than a
    reference is, once the reference's own head start on those same respondents
    is differenced out.

Each figure has a CSV beside it carrying exactly the plotted numbers, so a value
quoted anywhere downstream traces back to a file that can be re-derived.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd

from machine_bias_reproduction.config import RunPaths, paths_for
from machine_bias_reproduction.figures import _save
from machine_bias_reproduction.inference import EVENT_LOG_NAME
from machine_bias_reproduction.questions import Question, resolve_questions

from .matching import country_of, home_splits
from .palette import MODES, mds_color
from .registry import CULTURE_FIGURES, CULTURE_ROOT, CultureModel
from .tables import read_csv, read_csv_tsv

matplotlib.use("Agg")
from matplotlib import pyplot as plt

REFERENCES: tuple[tuple[str, str], ...] = (
    ("Mixtral archived", "archived"),
    ("Mixtral fresh", "fresh"),
)
"""The paper's own model, twice: its published run and a fresh re-run.

Drawn side by side on purpose. The gap between two runs of one model is the size
of run-to-run noise, which is the only yardstick these charts offer for whether a
gap between models is worth reading.
"""

GRID = "#d8d8d8"
INK = "#1a1a1a"

RC = {
    "font.size": 12,
    "axes.edgecolor": "#888888",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


@contextmanager
def _styled() -> Iterator[None]:
    """Draw one figure under this module's shared chart styling.

    ``rc_context`` is typed against matplotlib's literal key set, which a plain
    dict cannot satisfy; the cast is confined here rather than at each figure.
    """
    with plt.rc_context(cast(Any, RC)):
        yield


def _series_color(series: str) -> str:
    """Colour a bar exactly as the MDS plates colour the same series.

    ``mds_color`` already spends black and grey on the two Mixtral references and
    lifts Spanish out of grey to keep it distinguishable from them, which is the
    same collision these bars have.
    """
    return mds_color(series)


def _summary_metrics(source: str, question: Question) -> dict[tuple[str, str], float] | None:
    """Return one run's ``summary_metrics.csv`` keyed by (metric, method)."""
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
    """Return one run's subpopulation distances, with the country broken out."""
    frame = read_csv(paths_for(source, question.var).outputs / "subpopulation_distances.csv")
    if frame is None:
        return None
    frame = frame.copy()
    frame["country"] = country_of(frame["subpopulation"])
    return frame


def _sources(model: CultureModel, cultures: list[str]) -> list[tuple[str, str]]:
    """Return every series to summarise, as (label, run source), references first."""
    return [
        *REFERENCES,
        *((culture, f"culture/{model.key}/{culture}") for culture in cultures),
    ]


def distance_table(
    model: CultureModel,
    cultures: list[str],
    questions: list[Question],
) -> pd.DataFrame:
    """Tabulate overall distance and dispersion for every series and question.

    ``compression`` is the survey's own median pairwise distance divided by the
    model's. It is the paper's homogeneity measure: above 1.0 the model's
    subpopulations sit closer together than the real ones do.
    """
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


def home_advantage_table(
    model: CultureModel,
    cultures: list[str],
    questions: list[Question],
) -> pd.DataFrame:
    """Difference out the reference's head start on a culture's own respondents.

    A finetuned culture MLLM sitting closer to its own countries than to the rest
    proves nothing on its own: those respondents may simply be easier for every
    model. So the same gap is measured for the reference, on the *same*
    subpopulations, and subtracted. Below zero is a real home advantage.

    One row per home slice: the pooled countries, and -- where a culture spans
    more than one -- each country on its own, so a pooled result cannot hide two
    country effects pulling in opposite directions.
    """
    rows: list[dict[str, Any]] = []
    for question in questions:
        reference = _distances("archived", question)
        if reference is None:
            continue
        for culture in cultures:
            tuned = _distances(f"culture/{model.key}/{culture}", question)
            if tuned is None:
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
    """Compute one difference-in-differences cell on identical subpopulations."""
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


def session_hours(paths: RunPaths) -> tuple[float, int] | None:
    """Return one run's generation hours and how many sessions produced them.

    Timed from the run's own ``inference_events.jsonl`` rather than from the
    sweep summary, because the summary is written by the sweep and therefore only
    knows about runs the sweep itself launched:

    * a run produced by an earlier sweep is recorded as ``skipped`` with no
      duration at all, and would otherwise vanish from the chart;
    * a run that was interrupted and resumed is recorded with the *last*
      invocation's wall clock, understating what it cost.

    Each ``run_id`` in the log is one generation session; the idle time between
    sessions is not compute and is excluded by spanning each separately.
    """
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
    """Tabulate what each finished run cost, timed from its own event log.

    ``sweep_status`` carries what the sweep summary said about the same cell, so
    a row whose hours came from the log alone is still identifiable as one the
    sweep never timed.
    """
    summary = read_csv_tsv(CULTURE_ROOT / "logs" / "sweep_summary.tsv")
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
                    (summary["culture"] == culture) & (summary["question"] == question.var)
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
    """Draw one bar per series within each question, in a stable series order."""
    width = 0.8 / max(len(series_order), 1)
    positions = np.arange(len(questions))
    for index, series in enumerate(series_order):
        offset = (index - (len(series_order) - 1) / 2) * width
        heights: list[float] = []
        missing: list[float] = []
        for slot, question in enumerate(questions):
            cell = frame[(frame["series"] == series) & (frame["question"] == question.var)]
            if cell.empty or pd.isna(cell.iloc[0][value]):
                heights.append(0.0)
                missing.append(positions[slot] + offset)
            else:
                heights.append(float(cell.iloc[0][value]))
        axis.bar(
            positions + offset,
            heights,
            width,
            label=series,
            color=_series_color(series),
            edgecolor="white",
            linewidth=0.6,
        )
        for x in missing:
            axis.text(x, 0.0, "n/a", ha="center", va="bottom", fontsize=8, color="#888888")
    axis.set_xticks(positions, [question.label for question in questions])
    axis.yaxis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)


def _series_order(frame: pd.DataFrame, cultures: list[str]) -> list[str]:
    """References first, then the cultures actually present, in the given order."""
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
    """Chart how far each series sits from the survey, next-token probabilities."""
    ntp = frame[frame["mode"] == "NTP"]
    with _styled():
        figure, axis = plt.subplots(figsize=(11, 5.2))
        _grouped_bars(axis, ntp, "overall_nEMD", questions, _series_order(ntp, cultures))
        axis.set_ylabel("Overall nEMD (lower = closer)")
        axis.set_title(
            "Distance from the survey's answer distribution", fontsize=14, pad=14, loc="left"
        )
        axis.legend(frameon=False, ncol=5, fontsize=10, loc="upper left")
        axis.set_ylim(0, max(0.42, float(ntp["overall_nEMD"].max()) * 1.35))
        figure.tight_layout()
        return _save(figure, destination / "fig_overall_nemd")


def compression_figure(
    frame: pd.DataFrame,
    questions: list[Question],
    cultures: list[str],
    destination: Path,
) -> list[Path]:
    """Chart how much each series compresses the spread between subpopulations."""
    ntp = frame[frame["mode"] == "NTP"]
    with _styled():
        figure, axis = plt.subplots(figsize=(11, 5.2))
        _grouped_bars(axis, ntp, "compression", questions, _series_order(ntp, cultures))
        axis.axhline(1.0, color="#d73027", linestyle="--", linewidth=1.4, zorder=0)
        axis.set_ylabel("Compression ratio (survey spread / model spread)")
        axis.set_title(
            "How much the model flattens the differences between groups",
            fontsize=14,
            pad=14,
            loc="left",
        )
        axis.legend(frameon=False, ncol=5, fontsize=10, loc="upper left")
        axis.set_ylim(0, max(7.6, float(ntp["compression"].max()) * 1.25))
        axis.annotate(
            "above 1.0 = model makes groups\nmore alike than they really are",
            xy=(len(questions) - 0.68, 1.05),
            fontsize=9,
            color="#d73027",
            ha="right",
        )
        figure.tight_layout()
        return _save(figure, destination / "fig_compression")


def home_advantage_figure(
    frame: pd.DataFrame,
    questions: list[Question],
    cultures: list[str],
    destination: Path,
) -> list[Path]:
    """Chart the pooled difference-in-differences, with each country marked.

    The bar is the pooled home slice. Where a culture spans more than one country
    each country's own value is drawn over the bar as a tick, so a pooled bar that
    hides a split between its countries cannot be read as a single effect.
    """
    ntp = frame[frame["mode"] == "NTP"]
    present = [culture for culture in cultures if culture in set(ntp["culture"])]
    width = 0.8 / max(len(present), 1)
    positions = np.arange(len(questions))
    # Headroom for the per-country ticks, which can sit well outside the pooled
    # bars they annotate -- English religion is +0.43 over a +0.16 bar.
    span = ntp["difference_in_differences"]
    with _styled():
        figure, axis = plt.subplots(figsize=(11, 5.2))
        axis.set_ylim(float(span.min()) * 1.30, float(span.max()) * 1.30)
        for index, culture in enumerate(present):
            offset = (index - (len(present) - 1) / 2) * width
            rows = ntp[ntp["culture"] == culture]
            for slot, question in enumerate(questions):
                cell = rows[rows["question"] == question.var]
                if cell.empty:
                    axis.text(
                        positions[slot] + offset,
                        0.004,
                        "n/a",
                        ha="center",
                        fontsize=8,
                        color="#888888",
                    )
                    continue
                splits = cell.set_index("home_country")["difference_in_differences"]
                combined = [name for name in splits.index if "; " in name] or list(splits.index)
                axis.bar(
                    positions[slot] + offset,
                    float(splits[combined[0]]),
                    width,
                    label=culture if slot == 0 else None,
                    color=_series_color(culture),
                    edgecolor="white",
                    linewidth=0.6,
                )
                for name, value in splits.items():
                    if name in combined:
                        continue
                    axis.hlines(
                        float(value),
                        positions[slot] + offset - width / 2,
                        positions[slot] + offset + width / 2,
                        color=INK,
                        linewidth=1.4,
                        zorder=3,
                    )
        axis.axhline(0, color=INK, linewidth=1.2)
        axis.set_xticks(positions, [question.label for question in questions])
        axis.set_ylabel("Difference-in-differences (nEMD)")
        axis.set_title(
            "Is the model closer to its own culture than the reference is?",
            fontsize=14,
            pad=14,
            loc="left",
        )
        axis.yaxis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, ncol=3, fontsize=10, loc="upper left")
        axis.annotate(
            "below 0 = really is closer to its own culture  ·  "
            "black ticks = one country on its own",
            xy=(0.01, 0.03),
            xycoords="axes fraction",
            fontsize=9,
            color="#1b9e77",
        )
        figure.tight_layout()
        return _save(figure, destination / "fig_home_advantage")


def sweep_cost_figure(runs: pd.DataFrame, destination: Path) -> list[Path]:
    """Chart the wall clock of each run, one bar per culture within each outcome.

    Every bar is drawn the same way. A run that took two sittings is still one
    run costing the sum of them, so singling it out visually would invite the
    reader to discount a bar that is exactly as real as its neighbours. The
    session count stays in ``sweep_cost.csv`` for anyone who needs it.
    """
    if runs.empty:
        return []
    labels = {question.var: question.label for question in resolve_questions(None)}
    order = [var for var in labels if var in set(runs["question"])]
    cultures = sorted(set(runs["culture"]))
    width = 0.8 / max(len(cultures), 1)
    positions = np.arange(len(order))
    ceiling = float(runs["hours"].max())
    with _styled():
        figure, axis = plt.subplots(figsize=(11, 4.8))
        for index, culture in enumerate(cultures):
            offset = (index - (len(cultures) - 1) / 2) * width
            rows = runs[runs["culture"] == culture].set_index("question")
            for slot, var in enumerate(order):
                if var not in rows.index:
                    continue
                height = float(rows.loc[var, "hours"])
                axis.bar(
                    positions[slot] + offset,
                    height,
                    width,
                    label=culture if slot == 0 else None,
                    color=_series_color(culture),
                    edgecolor="white",
                    linewidth=0.6,
                )
                axis.text(
                    positions[slot] + offset,
                    height + ceiling * 0.025,
                    f"{height:.0f}h",
                    ha="center",
                    fontsize=10,
                    color="#333333",
                )
        axis.set_xticks(positions, [labels[var] for var in order])
        axis.set_ylabel("GPU hours per run")
        axis.set_ylim(0, ceiling * 1.18)
        axis.yaxis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, ncol=3, fontsize=11, loc="upper left")
        axis.set_title(
            "Inference cost per run, one card, identical prompt counts",
            fontsize=13,
            loc="left",
            pad=12,
        )
        figure.tight_layout()
        return _save(figure, destination / "fig_sweep_cost")


def _write(frame: pd.DataFrame, destination: Path) -> Path | None:
    """Write one table, creating its directory, skipping an empty frame."""
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
    """Build the cross-question summary figures and their CSVs, per model.

    Reads only what the sweep already wrote, so it is safe to call while a run is
    still going; a question with no finished run simply contributes no bar.
    """
    outcomes = resolve_questions(questions)
    produced: list[Path] = []
    written: list[Path] = []

    for model in models:
        figures = CULTURE_FIGURES / model.key / "summary"
        outputs = CULTURE_ROOT / model.key / "summary"
        distances = distance_table(model, cultures, outcomes)
        advantage = home_advantage_table(model, cultures, outcomes)
        cost = sweep_cost_table(model, cultures, outcomes)
        drawn = [question for question in outcomes if question.var in set(distances["question"])]
        if distances.empty or not drawn:
            continue
        figures.mkdir(parents=True, exist_ok=True)
        produced.extend(overall_figure(distances, drawn, cultures, figures))
        produced.extend(compression_figure(distances, drawn, cultures, figures))
        if not advantage.empty:
            shown = [
                question for question in outcomes if question.var in set(advantage["question"])
            ]
            produced.extend(home_advantage_figure(advantage, shown, cultures, figures))
        produced.extend(sweep_cost_figure(cost, figures))

        for frame, name in (
            (distances, "culture_summary.csv"),
            (advantage, "culture_home_advantage.csv"),
            (cost, "sweep_cost.csv"),
        ):
            path = _write(frame, outputs / name)
            if path is not None:
                written.append(path)

    return {
        "questions": [question.var for question in outcomes],
        "figures": [str(path) for path in produced],
        "tables": [str(path) for path in written],
    }
