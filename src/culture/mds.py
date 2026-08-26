from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib import pyplot as plt

from machine_bias_reproduction.analysis import load_source
from machine_bias_reproduction.config import paths_for
from machine_bias_reproduction.data import PreparedData, group_responses
from machine_bias_reproduction.figures import classical_mds, mds_block, save_plate
from machine_bias_reproduction.io_utils import write_csv
from machine_bias_reproduction.metrics import nemd, pairwise_nemd_matrix
from machine_bias_reproduction.plates import MUTED_INK, SEPARATOR, SURVEY_TONE
from machine_bias_reproduction.questions import Question, resolve_question

from .matching import (
    MATCHED_CULTURES,
    WVS_COUNTRIES,
    countries_for,
    country_of,
    restrict_to,
)
from .palette import (
    MDS_REFERENCES,
    MODES,
    arm_marker,
    is_reference,
    mds_arm_tone,
)
from .registry import BASE_ARM, CULTURE_FIGURES, CULTURE_ROOT, CultureModel, csv_stems

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

Embedding = tuple[pd.Index, FloatArray, dict[tuple[str, str], FloatArray]]

OUTCOME_ROW: tuple[str, str, str] = ("d_happy", "d_polpos", "d_religiousp")


def _series(
    model: CultureModel,
    culture: str,
    question: Question,
) -> dict[str, PreparedData]:
    wanted: list[tuple[str, str, tuple[str, str] | None]] = [
        (label, source, None) for label, source in MDS_REFERENCES
    ]
    wanted.append(
        (culture, f"culture/{model.key}/{culture}", csv_stems(model.key, culture, question))
    )
    loaded: dict[str, PreparedData] = {}
    for label, source, stems in wanted:
        try:
            loaded[label] = load_source(source, paths_for(source, question.var), question, stems)
        except (FileNotFoundError, ValueError):
            continue
    return loaded


def _embed(
    series: dict[str, PreparedData],
    names: pd.Index,
    question: Question,
) -> Embedding | None:
    if names.empty:
        return None
    columns = list(question.answer_columns)
    order = [(label, mode) for label in series for mode in MODES]
    props = {
        (label, mode): series[label].props(mode.lower()).loc[names, columns]
        for label, mode in order
    }
    wvs = next(iter(series.values())).wvs_props.loc[names, columns].to_numpy(dtype=np.float64)

    blocks = [wvs] * len(order)
    blocks.extend(props[key].to_numpy(dtype=np.float64) for key in order)
    coordinates = classical_mds(pairwise_nemd_matrix(np.vstack(blocks)))

    count = len(names)
    model_coordinates = {
        key: mds_block(coordinates, len(order) + index, count) for index, key in enumerate(order)
    }
    return names, coordinates[:count], model_coordinates


def _shared_names(series: dict[str, PreparedData]) -> pd.Index | None:
    shared: pd.Index | None = None
    for prepared in series.values():
        shared = prepared.names if shared is None else shared.intersection(prepared.names)
    return None if shared is None or shared.empty else shared


def _matched_mds_embedding(
    series: dict[str, PreparedData],
    culture: str,
    question: Question,
) -> Embedding | None:
    shared = _shared_names(series)
    if shared is None:
        return None
    frame = pd.DataFrame({"subpopulation": shared})
    matched = restrict_to(frame, countries_for(culture))
    return _embed(series, pd.Index(matched["subpopulation"]), question)


def _all_countries_mds_embedding(
    series: dict[str, PreparedData],
    question: Question,
) -> Embedding | None:
    shared = _shared_names(series)
    return None if shared is None else _embed(series, shared, question)


def _home_mask(names: pd.Index, culture: str) -> BoolArray:
    countries = countries_for(culture)
    mask = country_of(pd.Series(names, dtype="object")).isin(list(countries))
    return np.asarray(mask.to_numpy(), dtype=bool)


def home_countries(culture: str) -> str:
    return ", ".join(countries_for(culture))


def _draw_panel(
    axis: Any,
    series: dict[str, PreparedData],
    model_coordinates: dict[tuple[str, str], FloatArray],
    mode: str,
    names: pd.Index,
    question: Question,
    wvs_coordinates: FloatArray,
    *,
    sample: str,
    home: BoolArray | None = None,
    home_countries: str = "",
    only: Sequence[str] | None = None,
    hollow: Sequence[str] | None = None,
) -> list[tuple[str, float]]:
    columns = list(question.answer_columns)
    wvs_props = next(iter(series.values())).wvs_props.loc[names, columns]
    axis.scatter(
        wvs_coordinates[:, 0],
        wvs_coordinates[:, 1],
        s=22,
        alpha=0.6,
        marker="^",
        label=f"WVS — {sample}",
        color=SURVEY_TONE.fill,
        edgecolors=SURVEY_TONE.ink,
        linewidths=0.3,
    )
    errors: list[tuple[str, float]] = []
    for label in series:
        if only is not None and label not in only:
            continue
        coordinates = model_coordinates.get((label, mode))
        if coordinates is None:
            continue
        model_props = series[label].props(mode.lower()).loc[names, columns]
        error = float(np.mean(nemd(wvs_props.to_numpy(), model_props.to_numpy())))
        tuned = not is_reference(label)
        tone = mds_arm_tone(label)
        if hollow is not None and label in hollow:
            axis.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                s=30,
                alpha=0.8,
                marker=arm_marker(label),
                label=label,
                facecolors="none",
                edgecolors=tone.ink,
                linewidths=1.0,
                zorder=3,
            )
            errors.append((label, error))
            continue
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=28 if tuned else 22,
            alpha=0.75 if tuned else 0.5,
            marker=arm_marker(label),
            label=label,
            color=tone.fill if tuned else tone.ink,
            edgecolors=tone.ink if tuned else SEPARATOR,
            linewidths=0.5 if tuned else 0.0,
            zorder=3 if tuned else 2,
        )
        if tuned and home is not None and home.any():
            axis.scatter(
                coordinates[home, 0],
                coordinates[home, 1],
                s=58,
                facecolors="none",
                marker="o",
                edgecolors=tone.ink,
                linewidths=1.1,
                zorder=4,
                label=f"own countries — {home_countries}",
            )
        errors.append((label, error))
    axis.set_aspect("equal")
    axis.set_xlabel("Classical MDS dimension 1")
    axis.annotate(
        f"{mode} · {len(names)} subpopulations · mean nEMD "
        + ", ".join(f"{label} {error:.3f}" for label, error in errors),
        xy=(0.0, 1.01),
        xycoords="axes fraction",
        fontsize=8.5,
        color=MUTED_INK,
        va="bottom",
    )
    return errors


def _ntp_props_over_every_subpopulation(
    prepared: PreparedData,
    question: Question,
) -> pd.DataFrame:
    ntp_raw = prepared.ntp_raw
    if ntp_raw is None:
        raise ValueError("no ntp data in this run")
    by_profile = question.ntp_answers(ntp_raw.set_index("profile"))
    matched = by_profile.reindex(prepared.wvs["profile"]).reset_index(drop=True)
    return group_responses(matched, prepared.subpops["subpop"])


def _ntp_only_series(
    series: dict[str, PreparedData],
    question: Question,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame] | None:
    props = {
        label: _ntp_props_over_every_subpopulation(prepared, question)
        for label, prepared in series.items()
    }
    shared: pd.Index | None = None
    for frame in props.values():
        usable = frame.index[frame.notna().all(axis=1).to_numpy()]
        shared = usable if shared is None else shared.intersection(usable)
    if shared is None or shared.empty:
        return None
    reference = max(series.values(), key=lambda prepared: prepared.names.size)
    wvs = reference.wvs_props.reindex(shared)
    shared = shared[wvs.notna().all(axis=1).to_numpy()]
    if shared.empty:
        return None
    return {label: frame.loc[shared] for label, frame in props.items()}, wvs.loc[shared]


def _embed_ntp_only(
    props: dict[str, pd.DataFrame],
    wvs_props: pd.DataFrame,
    question: Question,
) -> Embedding:
    columns = list(question.answer_columns)
    order = [(label, "NTP") for label in props]
    wvs = wvs_props.loc[:, columns].to_numpy(dtype=np.float64)

    blocks = [wvs] * len(order)
    blocks.extend(props[label].loc[:, columns].to_numpy(dtype=np.float64) for label, _ in order)
    coordinates = classical_mds(pairwise_nemd_matrix(np.vstack(blocks)))

    count = len(wvs_props.index)
    model_coordinates = {
        key: mds_block(coordinates, len(order) + index, count) for index, key in enumerate(order)
    }
    return wvs_props.index, coordinates[:count], model_coordinates


def _share_scale(
    axes: Any,
    wvs_coordinates: FloatArray,
    model_coordinates: Mapping[Any, FloatArray],
) -> None:
    spread = np.vstack([wvs_coordinates, *model_coordinates.values()])
    half = max(float(np.ptp(spread[:, 0])), float(np.ptp(spread[:, 1]))) / 2 * 1.05
    for axis in axes:
        for limit, values in ((axis.set_xlim, spread[:, 0]), (axis.set_ylim, spread[:, 1])):
            middle = (float(values.min()) + float(values.max())) / 2
            limit(middle - half, middle + half)


def _rows(
    errors: list[tuple[str, float]],
    *,
    culture: str,
    question: Question,
    mode: str,
    sample: str,
    countries: str,
    subpopulations: int,
) -> list[dict[str, Any]]:
    return [
        {
            "culture": culture,
            "question": question.var,
            "sample": sample,
            "series": label,
            "mode": mode,
            "subpopulations": subpopulations,
            "countries": countries,
            "mean_nEMD": error,
        }
        for label, error in errors
    ]


def _plate_series(model: CultureModel, culture: str, question: Question) -> dict[str, PreparedData]:
    if culture not in MATCHED_CULTURES:
        return {}
    series = _series(model, culture, question)
    return series if culture in series and len(series) >= 2 else {}


def _sample_plate(
    model: CultureModel,
    culture: str,
    question: Question,
    *,
    all_countries: bool,
) -> tuple[list[Path], pd.DataFrame]:
    series = _plate_series(model, culture, question)
    if not series:
        return [], pd.DataFrame()
    embedded = (
        _all_countries_mds_embedding(series, question)
        if all_countries
        else _matched_mds_embedding(series, culture, question)
    )
    if embedded is None:
        return [], pd.DataFrame()
    names, wvs_coordinates, model_coordinates = embedded
    countries: tuple[str, ...]
    if all_countries:
        home = _home_mask(names, culture)
        sample_label = f"all {len(WVS_COUNTRIES)} WVS countries"
        sample = "all_countries"
        countries = WVS_COUNTRIES
        stem = "fig_culture_mds_all_countries"
    else:
        home = None
        sample_label = home_countries(culture)
        sample = "matched"
        countries = countries_for(culture)
        stem = "fig_culture_mds_matched"

    destination = CULTURE_FIGURES / model.key / culture / question.var
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    figure, axes = plt.subplots(1, len(MODES), figsize=(13, 5.4))
    for axis, mode in zip(axes, MODES, strict=True):
        errors = _draw_panel(
            axis,
            series,
            model_coordinates,
            mode,
            names,
            question,
            wvs_coordinates,
            sample=sample_label,
            home=home,
            home_countries=home_countries(culture),
        )
        rows.extend(
            _rows(
                errors,
                culture=culture,
                question=question,
                mode=mode,
                sample=sample,
                countries="; ".join(countries),
                subpopulations=len(names),
            )
        )

    _share_scale(axes, wvs_coordinates, model_coordinates)
    axes[0].set_ylabel("Classical MDS dimension 2")
    axes[0].legend(loc="upper left", markerscale=1.6)
    figure.tight_layout()
    return save_plate(figure, destination / stem), pd.DataFrame(rows)


def _comparison_series(
    model: CultureModel,
    culture: str,
    question: Question,
) -> dict[str, PreparedData]:
    series = _plate_series(model, culture, question)
    if not series or BASE_ARM in series:
        return {}
    source = f"culture/{model.key}/{BASE_ARM}"
    try:
        base = load_source(
            source,
            paths_for(source, question.var),
            question,
            csv_stems(model.key, BASE_ARM, question),
        )
    except (FileNotFoundError, ValueError):
        return {}
    return {BASE_ARM: base, **series}


def _comparison_plate(
    model: CultureModel,
    culture: str,
    question: Question,
) -> tuple[list[Path], pd.DataFrame]:
    series = _comparison_series(model, culture, question)
    if not series:
        return [], pd.DataFrame()
    embedded = _matched_mds_embedding(series, culture, question)
    if embedded is None:
        return [], pd.DataFrame()
    names, wvs_coordinates, model_coordinates = embedded

    arms = [BASE_ARM, culture]
    references = [label for label, _ in MDS_REFERENCES]
    destination = CULTURE_FIGURES / model.key / culture / question.var
    destination.mkdir(parents=True, exist_ok=True)

    measured: dict[tuple[str, str], float] = {}
    figure, axes = plt.subplots(
        len(MODES),
        len(arms),
        figsize=(6.5 * len(arms), 5.4 * len(MODES)),
        squeeze=False,
    )
    for row, mode in zip(axes, MODES, strict=True):
        for axis, arm in zip(row, arms, strict=True):
            errors = _draw_panel(
                axis,
                series,
                model_coordinates,
                mode,
                names,
                question,
                wvs_coordinates,
                sample=home_countries(culture),
                only=[*references, arm],
            )
            measured.update({(mode, label): error for label, error in errors})
        row[0].set_ylabel("Classical MDS dimension 2")

    _share_scale([axis for row in axes for axis in row], wvs_coordinates, model_coordinates)
    for axis in axes[0]:
        axis.legend(loc="upper left", markerscale=1.6)
    figure.tight_layout()

    rows: list[dict[str, Any]] = []
    for mode in MODES:
        rows.extend(
            _rows(
                [(label, error) for (drawn, label), error in measured.items() if drawn == mode],
                culture=culture,
                question=question,
                mode=mode,
                sample="comparison",
                countries="; ".join(countries_for(culture)),
                subpopulations=len(names),
            )
        )
    return save_plate(figure, destination / "fig_culture_mds_comparison"), pd.DataFrame(rows)


def _outcomes_row(
    model: CultureModel,
    culture: str,
    outcomes: tuple[str, ...] = OUTCOME_ROW,
) -> tuple[list[Path], pd.DataFrame]:
    destination = CULTURE_FIGURES / model.key / culture
    panels: list[tuple[Question, dict[str, PreparedData], Embedding]] = []
    for name in outcomes:
        question = resolve_question(name)
        series = _plate_series(model, culture, question)
        if not series:
            continue
        embedded = _matched_mds_embedding(series, culture, question)
        if embedded is not None:
            panels.append((question, series, embedded))
    if not panels:
        return [], pd.DataFrame()

    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    figure, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 5.6), squeeze=False)
    for axis, (question, series, embedded) in zip(axes[0], panels, strict=True):
        names, wvs_coordinates, model_coordinates = embedded
        errors = _draw_panel(
            axis,
            series,
            model_coordinates,
            "NTP",
            names,
            question,
            wvs_coordinates,
            sample=home_countries(culture),
        )
        _share_scale([axis], wvs_coordinates, model_coordinates)
        rows.extend(
            _rows(
                errors,
                culture=culture,
                question=question,
                mode="NTP",
                sample="matched",
                countries="; ".join(countries_for(culture)),
                subpopulations=len(names),
            )
        )
    axes[0][0].set_ylabel("Classical MDS dimension 2")
    axes[0][0].legend(loc="upper left", markerscale=1.6)
    figure.tight_layout()
    return save_plate(figure, destination / "fig_culture_mds_outcomes"), pd.DataFrame(rows)


def degenerate_ntp_plate(
    model: CultureModel,
    culture: str,
    question: str | Question,
    *,
    valid_answer_mass: float | None = None,
    distinct_answers: int | None = None,
) -> list[Path]:
    outcome = resolve_question(question)
    series = _series(model, culture, outcome)
    if culture not in series:
        return []
    prepared = _ntp_only_series(series, outcome)
    if prepared is None:
        return []
    props, wvs_props = prepared
    placed, wvs_coordinates, model_coordinates = _embed_ntp_only(props, wvs_props, outcome)

    destination = CULTURE_FIGURES / model.key / culture / outcome.var
    destination.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7.5, 6))
    axis.scatter(
        wvs_coordinates[:, 0],
        wvs_coordinates[:, 1],
        s=22,
        alpha=0.6,
        marker="^",
        label="WVS",
        color=SURVEY_TONE.fill,
        edgecolors=SURVEY_TONE.ink,
        linewidths=0.3,
    )
    for label in series:
        coordinates = model_coordinates.get((label, "NTP"))
        if coordinates is None:
            continue
        degenerate = label == culture
        tone = mds_arm_tone(label)
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=34 if degenerate else 22,
            alpha=0.75 if degenerate else 0.5,
            marker=arm_marker(label),
            label=f"{label} (no valid answers)" if degenerate else label,
            facecolors="none" if degenerate else tone.fill,
            edgecolors=tone.ink,
            linewidths=1.0 if degenerate else 0.4,
            zorder=3 if degenerate else 2,
        )
    axis.set_aspect("equal")
    axis.set_xlabel("Classical MDS dimension 1")
    axis.set_ylabel("Classical MDS dimension 2")
    axis.legend(loc="upper left", markerscale=1.4)

    measured = [f"NTP only · {len(placed)} subpopulations"]
    if valid_answer_mass is not None:
        measured.append(f"valid-answer mass {valid_answer_mass:.3g}")
    if distinct_answers is not None:
        measured.append(f"{distinct_answers} distinct answers")
    axis.annotate(
        "  ·  ".join(measured),
        xy=(0.0, 1.01),
        xycoords="axes fraction",
        fontsize=8.5,
        color=MUTED_INK,
        va="bottom",
    )
    figure.tight_layout()
    return save_plate(figure, destination / "fig_culture_mds_ntp_degenerate")


def _export(tables: list[pd.DataFrame], destination: Path) -> Path | None:
    if not tables:
        return None
    write_csv(pd.concat(tables, ignore_index=True), destination)
    return destination


def culture_mds(
    models: list[CultureModel],
    cultures: list[str],
    question: str | Question = "d_happy",
    *,
    all_countries: bool = False,
    outcomes: bool = False,
) -> dict[str, Any]:
    outcome = resolve_question(question)
    produced: list[Path] = []
    per_question: list[pd.DataFrame] = []
    per_culture: list[pd.DataFrame] = []

    def collect(
        target: list[pd.DataFrame],
        drawn: tuple[list[Path], pd.DataFrame],
        model: CultureModel,
    ) -> None:
        figures, table = drawn
        produced.extend(figures)
        if not table.empty:
            table.insert(0, "model_key", model.key)
            table.insert(1, "model_label", model.label)
            target.append(table)

    for model in models:
        for culture in cultures:
            collect(
                per_question, _sample_plate(model, culture, outcome, all_countries=False), model
            )
            collect(per_question, _comparison_plate(model, culture, outcome), model)
            if all_countries:
                collect(
                    per_question, _sample_plate(model, culture, outcome, all_countries=True), model
                )
            if outcomes:
                collect(per_culture, _outcomes_row(model, culture), model)

    exports = [
        _export(per_question, CULTURE_ROOT / outcome.var / "culture_mds_matched.csv"),
        *(
            _export(
                [frame for frame in per_culture if frame["model_key"].iloc[0] == model.key],
                CULTURE_ROOT / model.key / "culture_mds_outcomes.csv",
            )
            for model in models
        ),
    ]
    return {
        "question": outcome.var,
        "figures": [str(path) for path in produced],
        "tables": [str(path) for path in exports if path is not None],
    }
