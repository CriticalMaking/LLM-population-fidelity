from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd

from machine_bias_reproduction.analysis import _load_source
from machine_bias_reproduction.config import paths_for
from machine_bias_reproduction.data import PreparedData
from machine_bias_reproduction.figures import MDS_COLORS, _classical_mds, _save
from machine_bias_reproduction.metrics import nemd, pairwise_nemd_matrix
from machine_bias_reproduction.questions import Question, resolve_question

from .matching import (
    MATCHED_CULTURES,
    WVS_COUNTRIES,
    countries_for,
    country_of,
    restrict,
)
from .palette import (
    MDS_REFERENCES,
    MDS_SERIES_MARKERS,
    MODE_TITLES,
    MODES,
    is_reference,
    mds_color,
)
from .registry import CULTURE_FIGURES, CULTURE_ROOT, CultureModel, csv_stems

matplotlib.use("Agg")
from matplotlib import pyplot as plt

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
            loaded[label] = _load_source(source, paths_for(source, question.var), question, stems)
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
        (label, mode): (series[label].ntp_props if mode == "NTP" else series[label].fa_props).loc[
            names, columns
        ]
        for label, mode in order
    }
    wvs = next(iter(series.values())).wvs_props.loc[names, columns].to_numpy(dtype=np.float64)

    blocks = [wvs] * len(order)
    blocks.extend(props[key].to_numpy(dtype=np.float64) for key in order)
    coordinates = _classical_mds(pairwise_nemd_matrix(np.vstack(blocks)))

    count = len(names)
    model_coordinates = {
        key: coordinates[(len(order) + index) * count : (len(order) + index + 1) * count]
        for index, key in enumerate(order)
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
    return _embed(series, pd.Index(restrict(frame, culture)["subpopulation"]), question)


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
) -> list[tuple[str, float]]:
    columns = list(question.answer_columns)
    wvs_props = next(iter(series.values())).wvs_props.loc[names, columns]
    axis.scatter(
        wvs_coordinates[:, 0],
        wvs_coordinates[:, 1],
        s=22,
        alpha=0.35,
        marker="^",
        label=f"WVS respondents — {sample}",
        color=MDS_COLORS["WVS"],
        edgecolors="#8a8a2a",
        linewidths=0.3,
    )
    errors: list[tuple[str, float]] = []
    for label in series:
        coordinates = model_coordinates.get((label, mode))
        if coordinates is None:
            continue
        model_props = (series[label].ntp_props if mode == "NTP" else series[label].fa_props).loc[
            names, columns
        ]
        error = float(np.mean(nemd(wvs_props.to_numpy(), model_props.to_numpy())))
        tuned = not is_reference(label)
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=28 if tuned else 22,
            alpha=0.6 if tuned else 0.42,
            marker=MDS_SERIES_MARKERS.get(label, "o"),
            # "spanish-language", not "spanish": the fine-tune is on a language,
            # while the respondents it is scored against are a country. Naming
            # the language is what stops the pairing reading as "Spain".
            label=f"{label}-language finetuned culture MLLM" if tuned else label,
            color=mds_color(label),
            edgecolors="white" if tuned else "none",
            linewidths=0.4 if tuned else 0.0,
            zorder=3 if tuned else 2,
        )
        if tuned and home is not None and home.any():
            # The home subset is the matched plate's entire sample. Ringing it
            # here is what lets the two figures be read against each other.
            axis.scatter(
                coordinates[home, 0],
                coordinates[home, 1],
                s=58,
                facecolors="none",
                marker="o",
                edgecolors=mds_color(label),
                linewidths=1.1,
                zorder=4,
                label=f"its own countries: {home_countries}",
            )
        errors.append((label, error))
    axis.set_aspect("equal")
    axis.set_xlabel("Classical MDS dimension 1")
    # Say the sample out loud: these means are over the panel's own
    # subpopulations and are not the run's overall nEMD.
    axis.annotate(
        f"Mean nEMD over these {len(names)} subpopulations\n"
        + "  ·  ".join(f"{label} {error:.3f}" for label, error in errors),
        xy=(0.03, 0.03),
        xycoords="axes fraction",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#666666"},
    )
    return errors


def _share_scale(
    axes: Any,
    wvs_coordinates: FloatArray,
    model_coordinates: dict[tuple[str, str], FloatArray],
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


def _matched_plate(
    model: CultureModel,
    culture: str,
    question: Question,
) -> tuple[list[Path], pd.DataFrame]:
    series = _plate_series(model, culture, question)
    if not series:
        return [], pd.DataFrame()
    embedded = _matched_mds_embedding(series, culture, question)
    if embedded is None:
        return [], pd.DataFrame()
    names, wvs_coordinates, model_coordinates = embedded

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
            sample=home_countries(culture),
        )
        axis.set_title(MODE_TITLES[mode], fontsize=12)
        rows.extend(
            _rows(
                errors,
                culture=culture,
                question=question,
                mode=mode,
                sample="matched",
                countries="; ".join(countries_for(culture)),
                subpopulations=len(names),
            )
        )

    _share_scale(axes, wvs_coordinates, model_coordinates)
    axes[0].set_ylabel("Classical MDS dimension 2")
    axes[0].legend(frameon=False, loc="upper left", fontsize=9, markerscale=1.6)
    # Bare plate: no figure title. Which culture and which question this is
    # belongs to the caption; which respondents it covers is in the legend,
    # because that is the thing a reader assumes wrongly.
    figure.tight_layout()
    return _save(figure, destination / "fig_culture_mds_matched"), pd.DataFrame(rows)


def _all_countries_plate(
    model: CultureModel,
    culture: str,
    question: Question,
) -> tuple[list[Path], pd.DataFrame]:
    series = _plate_series(model, culture, question)
    if not series:
        return [], pd.DataFrame()
    embedded = _all_countries_mds_embedding(series, question)
    if embedded is None:
        return [], pd.DataFrame()
    names, wvs_coordinates, model_coordinates = embedded
    home = _home_mask(names, culture)

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
            sample=f"all {len(WVS_COUNTRIES)} WVS countries",
            home=home,
            home_countries=home_countries(culture),
        )
        axis.set_title(MODE_TITLES[mode], fontsize=12)
        rows.extend(
            _rows(
                errors,
                culture=culture,
                question=question,
                mode=mode,
                sample="all_countries",
                countries="; ".join(WVS_COUNTRIES),
                subpopulations=len(names),
            )
        )

    _share_scale(axes, wvs_coordinates, model_coordinates)
    axes[0].set_ylabel("Classical MDS dimension 2")
    axes[0].legend(frameon=False, loc="upper left", fontsize=9, markerscale=1.6)
    figure.tight_layout()
    return _save(figure, destination / "fig_culture_mds_all_countries"), pd.DataFrame(rows)


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
        axis.set_title(question.label, fontsize=13)
        # Each panel is its own embedding, so each gets its own scale. Sharing
        # one here would imply a distance between outcomes that does not exist.
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
    axes[0][0].legend(frameon=False, loc="upper left", fontsize=9, markerscale=1.6)
    figure.tight_layout()
    return _save(figure, destination / "fig_culture_mds_outcomes"), pd.DataFrame(rows)


def _export(tables: list[pd.DataFrame], destination: Path) -> Path | None:
    if not tables:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(destination, index=False)
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
    # Two exports, because the outcomes row spans questions: filing its panels
    # under the question this call was for would put politics and religion rows
    # inside the happiness table.
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
            collect(per_question, _matched_plate(model, culture, outcome), model)
            if all_countries:
                collect(per_question, _all_countries_plate(model, culture, outcome), model)
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
