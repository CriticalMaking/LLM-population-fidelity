from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from machine_bias_reproduction.data import PreparedData
from machine_bias_reproduction.figures import classical_mds, mds_block, save_plate
from machine_bias_reproduction.metrics import pairwise_nemd_matrix
from machine_bias_reproduction.questions import Question, resolve_question, resolve_questions

from .matching import WVS_COUNTRIES, countries_for, restrict_to
from .mds import FloatArray, _draw_panel, _share_scale, _shared_names, culture_mds
from .palette import MDS_REFERENCES, MODES
from .population import load_run
from .registry import (
    BASE_ARM,
    CULTURE_FIGURES,
    CULTURE_MODELS,
    CULTURE_ROOT,
    SIZE_TIERS,
    CultureModel,
    is_served,
    resolve_cultures,
    resolve_models,
    run_slug,
    tier_models,
)

GROUPS_TABLE = CULTURE_ROOT / "culture_mds_groups.csv"

FINETUNED_CULTURE = "german"

FULL_ANSWERS: tuple[str, ...] = ("FA",)

GRID_COLUMNS = 3

Member = tuple[CultureModel, dict[str, PreparedData]]

Placed = dict[tuple[str, str, str], FloatArray]

Joint = tuple[pd.Index, FloatArray, Placed]


def open_models() -> list[CultureModel]:
    return [model for model in CULTURE_MODELS.values() if not is_served(model.key)]


def _references(question: Question) -> dict[str, PreparedData]:
    loaded: dict[str, PreparedData] = {}
    for label, source in MDS_REFERENCES:
        prepared = load_run(source, None, None, question)
        if prepared is not None:
            loaded[label] = prepared
    return loaded


def _drawable(prepared: PreparedData, modes: Sequence[str]) -> bool:
    return any(mode.lower() in prepared.modes() for mode in modes)


def _members(
    models: Sequence[CultureModel],
    arms: Sequence[str],
    question: Question,
    modes: Sequence[str],
) -> list[Member]:
    found: list[Member] = []
    for model in models:
        loaded: dict[str, PreparedData] = {}
        for arm in arms:
            prepared = load_run(run_slug(model.key, arm), model.key, arm, question)
            if prepared is not None and _drawable(prepared, modes):
                loaded[arm] = prepared
        if len(loaded) == len(arms):
            found.append((model, loaded))
    return found


def _every_series(
    references: dict[str, PreparedData],
    members: Sequence[Member],
) -> dict[str, PreparedData]:
    everything = dict(references)
    for model, arms in members:
        for arm, prepared in arms.items():
            everything[f"{model.key}/{arm}"] = prepared
    return everything


def _sample_names(
    everything: dict[str, PreparedData],
    *,
    matched: bool,
) -> pd.Index | None:
    shared = _shared_names(everything)
    if shared is None:
        return None
    if matched:
        frame = pd.DataFrame({"subpopulation": shared})
        shared = pd.Index(restrict_to(frame, countries_for(FINETUNED_CULTURE))["subpopulation"])
    return None if shared.empty else shared


def _joint_embedding(
    references: dict[str, PreparedData],
    members: Sequence[Member],
    question: Question,
    modes: Sequence[str],
    *,
    matched: bool,
) -> Joint | None:
    everything = _every_series(references, members)
    names = _sample_names(everything, matched=matched)
    if names is None:
        return None

    columns = list(question.answer_columns)
    order: list[tuple[str, str, str]] = []
    blocks: list[FloatArray] = []
    owned: list[tuple[str, str, PreparedData]] = [
        ("", label, prepared) for label, prepared in references.items()
    ]
    owned.extend(
        (model.key, arm, prepared) for model, arms in members for arm, prepared in arms.items()
    )
    for owner, label, prepared in owned:
        for mode in modes:
            if mode.lower() not in prepared.modes():
                continue
            order.append((owner, label, mode))
            frame = prepared.props(mode.lower()).loc[names, columns]
            blocks.append(np.asarray(frame.to_numpy(), dtype=np.float64))
    if not order:
        return None

    survey = next(iter(everything.values())).wvs_props.loc[names, columns]
    stacked = np.vstack([np.asarray(survey.to_numpy(), dtype=np.float64), *blocks])
    coordinates = classical_mds(pairwise_nemd_matrix(stacked))

    count = len(names)
    placed = {key: mds_block(coordinates, 1 + index, count) for index, key in enumerate(order)}
    return names, coordinates[:count], placed


def _panel_coordinates(placed: Placed, model_key: str) -> dict[tuple[str, str], FloatArray]:
    return {
        (label, mode): block
        for (owner, label, mode), block in placed.items()
        if owner in ("", model_key)
    }


def _rows(
    errors: list[tuple[str, float]],
    references: dict[str, PreparedData],
    model: CultureModel,
    *,
    plate: str,
    question: Question,
    mode: str,
    sample: str,
    countries: str,
    subpopulations: int,
) -> list[dict[str, Any]]:
    return [
        {
            "plate": plate,
            "question": question.var,
            "model_key": None if label in references else model.key,
            "model_label": None if label in references else model.label,
            "series": label,
            "mode": mode,
            "sample": sample,
            "countries": countries,
            "subpopulations": subpopulations,
            "mean_nEMD": error,
        }
        for label, error in errors
    ]


Cell = tuple[str, Member] | None


def _layout(
    members: Sequence[Member],
    modes: Sequence[str],
    wrap: int | None,
) -> tuple[int, int, list[list[Cell]]]:
    if wrap is None or len(members) <= wrap:
        return len(modes), len(members), [[(mode, member) for member in members] for mode in modes]
    placement: list[list[Cell]] = []
    for mode in modes:
        for start in range(0, len(members), wrap):
            band: list[Cell] = [(mode, member) for member in members[start : start + wrap]]
            placement.append(band + [None] * (wrap - len(band)))
    return len(placement), wrap, placement


def _grid_plate(
    models: Sequence[CultureModel],
    arms: Sequence[str],
    question: Question,
    modes: Sequence[str],
    *,
    plate: str,
    stem: str,
    matched: bool,
    wrap: int | None = None,
) -> tuple[list[Path], pd.DataFrame]:
    references = _references(question)
    members = _members(models, arms, question, modes)
    if not members:
        return [], pd.DataFrame()
    embedded = _joint_embedding(references, members, question, modes, matched=matched)
    if embedded is None:
        return [], pd.DataFrame()
    names, wvs_coordinates, placed = embedded

    sample = "german_matched" if matched else "all_countries"
    countries = countries_for(FINETUNED_CULTURE) if matched else WVS_COUNTRIES
    drawn_over = ", ".join(countries)

    destination = CULTURE_FIGURES / question.var
    destination.mkdir(parents=True, exist_ok=True)

    height, width, placement = _layout(members, modes, wrap)
    wrapped = height > len(modes)
    figure, axes = plt.subplots(
        height, width, figsize=(6.2 * width, 5.8 * height), squeeze=False
    )
    rows: list[dict[str, Any]] = []
    drawn: list[Any] = []
    for index, (row_axes, row_cells) in enumerate(zip(axes, placement, strict=True)):
        for axis, cell in zip(row_axes, row_cells, strict=True):
            if cell is None:
                axis.axis("off")
                continue
            mode, (model, loaded) = cell
            errors = _draw_panel(
                axis,
                {**references, **loaded},
                _panel_coordinates(placed, model.key),
                mode,
                names,
                question,
                wvs_coordinates,
                sample=drawn_over,
                hollow=[BASE_ARM],
            )
            rows.extend(
                _rows(
                    errors,
                    references,
                    model,
                    plate=plate,
                    question=question,
                    mode=mode,
                    sample=sample,
                    countries="; ".join(countries),
                    subpopulations=len(names),
                )
            )
            if wrapped or index == 0:
                axis.set_title(model.label, pad=26)
            drawn.append(axis)
        row_axes[0].set_ylabel("Classical MDS dimension 2")

    _share_scale(drawn, wvs_coordinates, placed)
    drawn[0].legend(loc="upper left", markerscale=1.4)
    figure.tight_layout()

    table = pd.DataFrame(rows).drop_duplicates(
        subset=["plate", "question", "model_key", "series", "mode"], ignore_index=True
    )
    return save_plate(figure, destination / stem), table


def tier_plate(tier: str, question: str | Question) -> tuple[list[Path], pd.DataFrame]:
    return _grid_plate(
        tier_models(tier),
        (BASE_ARM, FINETUNED_CULTURE),
        resolve_question(question),
        MODES,
        plate=f"tier_{tier}",
        stem=f"fig_culture_mds_tier_{tier}",
        matched=True,
    )


def base_families_plate(question: str | Question) -> tuple[list[Path], pd.DataFrame]:
    return _grid_plate(
        list(CULTURE_MODELS.values()),
        (BASE_ARM,),
        resolve_question(question),
        FULL_ANSWERS,
        plate="base_families",
        stem="fig_culture_mds_base_families",
        matched=False,
        wrap=GRID_COLUMNS,
    )


def finetuned_plate(question: str | Question) -> tuple[list[Path], pd.DataFrame]:
    return _grid_plate(
        open_models(),
        (FINETUNED_CULTURE,),
        resolve_question(question),
        MODES,
        plate="finetuned",
        stem="fig_culture_mds_finetuned",
        matched=True,
        wrap=GRID_COLUMNS,
    )


def _collect(drawn: list[tuple[list[Path], pd.DataFrame]]) -> tuple[list[Path], pd.DataFrame]:
    files = [path for paths, _ in drawn for path in paths]
    tables = [table for _, table in drawn if not table.empty]
    return files, pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def tier_plates(questions: Sequence[str] | None = None) -> tuple[list[Path], pd.DataFrame]:
    return _collect(
        [
            tier_plate(tier, question)
            for question in resolve_questions(questions)
            for tier in SIZE_TIERS
        ]
    )


def base_families_plates(
    questions: Sequence[str] | None = None,
) -> tuple[list[Path], pd.DataFrame]:
    return _collect([base_families_plate(question) for question in resolve_questions(questions)])


def finetuned_plates(questions: Sequence[str] | None = None) -> tuple[list[Path], pd.DataFrame]:
    return _collect([finetuned_plate(question) for question in resolve_questions(questions)])


def refresh_model_plates(questions: Sequence[str] | None = None) -> list[str]:
    resolved = resolve_questions(questions)
    models = resolve_models(None)
    cultures = resolve_cultures(None)
    produced: list[str] = []
    for question in resolved:
        result = culture_mds(
            models,
            cultures,
            question,
            all_countries=True,
            outcomes=question is resolved[0],
        )
        produced.extend(result["figures"])
    return produced
