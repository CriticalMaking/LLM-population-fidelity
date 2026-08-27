from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import spearmanr

from machine_bias_reproduction.config import EXPECTED_SUBPOPULATIONS, OUTPUTS_ROOT, paths_for
from machine_bias_reproduction.data import load_subpops
from machine_bias_reproduction.metrics import nemd, pairwise_nemd
from machine_bias_reproduction.questions import Question, resolve_questions

from .matching import WVS_COUNTRIES
from .population import MODES, load_run, sources
from .registry import BASE_ARM
from .tables import read_csv

FloatArray = npt.NDArray[np.float64]

POPULATION = "population"

EVERY_CELL = "every retained cell"

MISSING_TOKEN = "NA"

MIN_CELLS_FOR_DISPERSION = 2

MIN_CELLS_FOR_STRUCTURE = 3

AGE_BANDS: tuple[str, ...] = ("<25", "25-34", "35-44", "45-54", "55-64", "65-74", "75+")

WAVES: dict[str, str] = {
    "1995": "1995-1997",
    "1996": "1995-1997",
    "1997": "1995-1997",
    "2005": "2005-2006",
    "2006": "2005-2006",
    "2017": "2017-2018",
    "2018": "2017-2018",
}

FAMILIES: tuple[str, ...] = (
    "country",
    "wave",
    "sex",
    "age",
    "education",
    "employment",
    "marital_status",
)

FAMILY_LABELS: dict[str, str] = {
    POPULATION: "Every subpopulation",
    "country": "Country",
    "wave": "Survey wave",
    "sex": "Sex",
    "age": "Age band",
    "education": "Education",
    "employment": "Employment",
    "marital_status": "Marital status",
}

LEVEL_ORDER: dict[str, tuple[str, ...]] = {
    "country": WVS_COUNTRIES,
    "wave": ("1995-1997", "2005-2006", "2017-2018"),
    "sex": ("Female", "Male"),
    "age": AGE_BANDS,
    "education": ("Low", "Middle", "High"),
    "employment": ("Working", "Student", "Homemaker", "Unemployed", "Retired"),
    "marital_status": ("Single", "Cohabiting", "Married", "Divorced or separated", "Widowed"),
}

CELL_PATTERN = re.compile(
    r"^(?P<country>.+?) (?P<year>\d{4}) "
    r"(?P<sex>Male|Female|NA) "
    r"(?P<age><25|25-34|35-44|45-54|55-64|65-74|75\+|NA) "
    r"(?P<education>High|Low|Middle|NA) "
    r"(?P<employment>Homemaker|Retired|Student|Unemployed|Working|NA) "
    r"(?P<marital_status>Cohabiting|Divorced or separated|Married|Single|Widowed|NA)$"
)

QUANTILES: tuple[tuple[str, float], ...] = (
    ("e_q10_nemd", 0.10),
    ("e_q25_nemd", 0.25),
    ("e_median_nemd", 0.50),
    ("e_q75_nemd", 0.75),
    ("e_q90_nemd", 0.90),
)

SCORE_COLUMNS: tuple[str, ...] = (
    "score_accuracy",
    "score_dispersion",
    "score_structure",
    "score_center",
    "pfs",
    "pfs_with_center",
)

DELTA_COLUMNS: tuple[str, ...] = ("pfs", "pfs_with_center", "score_center")

TERMS: tuple[tuple[str, str], ...] = (
    ("score_accuracy", "accuracy"),
    ("score_dispersion", "dispersion"),
    ("score_structure", "structure"),
)

IDENTITY: tuple[str, ...] = (
    "question",
    "question_label",
    "model_key",
    "model_label",
    "arm",
    "series",
    "source",
    "mode",
)

CELLS_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_cells.csv"

GROUPS_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_groups.csv"

OVERALL_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_overall.csv"


def _parse(name: str) -> dict[str, str]:
    matched = CELL_PATTERN.fullmatch(name)
    if matched is None:
        raise ValueError(f"unparsable subpopulation label: {name}")
    return matched.groupdict()


def cell_facets(labels: Iterable[str] | None = None) -> pd.DataFrame:
    known = labels if labels is not None else load_subpops()["subpop"].unique()
    names = pd.Index(sorted(set(known)), name="subpopulation")
    frame = pd.DataFrame([_parse(str(name)) for name in names], index=names)
    frame["wave"] = frame.pop("year").map(WAVES)
    return frame[list(FAMILIES)].replace(MISSING_TOKEN, np.nan)


def levels(family: str, present: pd.Series) -> list[str]:
    seen = set(present.dropna().unique())
    return [level for level in LEVEL_ORDER[family] if level in seen]


def geometric_mean(values: npt.ArrayLike) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or bool(np.isnan(array).any()):
        return float("nan")
    if bool((array <= 0).any()):
        return 0.0
    return float(np.exp(np.log(array).mean()))


def _geometric_mean_of(values: pd.Series) -> float:
    return geometric_mean(pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64))


def closeness_score(error: float) -> float:
    return float(np.clip(1.0 - error, 0.0, 1.0))


def dispersion_score(ratio: float) -> float:
    if not np.isfinite(ratio):
        return float("nan")
    if ratio <= 0.0:
        return 0.0
    return float(min(ratio, 1.0 / ratio))


def structure_score(rho: float) -> float:
    return float(np.clip(rho, 0.0, 1.0))


def group_fidelity(wvs: FloatArray, llm: FloatArray) -> dict[str, Any]:
    count = int(wvs.shape[0])
    error = np.atleast_1d(np.asarray(nemd(wvs, llm), dtype=np.float64))
    paired = count >= MIN_CELLS_FOR_DISPERSION
    survey = pairwise_nemd(wvs) if paired else np.empty(0, dtype=np.float64)
    model = pairwise_nemd(llm) if paired else np.empty(0, dtype=np.float64)
    d_wvs = float(np.median(survey)) if paired else float("nan")
    d_llm = float(np.median(model)) if paired else float("nan")
    ratio = d_llm / d_wvs if d_wvs > 0.0 else float("nan")
    flat = bool(model.size > 0 and float(np.ptp(model)) == 0.0)
    if count >= MIN_CELLS_FOR_STRUCTURE and not flat:
        ranked = spearmanr(survey, model)
        rho = float(ranked.statistic)
        p_value = float(ranked.pvalue)
    else:
        rho = float("nan")
        p_value = float("nan")
    center = float(nemd(wvs.mean(axis=0), llm.mean(axis=0)))
    row: dict[str, Any] = {
        "n_cells": count,
        "n_pairs": int(survey.size),
        "e_mean_nemd": float(error.mean()),
        **{name: float(np.quantile(error, share)) for name, share in QUANTILES},
        "d_wvs": d_wvs,
        "d_llm": d_llm,
        "adaptability_ratio": ratio,
        "rho_structure": rho,
        "rho_structure_p_value": p_value,
        "model_flat": flat,
        "c_center_nemd": center,
    }
    row["score_accuracy"] = closeness_score(row["e_mean_nemd"])
    row["score_dispersion"] = dispersion_score(ratio)
    row["score_structure"] = 0.0 if flat else structure_score(rho)
    row["score_center"] = closeness_score(center)
    parts = [row["score_accuracy"], row["score_dispersion"], row["score_structure"]]
    row["pfs"] = geometric_mean(parts)
    row["pfs_with_center"] = geometric_mean([*parts, row["score_center"]])
    return row


def binding_term(frame: pd.DataFrame) -> pd.Series:
    names = dict(TERMS)
    return frame[[column for column, _ in TERMS]].idxmin(axis=1).map(names)


def _lead_with(frame: pd.DataFrame, lead: Sequence[str]) -> pd.DataFrame:
    rest = [column for column in frame.columns if column not in lead]
    return frame[[*lead, *rest]]


def base_deltas(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    columns = [column for column in DELTA_COLUMNS if column in frame.columns]
    untuned = frame[frame["arm"].eq(BASE_ARM)]
    lookup = untuned[[*keys, *columns]].rename(
        columns={column: f"_base_{column}" for column in columns}
    )
    merged = frame.merge(lookup, on=list(keys), how="left")
    for column in columns:
        merged[f"delta_{column}_vs_base"] = merged[column] - merged[f"_base_{column}"]
    return merged.drop(columns=[f"_base_{column}" for column in columns])


def build_fidelity(
    questions: list[Question] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    facets = cell_facets()
    cell_frames: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    for question in questions or resolve_questions(None):
        columns = list(question.answer_columns)
        for series, source, key, label, arm in sources():
            prepared = load_run(source, key, arm, question)
            if prepared is None:
                continue
            names = prepared.names
            facet = facets.reindex(names)
            wvs = prepared.wvs_props.loc[names, columns].to_numpy(dtype=np.float64)
            for mode in MODES:
                if mode not in prepared.modes():
                    continue
                llm = prepared.props(mode).loc[names, columns].to_numpy(dtype=np.float64)
                identity: dict[str, Any] = {
                    "question": question.var,
                    "question_label": question.label,
                    "model_key": key,
                    "model_label": label,
                    "arm": arm,
                    "series": series,
                    "source": source,
                    "mode": mode,
                }
                error = np.atleast_1d(np.asarray(nemd(wvs, llm), dtype=np.float64))
                cell_frames.append(
                    facet.assign(
                        **identity,
                        subpopulation=names,
                        nEMD=error,
                        score_accuracy=np.clip(1.0 - error, 0.0, 1.0),
                    ).reset_index(drop=True)
                )
                rows.append(
                    {
                        **identity,
                        "group": POPULATION,
                        "level": EVERY_CELL,
                        **group_fidelity(wvs, llm),
                    }
                )
                for family in FAMILIES:
                    values = facet[family]
                    for level in levels(family, values):
                        keep = values.eq(level).to_numpy()
                        rows.append(
                            {
                                **identity,
                                "group": family,
                                "level": level,
                                **group_fidelity(wvs[keep], llm[keep]),
                            }
                        )
    cells = _lead_with(pd.concat(cell_frames, ignore_index=True), [*IDENTITY, "subpopulation"])
    scored = pd.DataFrame(rows)
    scored["binding_term"] = binding_term(scored)
    groups = base_deltas(scored, ("model_key", "mode", "question", "group", "level"))
    return cells, _lead_with(groups, [*IDENTITY, "group", "level"])


def overall_fidelity(groups: pd.DataFrame) -> pd.DataFrame:
    keys = ["series", "source", "model_key", "model_label", "arm", "mode", "group", "level"]
    overall = (
        groups.groupby(keys, dropna=False)
        .agg(
            n_questions=("question", "nunique"),
            n_cells=("n_cells", "min"),
            pfs=("pfs", _geometric_mean_of),
            pfs_with_center=("pfs_with_center", _geometric_mean_of),
            score_accuracy=("score_accuracy", "mean"),
            score_dispersion=("score_dispersion", "mean"),
            score_structure=("score_structure", "mean"),
            score_center=("score_center", "mean"),
            e_mean_nemd=("e_mean_nemd", "mean"),
            e_q10_nemd=("e_q10_nemd", "mean"),
            e_q25_nemd=("e_q25_nemd", "mean"),
            e_median_nemd=("e_median_nemd", "mean"),
            e_q75_nemd=("e_q75_nemd", "mean"),
            e_q90_nemd=("e_q90_nemd", "mean"),
            adaptability_ratio=("adaptability_ratio", "mean"),
            rho_structure=("rho_structure", "mean"),
            model_flat=("model_flat", "any"),
            c_center_nemd=("c_center_nemd", "mean"),
        )
        .reset_index()
    )
    overall["binding_term"] = binding_term(overall)
    return base_deltas(overall, ("model_key", "mode", "group", "level"))


def cell_summary(cells: pd.DataFrame) -> pd.DataFrame:
    keys = ["question", "question_label", "mode", "subpopulation", *FAMILIES]
    summary = (
        cells.groupby(keys, dropna=False)
        .agg(
            n_runs=("series", "nunique"),
            mean_nEMD=("nEMD", "mean"),
            median_nEMD=("nEMD", "median"),
            min_nEMD=("nEMD", "min"),
            max_nEMD=("nEMD", "max"),
            mean_score_accuracy=("score_accuracy", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(["question", "mode", "mean_nEMD"], ascending=[True, True, False])


def _population(groups: pd.DataFrame) -> pd.DataFrame:
    return groups[groups["group"].eq(POPULATION)]


def cross_check_cells(cells: pd.DataFrame) -> int:
    reconciled = 0
    for (source, question, mode), run in cells.groupby(["source", "question", "mode"]):
        stored = read_csv(
            paths_for(str(source), str(question)).outputs / "subpopulation_distances.csv"
        )
        if stored is None:
            continue
        split = stored[stored["method"].eq(str(mode).upper())].set_index("subpopulation")["nEMD"]
        if split.empty:
            continue
        shared = run.set_index("subpopulation")["nEMD"].reindex(split.index).dropna()
        assert np.allclose(shared.to_numpy(), split.reindex(shared.index).to_numpy()), (
            f"recomputed cells disagree with subpopulation_distances.csv for {source}/{question}"
        )
        reconciled += 1
    return reconciled


def check_fidelity(cells: pd.DataFrame, groups: pd.DataFrame) -> None:
    assert not cells.empty
    assert not groups.empty
    assert set(groups["group"]) == {POPULATION, *FAMILIES}
    assert groups["n_cells"].le(EXPECTED_SUBPOPULATIONS).all()
    pooled = _population(groups).set_index(["question", "mode", "series"])["n_cells"]
    for family in FAMILIES:
        family_rows = groups[groups["group"].eq(family)]
        totals = family_rows.groupby(["question", "mode", "series"])["n_cells"].sum()
        assert totals.le(pooled.reindex(totals.index)).all(), f"{family} covers unseen cells"
    scored = groups[list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)
    finite = scored[np.isfinite(scored)]
    assert np.all((finite >= 0.0) & (finite <= 1.0))
    population = _population(groups)
    for (question, mode), group in population.groupby(["question", "mode"]):
        complete = group[group["n_cells"].eq(group["n_cells"].max())]
        spread = complete["d_wvs"].to_numpy(dtype=np.float64)
        assert np.allclose(spread, spread[0]), f"survey dispersion moved for {question}/{mode}"
    assert cells.groupby(["question", "mode", "series"])["subpopulation"].nunique().gt(0).all()
