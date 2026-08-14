from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from .config import (
    EXPECTED_NTP_PROFILES,
    EXPECTED_SUBPOPULATIONS,
    EXPECTED_WVS_ROWS,
    UPSTREAM_DATA,
)
from .questions import Question, one_hot, resolve_question

FloatArray = npt.NDArray[np.float64]

MIN_VALID_ANSWERS_PER_SUBPOPULATION = 20


@dataclass(frozen=True, slots=True)
class Coverage:
    ntp_expected: int
    ntp_observed: int
    fa_expected: int
    fa_observed: int
    subpopulations_total: int
    subpopulations_retained: int

    @property
    def subpopulations_dropped(self) -> int:
        return self.subpopulations_total - self.subpopulations_retained

    @property
    def complete(self) -> bool:
        return (
            self.ntp_observed == self.ntp_expected
            and self.fa_observed == self.fa_expected
            and self.subpopulations_retained == self.subpopulations_total
        )

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "ntp_expected": self.ntp_expected,
            "ntp_observed": self.ntp_observed,
            "fa_expected": self.fa_expected,
            "fa_observed": self.fa_observed,
            "subpopulations_total": self.subpopulations_total,
            "subpopulations_retained": self.subpopulations_retained,
            "subpopulations_dropped": self.subpopulations_dropped,
            "complete": self.complete,
        }


@dataclass(frozen=True, slots=True)
class PreparedData:
    question: Question
    wvs: pd.DataFrame
    subpops: pd.DataFrame
    ntp_raw: pd.DataFrame
    fa_raw: pd.DataFrame
    names: pd.Index
    wvs_props: pd.DataFrame
    ntp_props: pd.DataFrame
    fa_props: pd.DataFrame
    social_predictors: pd.DataFrame
    coverage: Coverage


def load_wvs() -> pd.DataFrame:
    path = UPSTREAM_DATA / "WVS" / "wvs-data.csv"
    frame = pd.read_csv(path)
    if len(frame) != EXPECTED_WVS_ROWS:
        raise ValueError(f"expected {EXPECTED_WVS_ROWS} WVS rows, found {len(frame)}")
    return frame


def load_subpops() -> pd.DataFrame:
    return pd.read_csv(UPSTREAM_DATA / "subpops.csv")


def archived_ntp(question: str | Question, model: str = "Mixtral-8x7B") -> pd.DataFrame:
    outcome = resolve_question(question)
    path = UPSTREAM_DATA / "LLM-outputs" / "csv" / f"NTP-{model}-{outcome.var}.csv"
    frame = pd.read_csv(path)
    if len(frame) != EXPECTED_NTP_PROFILES:
        raise ValueError(f"expected {EXPECTED_NTP_PROFILES} NTP profiles, found {len(frame)}")
    return frame


def archived_fa(question: str | Question, model: str = "Mixtral-8x7B") -> pd.DataFrame:
    outcome = resolve_question(question)
    path = UPSTREAM_DATA / "LLM-outputs" / "csv" / f"FA-{model}.csv"
    return pd.read_csv(path)[["id", "profile", outcome.var]]


def load_linear_baseline(question: str | Question) -> pd.DataFrame:
    outcome = resolve_question(question)
    return pd.read_csv(UPSTREAM_DATA / "Linear" / f"Linear-baseline-{outcome.var}.csv")


def _group_responses(responses: pd.DataFrame, subpopulation: pd.Series) -> pd.DataFrame:
    frame = responses.copy()
    frame["name"] = subpopulation.to_numpy()
    answers = [column for column in frame.columns if column != "name"]
    return frame.groupby("name", sort=True)[answers].mean()


def _group_counts(responses: pd.DataFrame, subpopulation: pd.Series) -> pd.Series:
    answered = responses.notna().any(axis=1)
    return answered.groupby(subpopulation.to_numpy()).sum()


def social_predictors(wvs: pd.DataFrame, subpopulation: pd.Series) -> pd.DataFrame:
    age_labels = ("<25", "25-34", "35-44", "45-54", "55-64", "65-74", "75+")
    age = pd.cut(
        wvs["i_age"],
        bins=(10, 25, 35, 45, 55, 65, 75, 999),
        labels=age_labels,
    )
    variables: dict[str, tuple[pd.Series, tuple[str, ...]]] = {
        "Sex": (wvs["i_sex"], ("Male", "Female")),
        "Age": (pd.Series(age, index=wvs.index), age_labels),
        "Education": (wvs["i_education"], ("High", "Low", "Middle")),
        "Employment": (
            wvs["i_employment"],
            ("Homemaker", "Retired", "Student", "Unemployed", "Working"),
        ),
        "Marstat": (
            wvs["i_marstat"],
            ("Cohabiting", "Divorced or separated", "Married", "Single", "Widowed"),
        ),
    }
    parts: list[pd.DataFrame] = []
    for prefix, (values, categories) in variables.items():
        encoded = pd.get_dummies(
            pd.Categorical(values, categories=categories),
            prefix=prefix,
            prefix_sep="_",
            dtype=np.float64,
        )
        encoded.loc[values.isna(), :] = np.nan
        parts.append(encoded)
    all_predictors = pd.concat(parts, axis=1)
    all_predictors["name"] = subpopulation.to_numpy()
    return all_predictors.groupby("name", sort=True).mean()


def prepare_data(
    ntp: pd.DataFrame,
    fa: pd.DataFrame,
    question: str | Question,
    *,
    min_valid: int = MIN_VALID_ANSWERS_PER_SUBPOPULATION,
) -> PreparedData:
    outcome = resolve_question(question)
    wvs = load_wvs()
    subpops = load_subpops()
    if not wvs["id"].equals(subpops["id"]):
        raise ValueError("WVS and subpopulation IDs are not identically ordered")

    columns = outcome.answer_columns
    wvs_responses = one_hot(outcome.normalize(wvs[outcome.var]), outcome.wvs_labels, columns)
    wvs_props = _group_responses(wvs_responses, subpops["subpop"])

    ntp_by_profile = outcome.ntp_answers(ntp.set_index("profile"))
    matched_ntp = ntp_by_profile.reindex(wvs["profile"]).reset_index(drop=True)
    ntp_props = _group_responses(matched_ntp, subpops["subpop"])
    ntp_counts = _group_counts(matched_ntp, subpops["subpop"])

    matched_fa = fa.set_index("id").reindex(wvs["id"]).reset_index(drop=True)
    fa_responses = one_hot(outcome.normalize(matched_fa[outcome.var]), outcome.fa_answers, columns)
    fa_props = _group_responses(fa_responses, subpops["subpop"])
    fa_counts = _group_counts(fa_responses, subpops["subpop"])

    total = len(wvs_props.index)
    if total != EXPECTED_SUBPOPULATIONS:
        raise ValueError(f"expected {EXPECTED_SUBPOPULATIONS} subpopulations, found {total}")
    if not wvs_props.index.equals(ntp_props.index) or not wvs_props.index.equals(fa_props.index):
        raise ValueError("WVS, NTP, and FA subpopulation rows are not aligned")

    keep = (
        (ntp_counts.reindex(wvs_props.index, fill_value=0) >= min_valid)
        & (fa_counts.reindex(wvs_props.index, fill_value=0) >= min_valid)
        & wvs_props.notna().all(axis=1)
        & ntp_props.notna().all(axis=1)
        & fa_props.notna().all(axis=1)
    )
    names = wvs_props.index[keep.to_numpy()]

    unique_profiles = wvs["profile"].drop_duplicates()
    coverage = Coverage(
        ntp_expected=len(unique_profiles),
        ntp_observed=int(unique_profiles.isin(ntp_by_profile.index).sum()),
        fa_expected=len(wvs),
        fa_observed=int(fa_responses.notna().any(axis=1).sum()),
        subpopulations_total=total,
        subpopulations_retained=len(names),
    )

    predictors = social_predictors(wvs, subpops["subpop"]).reindex(names)
    if predictors.isna().any().any():
        raise ValueError("social predictor aggregation produced missing values")

    return PreparedData(
        question=outcome,
        wvs=wvs,
        subpops=subpops,
        ntp_raw=ntp,
        fa_raw=fa,
        names=names,
        wvs_props=wvs_props.loc[names],
        ntp_props=ntp_props.loc[names],
        fa_props=fa_props.loc[names],
        social_predictors=predictors,
        coverage=coverage,
    )


def canonical_run_paths(
    outputs: Path,
    question: str | Question,
    stems: tuple[str, str] | None = None,
) -> tuple[Path, Path]:
    outcome = resolve_question(question)
    ntp_stem, fa_stem = stems or (
        f"NTP-Mixtral-8x7B-{outcome.var}.csv",
        f"FA-Mixtral-8x7B-{outcome.var}.csv",
    )
    return outputs / ntp_stem, outputs / fa_stem
