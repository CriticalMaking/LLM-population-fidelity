from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from machine_bias_reproduction.config import GLOBAL_SEED, UPSTREAM_DATA
from machine_bias_reproduction.data import group_responses, load_subpops, load_wvs
from machine_bias_reproduction.questions import Question, one_hot, resolve_question

from .series import Series

RANDOM_REPLICATES = 20
CSV_ROOT = UPSTREAM_DATA / "LLM-outputs" / "csv"


@dataclass(frozen=True, slots=True)
class QuestionData:
    question: Question
    names: pd.Index
    wvs: pd.DataFrame
    wvs_marginal: np.ndarray
    series: dict[str, pd.DataFrame]
    marginals: dict[str, np.ndarray]
    linear: pd.DataFrame | None
    random: list[pd.DataFrame]
    ntp_mass: dict[str, float]

    def props(self, name: str) -> pd.DataFrame | None:
        if name == "WVS":
            return self.wvs
        if name == "Linear":
            return self.linear
        return self.series.get(name)


def _marginal(values: pd.Series, categories: tuple[str, ...]) -> np.ndarray:
    counts = values.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=np.float64)
    total = counts.sum()
    return np.asarray(counts / total, dtype=np.float64) if total else counts


def _ntp_frame(series: Series, question: Question) -> pd.DataFrame | None:
    path = CSV_ROOT / f"NTP-{series.model}-{question.var}.csv"
    return pd.read_csv(path) if path.is_file() else None


@lru_cache(maxsize=8)
def _fa_frame(model: str) -> pd.DataFrame | None:
    path = CSV_ROOT / f"FA-{model}.csv"
    return pd.read_csv(path) if path.is_file() else None


def _linear(question: Question) -> pd.DataFrame | None:
    path = UPSTREAM_DATA / "Linear" / f"Linear-baseline-{question.var}.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path).set_index("name")


def _random(question: Question, wvs: pd.DataFrame, subpops: pd.Series) -> list[pd.DataFrame]:
    answers = question.normalize(wvs[question.var]).to_numpy(copy=True)
    rng = np.random.default_rng(GLOBAL_SEED)
    frames: list[pd.DataFrame] = []
    for _ in range(RANDOM_REPLICATES):
        shuffled = pd.Series(rng.permutation(answers))
        responses = one_hot(shuffled, question.wvs_labels, question.answer_columns)
        frames.append(group_responses(responses, subpops))
    return frames


def load_question(question: str | Question, series: list[Series]) -> QuestionData:
    outcome = resolve_question(question)
    columns = outcome.answer_columns
    wvs = load_wvs()
    subpops = load_subpops()["subpop"]

    wvs_responses = one_hot(outcome.normalize(wvs[outcome.var]), outcome.wvs_labels, columns)
    wvs_props = group_responses(wvs_responses, subpops)
    defined = wvs_props.notna().all(axis=1)
    names = wvs_props.index[defined.to_numpy()]

    loaded: dict[str, pd.DataFrame] = {}
    marginals: dict[str, np.ndarray] = {}
    ntp_mass: dict[str, float] = {}
    for entry in series:
        if entry.strategy == "NTP":
            frame = _ntp_frame(entry, outcome)
            if frame is None:
                continue
            answers = outcome.ntp_answers(frame.set_index("profile"))
            matched = answers.reindex(wvs["profile"]).reset_index(drop=True)
            loaded[entry.name] = group_responses(matched, subpops).reindex(names)
            marginals[entry.name] = np.asarray(
                answers.mean().to_numpy(dtype=np.float64), dtype=np.float64
            )
            ntp_mass[entry.name] = float(frame["mass"].mean())
        else:
            frame = _fa_frame(entry.model)
            if frame is None or outcome.var not in frame.columns:
                continue
            matched = frame.set_index("id").reindex(wvs["id"]).reset_index(drop=True)
            values = outcome.normalize(matched[outcome.var])
            loaded[entry.name] = group_responses(
                one_hot(values, outcome.fa_answers, columns), subpops
            ).reindex(names)
            marginals[entry.name] = _marginal(values, outcome.fa_answers)

    linear = _linear(outcome)
    return QuestionData(
        question=outcome,
        names=names,
        wvs=wvs_props.reindex(names),
        wvs_marginal=_marginal(outcome.normalize(wvs[outcome.var]), outcome.wvs_labels),
        series=loaded,
        marginals=marginals,
        linear=linear.reindex(names)[list(columns)] if linear is not None else None,
        random=[frame.reindex(names) for frame in _random(outcome, wvs, subpops)],
        ntp_mass=ntp_mass,
    )


def robustness_root() -> Path:
    return UPSTREAM_DATA / "Robustness"
