from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from culture import mds_groups
from culture.mds_groups import (
    base_families_plate,
    finetuned_plate,
    open_models,
    tier_plate,
)
from culture.registry import CULTURE_MODELS, SIZE_TIERS, is_served
from machine_bias_reproduction.data import Coverage, PreparedData
from machine_bias_reproduction.questions import Question, resolve_question

QUESTION = resolve_question("d_happy")
COLUMNS = list(QUESTION.answer_columns)

GERMAN = (
    "Germany 1997 Female 25-34 High Student Cohabiting",
    "Germany 2006 Male 35-44 Low Working Married",
)

ELSEWHERE = (
    "Australia 1995 Female 25-34 High Working Married",
    "United States 1995 Male 35-44 Low Working Single",
    "Mexico 2005 Male 45-54 Low Working Married",
)

EVERYWHERE = (*GERMAN, *ELSEWHERE)

Loader = Callable[[str, str | None, str | None, Question], PreparedData | None]


def _props(names: tuple[str, ...], seed: int) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    values = generator.random((len(names), len(COLUMNS))) + 0.1
    values /= values.sum(axis=1, keepdims=True)
    return pd.DataFrame(values, index=pd.Index(names, name="subpop"), columns=COLUMNS)


def _prepared(
    names: tuple[str, ...], seed: int, *, full_answers_only: bool = False
) -> PreparedData:
    empty = pd.DataFrame()
    return PreparedData(
        question=QUESTION,
        wvs=empty,
        subpops=empty,
        ntp_raw=empty,
        fa_raw=empty,
        names=pd.Index(names),
        wvs_props=_props(names, 0),
        ntp_props=None if full_answers_only else _props(names, seed),
        fa_props=_props(names, seed + 100),
        social_predictors=empty,
        coverage=Coverage(0, 0, 0, 0, len(names), len(names)),
    )


def _loader(runs: dict[str, PreparedData]) -> Loader:
    references = {
        "archived": _prepared(EVERYWHERE, 1),
        "fresh": _prepared(EVERYWHERE, 2),
    }

    def load(
        source: str, key: str | None, arm: str | None, question: Question
    ) -> PreparedData | None:
        return references.get(source) or runs.get(source)

    return load


def _both_arms(
    keys: tuple[str, ...], names: tuple[str, ...] = EVERYWHERE
) -> dict[str, PreparedData]:
    return {
        f"culture/{key}/{arm}": _prepared(names, seed)
        for seed, key in enumerate(keys, start=10)
        for arm in ("base", "german")
    }


def _install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runs: dict[str, PreparedData]
) -> None:
    monkeypatch.setattr(mds_groups, "load_run", _loader(runs))
    monkeypatch.setattr(mds_groups, "CULTURE_FIGURES", tmp_path)


def test_tier_plate_emits_the_tier_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(monkeypatch, tmp_path, _both_arms(SIZE_TIERS["small"]))

    produced, table = tier_plate("small", QUESTION)

    assert [path.name for path in produced] == [
        "fig_culture_mds_tier_small.png",
        "fig_culture_mds_tier_small.pdf",
    ]
    assert set(table["plate"]) == {"tier_small"}
    assert set(table["mode"]) == {"NTP", "FA"}
    assert set(table["sample"]) == {"german_matched"}
    assert set(table["series"]) == {"base", "german", "Mixtral archived", "Mixtral fresh"}
    assert set(table["model_key"].dropna()) == set(SIZE_TIERS["small"])
    assert set(table["subpopulations"]) == {len(GERMAN)}


def test_tier_plate_skips_a_model_missing_its_german_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    absent = SIZE_TIERS["small"][0]
    runs = _both_arms(SIZE_TIERS["small"])
    del runs[f"culture/{absent}/german"]
    _install(monkeypatch, tmp_path, runs)

    produced, table = tier_plate("small", QUESTION)

    assert produced != []
    assert absent not in set(table["model_key"].dropna())
    assert set(table["model_key"].dropna()) == set(SIZE_TIERS["small"][1:])


def test_tier_plate_draws_nothing_when_no_member_has_both_arms(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs = {
        f"culture/{key}/base": _prepared(EVERYWHERE, 10) for key in SIZE_TIERS["small"]
    }
    _install(monkeypatch, tmp_path, runs)

    produced, table = tier_plate("small", QUESTION)

    assert produced == []
    assert table.empty


def test_base_families_is_full_answers_only_and_keeps_a_served_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs = {
        "culture/gemma4_31b/base": _prepared(EVERYWHERE, 20),
        "culture/terra/base": _prepared(EVERYWHERE, 21, full_answers_only=True),
    }
    _install(monkeypatch, tmp_path, runs)

    produced, table = base_families_plate(QUESTION)

    assert [path.name for path in produced] == [
        "fig_culture_mds_base_families.png",
        "fig_culture_mds_base_families.pdf",
    ]
    assert set(table["mode"]) == {"FA"}
    assert set(table["sample"]) == {"all_countries"}
    assert set(table["model_key"].dropna()) == {"gemma4_31b", "terra"}
    assert set(table["series"]) - {"Mixtral archived", "Mixtral fresh"} == {"base"}
    assert set(table["subpopulations"]) == {len(EVERYWHERE)}


def test_base_families_intersects_names_across_every_series(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shared = (GERMAN[0], ELSEWHERE[0])
    runs = {
        "culture/gemma4_31b/base": _prepared(EVERYWHERE, 20),
        "culture/terra/base": _prepared(shared, 21, full_answers_only=True),
    }
    _install(monkeypatch, tmp_path, runs)

    _, table = base_families_plate(QUESTION)

    assert set(table["subpopulations"]) == {len(shared)}


def test_finetuned_plate_restricts_to_germany(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs = {
        f"culture/{model.key}/german": _prepared(EVERYWHERE, 30 + index)
        for index, model in enumerate(open_models())
    }
    _install(monkeypatch, tmp_path, runs)

    produced, table = finetuned_plate(QUESTION)

    assert [path.name for path in produced] == [
        "fig_culture_mds_finetuned.png",
        "fig_culture_mds_finetuned.pdf",
    ]
    assert set(table["subpopulations"]) == {len(GERMAN)}
    assert set(table["series"]) - {"Mixtral archived", "Mixtral fresh"} == {"german"}
    assert set(table["model_key"].dropna()) == {model.key for model in open_models()}


def test_size_tiers_partition_the_open_models() -> None:
    tiered = [key for keys in SIZE_TIERS.values() for key in keys]
    assert len(tiered) == len(set(tiered))
    assert set(tiered) == {key for key in CULTURE_MODELS if not is_served(key)}
    assert not any(is_served(key) for key in tiered)
