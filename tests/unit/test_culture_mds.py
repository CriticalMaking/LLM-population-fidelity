from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from culture import mds as culture_mds_module
from culture.mds import (
    _all_countries_mds_embedding,
    _comparison_plate,
    _home_mask,
    _matched_mds_embedding,
    _sample_plate,
)
from culture.registry import CULTURE_MODELS
from machine_bias_reproduction.data import Coverage, PreparedData
from machine_bias_reproduction.questions import resolve_question

QUESTION = resolve_question("d_happy")
COLUMNS = list(QUESTION.answer_columns)

SUBPOPULATIONS = (
    "Australia 1995 Female 25-34 High Working Married",
    "United States 1995 Male 35-44 Low Working Single",
    "Germany 1997 Female 25-34 High Student Cohabiting",
    "Mexico 2005 Male 45-54 Low Working Married",
    "Russia 2011 Female 55-64 High Retired Widowed",
)


def _props(names: list[str], seed: int) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    values = generator.random((len(names), len(COLUMNS))) + 0.1
    values /= values.sum(axis=1, keepdims=True)
    return pd.DataFrame(values, index=pd.Index(names, name="subpop"), columns=COLUMNS)


def _prepared(names: list[str], seed: int) -> PreparedData:
    empty = pd.DataFrame()
    return PreparedData(
        question=QUESTION,
        wvs=empty,
        subpops=empty,
        ntp_raw=empty,
        fa_raw=empty,
        names=pd.Index(names),
        wvs_props=_props(names, 0),
        ntp_props=_props(names, seed),
        fa_props=_props(names, seed + 100),
        social_predictors=empty,
        coverage=Coverage(0, 0, 0, 0, len(names), len(names)),
    )


def _series(names: list[str]) -> dict[str, PreparedData]:
    return {
        "Mixtral archived": _prepared(names, 1),
        "Mixtral fresh": _prepared(names, 2),
        "german": _prepared(names, 3),
    }


def test_embedding_keeps_only_the_cultures_own_countries() -> None:
    embedded = _matched_mds_embedding(_series(list(SUBPOPULATIONS)), "german", QUESTION)
    assert embedded is not None
    names, _, _ = embedded
    assert list(names) == ["Germany 1997 Female 25-34 High Student Cohabiting"]


def test_embedding_intersects_subpopulations_across_series() -> None:
    shared = [
        "Germany 1997 Female 25-34 High Student Cohabiting",
        "Germany 2006 Male 35-44 Low Working Married",
    ]
    series = _series([*shared, "Germany 2013 Female 65+ High Retired Widowed"])
    series["german"] = _prepared(shared, 3)

    embedded = _matched_mds_embedding(series, "german", QUESTION)
    assert embedded is not None
    names, wvs, model = embedded
    assert sorted(names) == sorted(shared)
    assert wvs.shape == (len(shared), 2)
    for coordinates in model.values():
        assert coordinates.shape == (len(shared), 2)


def test_embedding_returns_one_block_per_series_and_mode() -> None:
    series = _series(["Germany 1997 A", "Germany 1998 B", "Germany 1999 C"])
    embedded = _matched_mds_embedding(series, "german", QUESTION)
    assert embedded is not None
    _, _, model = embedded
    assert set(model) == {(label, mode) for label in series for mode in ("NTP", "FA")}


def test_embedding_is_none_when_the_culture_has_no_respondents() -> None:
    series = _series(list(SUBPOPULATIONS))
    assert _matched_mds_embedding(series, "korean", QUESTION) is None


def test_all_countries_embedding_is_strictly_wider_than_the_matched_one() -> None:
    series = _series(list(SUBPOPULATIONS))
    matched = _matched_mds_embedding(series, "german", QUESTION)
    everyone = _all_countries_mds_embedding(series, QUESTION)
    assert matched is not None and everyone is not None
    assert set(matched[0]) < set(everyone[0])
    assert list(everyone[0]) == list(SUBPOPULATIONS)


def test_all_countries_embedding_still_intersects_across_series() -> None:
    shared = [SUBPOPULATIONS[0], SUBPOPULATIONS[2]]
    series = _series(list(SUBPOPULATIONS))
    series["german"] = _prepared(shared, 3)

    everyone = _all_countries_mds_embedding(series, QUESTION)
    assert everyone is not None
    assert sorted(everyone[0]) == sorted(shared)


def test_home_mask_marks_the_matched_subset_inside_the_wider_cloud() -> None:
    names = pd.Index(SUBPOPULATIONS)
    mask = _home_mask(names, "english")
    assert list(names[mask]) == [SUBPOPULATIONS[0], SUBPOPULATIONS[1]]
    assert _home_mask(names, "german").sum() == 1
    assert not _home_mask(names, "korean").any()


def test_no_plate_is_drawn_when_the_culture_run_never_happened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references = {
        "Mixtral archived": _prepared(list(SUBPOPULATIONS), 1),
        "Mixtral fresh": _prepared(list(SUBPOPULATIONS), 2),
    }
    monkeypatch.setattr(culture_mds_module, "_series", lambda *_: references)

    produced, table = _sample_plate(
        CULTURE_MODELS["gemma4_31b"], "spanish", QUESTION, all_countries=False
    )
    assert produced == []
    assert table.empty


def test_comparison_scores_both_arms_on_identical_subpopulations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shared = [
        "Germany 1997 Female 25-34 High Student Cohabiting",
        "Germany 2006 Male 35-44 Low Working Married",
    ]
    series = _series([*shared, "Germany 2013 Female 65+ High Retired Widowed"])
    base = _prepared(shared, 4)
    monkeypatch.setattr(culture_mds_module, "_plate_series", lambda *_: series)
    monkeypatch.setattr(culture_mds_module, "load_source", lambda *_: base)
    monkeypatch.setattr(culture_mds_module, "CULTURE_FIGURES", tmp_path)

    produced, table = _comparison_plate(CULTURE_MODELS["gemma4_31b"], "german", QUESTION)

    assert [path.name for path in produced] == [
        "fig_culture_mds_comparison.png",
        "fig_culture_mds_comparison.pdf",
    ]
    assert set(table["series"]) == {"base", "german", "Mixtral archived", "Mixtral fresh"}
    assert set(table["sample"]) == {"comparison"}
    assert set(table["subpopulations"]) == {len(shared)}
    assert len(table) == len(set(zip(table["series"], table["mode"], strict=True)))


def test_no_comparison_is_drawn_without_a_base_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        culture_mds_module, "_plate_series", lambda *_: _series(list(SUBPOPULATIONS))
    )
    monkeypatch.setattr(
        culture_mds_module,
        "load_source",
        lambda *_: (_ for _ in ()).throw(FileNotFoundError("no base run")),
    )

    produced, table = _comparison_plate(CULTURE_MODELS["gemma4_31b"], "german", QUESTION)
    assert produced == []
    assert table.empty
