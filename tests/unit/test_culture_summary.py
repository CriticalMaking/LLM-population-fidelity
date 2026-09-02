from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from culture import summary as culture_summary
from culture.matching import home_splits, restrict_to
from culture.registry import CULTURE_MODELS
from machine_bias_reproduction.questions import resolve_question

MODEL = CULTURE_MODELS["gemma4_31b"]
QUESTION = resolve_question("d_happy")

AUSTRALIA = "Australia 1995 Female 25-34 High Working Married"
UNITED_STATES = "United States 1995 Male 35-44 Low Working Single"
GERMANY = "Germany 1997 Female 25-34 High Student Cohabiting"
RUSSIA = "Russia 2011 Female 55-64 High Retired Widowed"


def _distances(values: dict[str, float]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "method": "NTP",
            "subpopulation": list(values),
            "nEMD": list(values.values()),
        }
    )
    frame["country"] = frame["subpopulation"].str.replace(r"^(.*?) \d.*$", r"\1", regex=True)
    return frame


def _patch(
    monkeypatch: pytest.MonkeyPatch,
    tuned: pd.DataFrame,
    reference: pd.DataFrame,
    base: pd.DataFrame | None = None,
) -> None:

    def serve(source: str, _question: object) -> pd.DataFrame | None:
        if source == "archived":
            return reference
        if source.endswith("/base"):
            return base
        return tuned

    monkeypatch.setattr(culture_summary, "_distances", serve)


def test_home_advantage_differences_out_the_references_own_head_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tuned = _distances({GERMANY: 0.20, RUSSIA: 0.30})
    reference = _distances({GERMANY: 0.05, RUSSIA: 0.15})
    _patch(monkeypatch, tuned, reference)

    table = culture_summary.home_advantage_table(MODEL, ["german"], [QUESTION])
    row = table[table["mode"] == "NTP"].iloc[0]
    assert row["tuned_home_nEMD"] == pytest.approx(0.20)
    assert row["reference_home_nEMD"] == pytest.approx(0.05)
    assert row["difference_in_differences"] == pytest.approx(0.0)


def test_home_advantage_scores_the_reference_on_identical_subpopulations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tuned = _distances({GERMANY: 0.20, RUSSIA: 0.30})
    reference = _distances(
        {GERMANY: 0.05, RUSSIA: 0.15, "Russia 2011 Male 25-34 Low Working Single": 9.0}
    )
    _patch(monkeypatch, tuned, reference)

    table = culture_summary.home_advantage_table(MODEL, ["german"], [QUESTION])
    row = table[table["mode"] == "NTP"].iloc[0]
    assert row["reference_away_nEMD"] == pytest.approx(0.15)
    assert row["subpopulations_away"] == 1


def test_english_yields_a_pooled_row_and_one_row_per_country(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tuned = _distances({AUSTRALIA: 0.10, UNITED_STATES: 0.30, GERMANY: 0.20, RUSSIA: 0.20})
    reference = _distances({AUSTRALIA: 0.10, UNITED_STATES: 0.10, GERMANY: 0.10, RUSSIA: 0.10})
    _patch(monkeypatch, tuned, reference)

    table = culture_summary.home_advantage_table(MODEL, ["english"], [QUESTION])
    ntp = table[table["mode"] == "NTP"].set_index("home_country")
    assert set(ntp.index) == {"Australia; United States", "Australia", "United States"}

    def cell(country: str, column: str) -> float:
        return float(cast(float, ntp.loc[country, column]))

    assert cell("Australia; United States", "subpopulations_home") == 2
    assert cell("Australia", "subpopulations_home") + cell(
        "United States", "subpopulations_home"
    ) == cell("Australia; United States", "subpopulations_home")
    assert cell("Australia", "difference_in_differences") < 0
    assert cell("United States", "difference_in_differences") > 0


def test_a_single_country_culture_yields_only_the_pooled_row() -> None:
    assert home_splits("german") == [("Germany", ("Germany",))]
    assert [label for label, _ in home_splits("english")] == [
        "Australia; United States",
        "Australia",
        "United States",
    ]
    assert home_splits("korean") == []


def test_restrict_to_never_falls_back_to_the_whole_frame() -> None:
    frame = _distances({AUSTRALIA: 0.1, GERMANY: 0.2})
    assert restrict_to(frame, ()).empty
    assert list(restrict_to(frame, ("Germany",))["subpopulation"]) == [GERMANY]


def test_compression_is_the_survey_spread_over_the_model_spread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = {
        ("overall_nEMD", "NTP"): 0.09,
        ("median_pairwise_nEMD", "NTP"): 0.02,
        ("median_pairwise_nEMD", "WVS"): 0.07,
    }
    monkeypatch.setattr(culture_summary, "_summary_metrics", lambda *_: metrics)

    table = culture_summary.distance_table(MODEL, ["english"], [QUESTION])
    row = table[table["mode"] == "NTP"].iloc[0]
    assert row["overall_nEMD"] == pytest.approx(0.09)
    assert row["compression"] == pytest.approx(3.5)
