from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

import culture
import culture.summary as culture_summary
from culture.palette import CULTURE_SLOTS, arm_marker, arm_order, arm_tone
from culture.registry import CULTURE_MODELS
from culture.runner import phase_records
from machine_bias_reproduction.prompts import PromptRecord
from machine_bias_reproduction.questions import resolve_question

MODEL = CULTURE_MODELS["gemma4_31b"]
QUESTION = resolve_question("d_happy")

GERMANY = "Germany 1997 Female 25-34 High Student Cohabiting"
GERMANY_TWO = "Germany 2018 Male 45-54 Low Retired Widowed"
MEXICO = "Mexico 2005 Male <25 Low Student Single"
UNITED_STATES = "United States 1995 Male 35-44 Low Working Single"


def test_base_is_an_arm_but_never_a_culture() -> None:
    assert culture.BASE_ARM == "base"
    assert culture.BASE_ARM not in culture.CULTURES
    assert list(culture.ARMS) == ["base", *culture.CULTURES]
    assert len(culture.CULTURES) == 10
    assert culture.is_base("base")
    assert not culture.is_base("german")


def test_resolvers_split_on_whether_weights_must_exist() -> None:
    assert culture.resolve_cultures(None) == list(culture.ARMS)
    assert culture.resolve_cultures(["all"]) == list(culture.ARMS)
    assert culture.resolve_cultures(["base"]) == ["base"]

    assert culture.resolve_finetuned_cultures(None) == list(culture.CULTURES)
    assert culture.resolve_finetuned_cultures(["all"]) == list(culture.CULTURES)
    with pytest.raises(ValueError, match="not culture-finetuned"):
        culture.resolve_finetuned_cultures(["base"])
    with pytest.raises(ValueError, match="unknown cultures"):
        culture.resolve_cultures(["klingon"])


def test_base_prints_as_released_and_every_culture_prints_itself() -> None:
    assert culture.arm_display("base") == "as released"
    assert culture.arm_display("german") == "german"
    assert culture.arm_display("Mixtral archived") == "Mixtral archived"


def test_base_run_is_labelled_as_released() -> None:
    assert MODEL.run_label("base") == "Gemma-4-31B-it (as released)"
    assert MODEL.run_label("german") == "Gemma-4-31B-it + german LoRA"


def test_base_takes_its_own_colour_and_draws_first() -> None:
    culture_inks = {arm_tone(name).ink for name in CULTURE_SLOTS}
    assert arm_tone("base").ink not in culture_inks
    assert arm_tone("base").ink != arm_tone("Mixtral archived").ink
    assert arm_marker("base") != arm_marker("Mixtral archived")
    assert arm_tone("german").ink in culture_inks
    assert arm_tone("klingon").ink  # an unknown arm degrades, it does not raise
    assert arm_order(["german", "base", "arabic"]) == ["base", "arabic", "german"]
    assert arm_order(["german", "arabic"]) == ["arabic", "german"]


def _record(profile: str) -> PromptRecord:
    return PromptRecord(prompt_id=profile, profile=profile, mode="ntp", text=f"prompt: {profile}")


def test_priority_countries_reorder_without_selecting() -> None:
    records = [_record(name) for name in (MEXICO, GERMANY, UNITED_STATES, GERMANY_TWO)]

    phases = phase_records(records, ["Germany"])

    assert [record.profile for record in phases[0]] == [GERMANY, GERMANY_TWO]
    assert [record.profile for record in phases[1]] == [MEXICO, UNITED_STATES]
    assert sorted(r.profile for phase in phases for r in phase) == sorted(
        r.profile for r in records
    )


def test_no_priority_leaves_one_phase_and_a_missing_country_leaves_none_empty() -> None:
    records = [_record(name) for name in (MEXICO, GERMANY)]
    assert phase_records(records, None) == [records]
    assert phase_records(records, []) == [records]
    assert [len(phase) for phase in phase_records(records, ["Russia"])] == [2]


def _distances(values: dict[str, float]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"method": "NTP", "subpopulation": list(values), "nEMD": list(values.values())}
    )
    frame["country"] = frame["subpopulation"].str.replace(r"^(.*?) \d.*$", r"\1", regex=True)
    return frame


def _cell(frame: pd.DataFrame, row: str, column: str) -> float:
    return float(cast(float, frame.loc[row, column]))


def _coefficients(center: float, germany: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mode": "ntp",
            "model": "full",
            "predictor": ["nEMD_center", "countryGermany"],
            "estimate": [center, germany],
        }
    )


def _serve(
    monkeypatch: pytest.MonkeyPatch,
    base: pd.DataFrame | None,
    arm: pd.DataFrame,
    base_fit: pd.DataFrame | None = None,
    arm_fit: pd.DataFrame | None = None,
) -> None:
    monkeypatch.setattr(
        culture_summary,
        "_distances",
        lambda source, _q: base if source.endswith("/base") else arm,
    )
    monkeypatch.setattr(
        culture_summary,
        "_full_coefficients",
        lambda source, _q: base_fit if source.endswith("/base") else arm_fit,
    )


def test_base_delta_is_negative_when_finetuning_moved_the_model_closer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _distances({GERMANY: 0.30, MEXICO: 0.20})
    arm = _distances({GERMANY: 0.10, MEXICO: 0.25})
    _serve(monkeypatch, base, arm, _coefficients(0.50, -0.01), _coefficients(0.42, -0.06))

    table = culture_summary.base_delta_table(MODEL, ["german"], [QUESTION])
    rows = table[table["mode"] == "NTP"].set_index("country")

    assert _cell(rows, "Germany", "delta_nEMD") == pytest.approx(-0.20)
    assert _cell(rows, "Mexico", "delta_nEMD") == pytest.approx(0.05)
    assert _cell(rows, "(all)", "delta_nEMD") == pytest.approx(-0.075)
    assert _cell(rows, "Germany", "beta_country_delta") == pytest.approx(-0.05)
    assert _cell(rows, "Germany", "nEMD_center_delta") == pytest.approx(-0.08)


def test_base_delta_averages_only_subpopulations_both_arms_scored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _distances({GERMANY: 0.30, GERMANY_TWO: 9.0})
    arm = _distances({GERMANY: 0.10})
    _serve(monkeypatch, base, arm)

    rows = culture_summary.base_delta_table(MODEL, ["german"], [QUESTION])
    germany = rows[(rows["mode"] == "NTP") & (rows["country"] == "Germany")].iloc[0]

    assert germany["subpopulations"] == 1
    assert germany["base_mean_nEMD"] == pytest.approx(0.30)
    assert germany["delta_nEMD"] == pytest.approx(-0.20)


def test_base_delta_is_empty_without_a_base_run(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, None, _distances({GERMANY: 0.10}))
    assert culture_summary.base_delta_table(MODEL, ["german"], [QUESTION]).empty


def test_omitted_reference_country_keeps_a_null_coefficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _distances({UNITED_STATES: 0.30})
    arm = _distances({UNITED_STATES: 0.10})
    _serve(monkeypatch, base, arm, _coefficients(0.5, -0.01), _coefficients(0.4, -0.06))

    rows = culture_summary.base_delta_table(MODEL, ["german"], [QUESTION])
    united_states = rows[rows["country"] == "United States"].iloc[0]

    assert pd.isna(united_states["beta_country_delta"])
    assert united_states["nEMD_center_delta"] == pytest.approx(-0.10)


def test_home_advantage_reports_both_references_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tuned = _distances({GERMANY: 0.10, MEXICO: 0.30})
    mixtral = _distances({GERMANY: 0.20, MEXICO: 0.20})
    base = _distances({GERMANY: 0.30, MEXICO: 0.30})
    monkeypatch.setattr(
        culture_summary,
        "_distances",
        lambda source, _q: {
            "archived": mixtral,
            f"culture/{MODEL.key}/base": base,
        }.get(source, tuned),
    )

    table = culture_summary.home_advantage_table(MODEL, ["german", "base"], [QUESTION])
    ntp = table[table["mode"] == "NTP"].set_index("reference")

    assert set(ntp.index) == {culture_summary.MIXTRAL_REFERENCE, culture_summary.BASE_REFERENCE}
    assert set(table["culture"]) == {"german"}
    mixtral_did = _cell(ntp, culture_summary.MIXTRAL_REFERENCE, "difference_in_differences")
    base_did = _cell(ntp, culture_summary.BASE_REFERENCE, "difference_in_differences")
    assert mixtral_did == pytest.approx(-0.20)
    assert base_did == pytest.approx(-0.20)
