from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest

from culture import response_panels
from culture.palette import arm_tone
from machine_bias_reproduction.plates import shaded
from machine_bias_reproduction.questions import resolve_question

matplotlib.use("Agg")

GERMAN_SHARES = np.array([0.2112, 0.6212, 0.1460, 0.0217])


def _labelled_rules(axes: list[Any]) -> dict[str, Any]:
    return {
        collection.get_label(): collection
        for axis in axes
        for collection in axis.collections
        if collection.get_label() and not collection.get_label().startswith("_")
    }


def _shift_frames(cultures: tuple[str, ...], *, silent: str = "") -> dict[str, dict[str, Any]]:
    columns = list(resolve_question("d_happy").answer_columns)
    shares = {
        "WVS": (0.30, 0.54, 0.14, 0.02),
        "NTP": (0.38, 0.61, 0.01, 0.00),
        "FA": (0.25, 0.55, 0.15, 0.05),
    }
    return {
        culture: {
            "responses": pd.DataFrame(
                [
                    {"method": method, **dict(zip(columns, values, strict=True))}
                    for method, values in shares.items()
                    if not (culture == silent and method == "FA")
                ]
            )
        }
        for culture in cultures
    }


def test_the_home_country_survey_gets_its_own_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, captured_axes: list[Any]
) -> None:
    asked: list[tuple[str, tuple[str, ...]]] = []

    def shares(question: Any, countries: tuple[str, ...]) -> Any:
        asked.append((question.var, countries))
        return GERMAN_SHARES

    monkeypatch.setattr(response_panels, "survey_shares", shares)

    response_panels.response_shift(
        _shift_frames(("base", "german")), resolve_question("d_happy"), tmp_path
    )

    assert asked == [("d_happy", ("Germany",))] * 2
    rules = _labelled_rules(captured_axes)
    assert sorted(rules) == ["WVS", "WVS-german"]
    assert rules["WVS-german"].get_segments()[0][0][1] == pytest.approx(GERMAN_SHARES[0])


def test_an_arm_with_no_home_country_keeps_only_the_pooled_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, captured_axes: list[Any]
) -> None:
    def refuse(*_: Any) -> Any:
        raise AssertionError("base has no home country to restrict to")

    monkeypatch.setattr(response_panels, "survey_shares", refuse)

    response_panels.response_shift(_shift_frames(("base",)), resolve_question("d_happy"), tmp_path)

    assert sorted(_labelled_rules(captured_axes)) == ["WVS"]


def test_the_pooled_rule_stands_alone_when_the_survey_is_not_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, captured_axes: list[Any]
) -> None:
    monkeypatch.setattr(response_panels, "survey_shares", lambda *_: None)

    response_panels.response_shift(
        _shift_frames(("base", "german")), resolve_question("d_happy"), tmp_path
    )

    assert sorted(_labelled_rules(captured_axes)) == ["WVS"]


def test_an_arm_that_did_not_answer_this_mode_gets_no_home_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, captured_axes: list[Any]
) -> None:
    monkeypatch.setattr(response_panels, "survey_shares", lambda *_: GERMAN_SHARES)

    response_panels.response_shift(
        _shift_frames(("base", "german"), silent="german"),
        resolve_question("d_happy"),
        tmp_path,
    )

    by_mode = [sorted(_labelled_rules([axis])) for axis in captured_axes]
    assert by_mode == [["WVS", "WVS-german"], ["WVS"]]


def test_the_home_rule_is_the_arms_own_green_darkened() -> None:
    assert shaded(arm_tone("german").ink) == "#385b10"
