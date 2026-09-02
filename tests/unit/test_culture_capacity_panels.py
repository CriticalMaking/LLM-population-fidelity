from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import pytest

from culture import capacity_panels
from machine_bias_reproduction.questions import resolve_question

matplotlib.use("Agg")


@pytest.mark.parametrize("informative", [False, 0])
def test_low_capacity_covers_both_the_bool_and_its_csv_round_trip(informative: object) -> None:
    assert capacity_panels.is_low_capacity(pd.Series({"informative": informative}))


@pytest.mark.parametrize("informative", [True, 1])
def test_an_informative_arm_is_not_marked(informative: object) -> None:
    assert not capacity_panels.is_low_capacity(pd.Series({"informative": informative}))


def _capacity(valid: int, failed: int, *, distinct: float | None = 28.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "mode": "NTP",
                "prompts": 100,
                "valid": valid,
                "invalid": 100 - valid,
                "failed": 0,
                "mean_valid_mass": 7e-30,
                "distinct_distributions": distinct,
                "tied_answers": 57.0,
            },
            {
                "mode": "FA",
                "prompts": 100,
                "valid": 0,
                "invalid": 0,
                "failed": failed,
                "mean_valid_mass": None,
                "distinct_distributions": None,
                "tied_answers": None,
            },
        ]
    )


def test_a_run_that_placed_nothing_still_gets_a_plate(tmp_path: Path) -> None:
    produced = capacity_panels.run_capacity_figure(
        _capacity(valid=0, failed=100),
        {"subpopulations_retained": 0, "subpopulations_total": 687},
        tmp_path,
    )

    assert sorted(path.name for path in produced) == [
        "fig_run_capacity.pdf",
        "fig_run_capacity.png",
    ]


def test_the_capacity_plate_needs_something_to_report(tmp_path: Path) -> None:
    assert capacity_panels.run_capacity_figure(pd.DataFrame(), {}, tmp_path) == []
    assert not list(tmp_path.iterdir())


def test_the_unplaced_plates_are_skipped_for_a_run_that_placed_subpopulations() -> None:
    from culture.figures import _unplaced_figures
    from culture.registry import CULTURE_MODELS

    tables = {
        "coverage": {"subpopulations_retained": 687, "subpopulations_total": 687},
        "capacity": _capacity(valid=100, failed=0),
    }

    assert (
        _unplaced_figures(
            CULTURE_MODELS["gemma4_31b"], "german", tables, resolve_question("d_happy")
        )
        == []
    )
