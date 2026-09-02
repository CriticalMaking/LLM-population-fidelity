from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from machine_bias_reproduction.comparison import (
    _align_subpopulation_distances,
    country_wave_labels,
    holm_adjust,
    paired_method_inference,
)


def test_country_wave_labels_preserve_multiword_country() -> None:
    subpopulations = pd.Series(
        [
            "United States 2017 Female 25-34 High Working Single",
            "Australia 1995 Male 35-44 Middle Retired Married",
        ]
    )

    assert country_wave_labels(subpopulations).tolist() == [
        "United States 2017",
        "Australia 1995",
    ]


def test_holm_adjust_returns_values_in_original_order() -> None:
    adjusted = holm_adjust([0.04, 0.01, 0.03])

    assert adjusted == pytest.approx([0.06, 0.03, 0.06])


def test_paired_inference_uses_fresh_minus_archived_direction() -> None:
    archived = np.array([0.10, 0.11, 0.12, 0.09, 0.08, 0.13])
    fresh = archived - 0.002 + np.array([0.001, -0.001, 0.0005, -0.0005, 0.001, -0.001])
    pairs = pd.DataFrame(
        {
            "nEMD_archived": archived,
            "nEMD_fresh": fresh,
            "country_wave": ["A 2000", "A 2000", "B 2000", "B 2000", "C 2000", "C 2000"],
        }
    )

    result = paired_method_inference(
        pairs,
        method="NTP",
        alpha=0.05,
        equivalence_margin=0.01,
        permutations=1_000,
        seed=1,
    )

    assert result["mean_change_fresh_minus_archived"] == pytest.approx(-0.002)
    assert result["fresh_better_count"] == 6
    assert result["fresh_worse_count"] == 0
    assert result["practically_equivalent"] is True


def _write_distances(path: Path, columns: list[str], nemd: list[float]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "method": ["NTP", "NTP"],
            "subpopulation": ["Australia 1995 Male", "Mexico 2005 Female"],
            "nEMD": nemd,
        }
    )
    frame.loc[:, columns].to_csv(path / "subpopulation_distances.csv", index=False)


def test_aligned_columns_do_not_depend_on_input_column_order(tmp_path: Path) -> None:
    archived_dir = tmp_path / "archived"
    fresh_dir = tmp_path / "fresh"
    _write_distances(archived_dir, ["subpopulation", "nEMD", "method"], [0.10, 0.20])
    _write_distances(fresh_dir, ["nEMD", "method", "subpopulation"], [0.11, 0.19])

    pairs = _align_subpopulation_distances(archived_dir, fresh_dir, ["NTP"])

    assert pairs.columns.tolist() == [
        "method",
        "subpopulation",
        "nEMD_archived",
        "nEMD_fresh",
        "country_wave",
        "change_fresh_minus_archived",
        "direction",
    ]
