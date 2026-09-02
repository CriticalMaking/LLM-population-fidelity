from __future__ import annotations

import pandas as pd
import pytest

from machine_bias_reproduction.config import PROJECT_ROOT
from machine_bias_reproduction.data import archived_fa, archived_ntp, prepare_data
from machine_bias_reproduction.r_rng import RMersenneTwister, center_holdout_indices


def test_r_set_seed_one_matches_known_uniform_prefix() -> None:
    generator = RMersenneTwister(1)
    draws = [generator.uniform() for _ in range(5)]
    assert draws == pytest.approx(
        [0.2655086631, 0.3721238996, 0.5728533634, 0.9082077900, 0.2016819310]
    )


def test_center_holdouts_match_golden_subpopulation_ids() -> None:
    data = prepare_data(archived_ntp("d_happy"), archived_fa("d_happy"), "d_happy")
    golden = pd.read_csv(PROJECT_ROOT / "tests" / "golden" / "center_holdouts.csv")
    for mode in ("ntp", "fa"):
        indices = center_holdout_indices(len(data.names), mode)
        actual = pd.DataFrame(
            {
                "mode": mode,
                "zero_based_index": indices,
                "subpopulation": data.names.take(indices),
            }
        )
        expected = golden[golden["mode"] == mode].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)
