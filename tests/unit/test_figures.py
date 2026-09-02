from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

from machine_bias_reproduction import figures
from machine_bias_reproduction.config import RunPaths
from machine_bias_reproduction.metrics import QUALITY_LABELS

matplotlib.use("Agg")


def _distances(methods: list[str]) -> pd.DataFrame:
    generator = np.random.default_rng(7)
    return pd.DataFrame(
        [
            {
                "method": method,
                "subpopulation": f"Germany 2017 sub {index}",
                "nEMD": float(value),
                "EMD": float(value) * 3,
                "KL": float(value) * 2,
                "JS": float(value) / 2,
                "MMD": float(value) * 1.5,
            }
            for method in methods
            for index, value in enumerate(generator.uniform(0.02, 0.4, 40))
        ]
    )


def _quality(methods: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"method": method, "quality": label, "percent": 20.0}
            for method in methods
            for label in QUALITY_LABELS
        ]
    )


@pytest.fixture
def run_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunPaths:
    monkeypatch.setattr("machine_bias_reproduction.config.FIGURES_ROOT", tmp_path)
    paths = RunPaths("culture/terra/base", "d_happy")
    paths.figures.mkdir(parents=True, exist_ok=True)
    return paths


def test_density_quality_plate_skips_a_method_the_run_never_produced(
    run_paths: RunPaths,
) -> None:
    methods = ["Linear", "Random_01", "FA"]

    produced = figures._density_quality(
        run_paths, _distances(methods), _quality(["Linear", "Random", "FA"])
    )

    assert [path.suffix for path in produced] == [".png", ".pdf"]
    assert all(path.is_file() for path in produced)


def test_distance_comparison_plate_skips_a_method_the_run_never_produced(
    run_paths: RunPaths,
) -> None:
    produced = figures._distance_comparison(run_paths, _distances(["Linear", "Random_01", "FA"]))

    assert all(path.is_file() for path in produced)


def test_drawable_methods_drops_a_method_with_no_positive_distance() -> None:
    frame = _distances(["Linear", "Random", "NTP", "FA"])
    frame.loc[frame["method"] == "NTP", "nEMD"] = 0.0

    assert figures._drawable_methods(frame) == ["Linear", "Random", "FA"]
