from __future__ import annotations

import pandas as pd
import pytest

from machine_bias_reproduction.analysis import run_analysis
from machine_bias_reproduction.config import paths_for
from machine_bias_reproduction.provenance import archive_inventory, verify_upstream
from machine_bias_reproduction.verification import verify_results


@pytest.mark.integration
def test_complete_archive_inventory_and_consumed_inputs() -> None:
    inventory = archive_inventory()
    assert inventory["total_entries"] == 1_580_438
    assert inventory["substantive_files"] == 790_118
    assert inventory["substantive_uncompressed_bytes"] == 253_479_248
    result = verify_upstream(full=False)
    assert result["ok"] is True


@pytest.mark.integration
def test_archived_analysis_reproduces_numeric_checkpoints() -> None:
    run_analysis("archived", "d_happy")
    result = verify_results("archived")
    assert result["ok"] is True
    paths = paths_for("archived", "d_happy")
    checkpoints = pd.read_csv(paths.outputs / "paper_regression_checkpoints.csv")
    assert checkpoints["passed"].all()
    fit = pd.read_csv(paths.outputs / "regression_fit.csv")
    ntp = fit[fit["mode"] == "ntp"].set_index("model")
    assert ntp.loc["social", "adjusted_r_squared"] == pytest.approx(0.387, abs=0.005)
    assert ntp.loc["center", "adjusted_r_squared"] == pytest.approx(0.829, abs=0.002)
    assert ntp.loc["full", "adjusted_r_squared"] == pytest.approx(0.858, abs=0.002)
