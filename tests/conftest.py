from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pytest

from machine_bias_reproduction.config import ARCHIVE_PATH, UPSTREAM_DATA

matplotlib.use("Agg")
from matplotlib import pyplot as plt


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def captured_axes(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    drawn: list[Any] = []
    original = plt.subplots

    def capture(*args: Any, **kwargs: Any) -> Any:
        figure, axis = original(*args, **kwargs)
        drawn.append(axis)
        return figure, axis

    monkeypatch.setattr(plt, "subplots", capture)
    return drawn


UPSTREAM_SENTINELS = (
    ARCHIVE_PATH,
    UPSTREAM_DATA / "WVS" / "wvs-data.csv",
    UPSTREAM_DATA / "LLM-outputs" / "csv" / "NTP-Mixtral-8x7B-d_happy.csv",
    UPSTREAM_DATA / "prompts" / "FA" / "d_happy",
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    missing = [str(path) for path in UPSTREAM_SENTINELS if not path.exists()]
    if not missing:
        return
    reason = f"upstream replication package incomplete, missing {', '.join(missing)}"
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(pytest.mark.skip(reason=reason))
