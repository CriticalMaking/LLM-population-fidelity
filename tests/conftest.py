from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pytest

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
