from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

from culture import ranking_panels
from culture.palette import MODES
from culture.registry import CULTURE_MODELS
from machine_bias_reproduction.questions import resolve_question

matplotlib.use("Agg")


def _comparison_row(
    model_key: str, culture: str, mode: str, *, informative: bool
) -> dict[str, object]:
    return {
        "model_key": model_key,
        "model_label": f"label-{model_key}",
        "culture": culture,
        "mode": mode,
        "mean_nEMD": 0.15,
        "median_nEMD": 0.14,
        "informative": informative,
        "valid_answer_rate": 1.0 if informative else 0.0,
    }


def _tables(*, qwen_informative: bool) -> dict[str, pd.DataFrame]:
    return {
        "gemma4_31b": pd.DataFrame(
            [_comparison_row("gemma4_31b", "german", mode, informative=True) for mode in MODES]
        ),
        "qwen3_vl_8b": pd.DataFrame(
            [
                _comparison_row("qwen3_vl_8b", "german", mode, informative=qwen_informative)
                for mode in MODES
            ]
        ),
    }


def test_cross_model_figure_splits_by_mode(tmp_path: Path) -> None:
    produced = ranking_panels.cross_model_figure(
        _tables(qwen_informative=True), resolve_question("d_happy"), tmp_path
    )

    assert sorted({path.stem for path in produced}) == [
        "fig_model_culture_comparison_fa",
        "fig_model_culture_comparison_ntp",
    ]


def test_cross_model_figure_marks_an_arm_that_could_not_answer(
    tmp_path: Path, captured_axes: list[Any]
) -> None:
    ranking_panels.cross_model_figure(
        _tables(qwen_informative=False), resolve_question("d_happy"), tmp_path
    )

    labels = {
        text.get_text()
        for axis in captured_axes
        if axis.get_legend() is not None
        for text in axis.get_legend().get_texts()
    }
    gemma = CULTURE_MODELS["gemma4_31b"].label
    qwen = CULTURE_MODELS["qwen3_vl_8b"].label
    assert f"{qwen} (low mass)" in labels
    assert gemma in labels
    assert f"{gemma} (low mass)" not in labels
