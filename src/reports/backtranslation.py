from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from machine_bias_reproduction.figures import _save
from machine_bias_reproduction.metrics import nemd
from machine_bias_reproduction.plates import category_colors
from machine_bias_reproduction.questions import QUESTIONS

from .load import QuestionData

NEIGHBOURS = (1, 10)


def _distance_matrix(model: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.vstack([nemd(truth, row) for row in model])


def _country(names: pd.Index) -> pd.Series:
    return names.to_series(index=names).str.replace(r"^(.*?) \d.*$", r"\1", regex=True)


def _decade(names: pd.Index) -> pd.Series:
    year = pd.to_numeric(
        names.to_series(index=names).str.replace(r"^.*? (\d+) .*$", r"\1", regex=True)
    )
    return (year // 10 * 10).astype(int).astype(str) + "s"


def _neighbours(data: QuestionData, name: str) -> np.ndarray | None:
    props = data.props(name)
    if props is None:
        return None
    truth = data.wvs.to_numpy(dtype=np.float64)
    values = props.to_numpy(dtype=np.float64)
    if values.shape != truth.shape or not np.isfinite(values).all():
        return None
    return np.argsort(_distance_matrix(values, truth), axis=1)


def table_s9(loaded: dict[str, QuestionData]) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, object]] = []
    orders: dict[str, dict[str, Any]] = {}
    for var, data in loaded.items():
        orders[var] = {}
        for name in data.series:
            order = _neighbours(data, name)
            if order is None:
                continue
            orders[var][name] = order
            own = np.arange(len(data.names))
            for k in NEIGHBOURS:
                reached = np.any(order[:, :k] == own[:, None], axis=1)
                rows.append(
                    {
                        "question": var,
                        "series": name,
                        "k": k,
                        "subpopulations": len(own),
                        "reached_percent": 100.0 * float(reached.mean()),
                    }
                )
    return pd.DataFrame(rows), orders


def _flow(
    data: QuestionData,
    order: np.ndarray,
    grouping: pd.Series,
    k: int = 10,
) -> pd.DataFrame:
    labels = grouping.to_numpy()
    categories = sorted(set(labels))
    index = {category: position for position, category in enumerate(categories)}
    matrix = np.zeros((len(categories), len(categories)), dtype=np.float64)
    for row, source in enumerate(labels):
        for neighbour in order[row, :k]:
            matrix[index[source], index[labels[neighbour]]] += 1.0
    totals = matrix.sum(axis=1, keepdims=True)
    shares = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)
    return pd.DataFrame(shares, index=categories, columns=categories)


def _sankey(flow: pd.DataFrame, axis: object) -> None:
    categories = list(flow.index)
    palette = category_colors(len(categories))
    source_offset = np.zeros(len(categories))
    target_offset = np.zeros(len(categories))
    source_base = np.cumsum([0.0, *[1.1] * (len(categories) - 1)])
    target_base = source_base.copy()
    for source_index, source in enumerate(categories):
        for target_index, target in enumerate(categories):
            share = float(flow.loc[source, target])
            if share <= 0:
                continue
            y0 = source_base[source_index] + source_offset[source_index]
            y1 = target_base[target_index] + target_offset[target_index]
            axis.fill_between(  # type: ignore[attr-defined]
                [0, 1],
                [y0, y1],
                [y0 + share, y1 + share],
                color=palette[source_index],
                alpha=0.55,
                linewidth=0,
            )
            source_offset[source_index] += share
            target_offset[target_index] += share
    axis.set_xlim(-0.35, 1.35)  # type: ignore[attr-defined]
    axis.set_yticks(source_base + 0.5, categories, fontsize=8)  # type: ignore[attr-defined]
    axis.set_xticks([0, 1], ["prompted", "nearest WVS"], fontsize=8)  # type: ignore[attr-defined]
    axis.invert_yaxis()  # type: ignore[attr-defined]
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)  # type: ignore[attr-defined]


def figures_s17_s18(
    loaded: dict[str, QuestionData],
    orders: dict[str, dict[str, Any]],
    destination: Path,
) -> list[Path]:
    produced: list[Path] = []
    for grouping_name, grouping_fn, number in (
        ("country", _country, "S17"),
        ("decade", _decade, "S18"),
    ):
        series_names = sorted({name for per_question in orders.values() for name in per_question})
        for name in series_names:
            questions = [var for var in QUESTIONS if var in orders and name in orders[var]]
            if not questions:
                continue
            figure, axes = plt.subplots(
                1, len(questions), figsize=(3.6 * len(questions), 4.4), squeeze=False
            )
            for axis, var in zip(axes[0], questions, strict=True):
                data = loaded[var]
                _sankey(_flow(data, orders[var][name], grouping_fn(data.names)), axis)
                axis.set_xlabel(QUESTIONS[var].label, fontsize=9)
            figure.tight_layout()
            produced.extend(
                _save(figure, destination / f"Figure-{number}-backnn-{grouping_name}-{name}")
            )
    return produced
