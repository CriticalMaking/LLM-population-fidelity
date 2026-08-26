from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DIMENSION_LABELS: dict[str, tuple[str, str, str]] = {
    "traditional_secular": ("traditional", "mixed traditional-secular", "secular-rational"),
    "survival_self_expression": (
        "survival-oriented",
        "mixed survival-self-expression",
        "self-expression-oriented",
    ),
}


@dataclass(frozen=True, slots=True)
class TheoryItem:
    variable: str
    dimension: str
    scores: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class InglehartWelzelMapping:
    source_variables: tuple[str, ...] = ()
    items: tuple[TheoryItem, ...] = ()


DEFAULT_MAPPING = InglehartWelzelMapping()


def source_variables(mapping: InglehartWelzelMapping = DEFAULT_MAPPING) -> tuple[str, ...]:
    if mapping.source_variables:
        return mapping.source_variables
    seen: list[str] = []
    for item in mapping.items:
        if item.variable not in seen:
            seen.append(item.variable)
    return tuple(seen)


def load_mapping(path: Path) -> InglehartWelzelMapping:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = tuple(
        TheoryItem(
            variable=str(item["variable"]),
            dimension=str(item["dimension"]),
            scores={str(key): float(value) for key, value in item["scores"].items()},
        )
        for item in data.get("items", ())
    )
    return InglehartWelzelMapping(
        source_variables=tuple(data.get("source_variables", ())),
        items=items,
    )


def warning(mapping: InglehartWelzelMapping = DEFAULT_MAPPING) -> tuple[str, ...]:
    if mapping.source_variables or mapping.items:
        return ()
    return (
        "inglehart_welzel has no verified WVS mapping in this checkout; "
        "provide source_variables after the metadata audit",
    )


def _label(dimension: str, score: float) -> str:
    low, middle, high = DIMENSION_LABELS.get(dimension, ("low", "mixed", "high"))
    if score <= -0.33:
        return low
    if score >= 0.33:
        return high
    return middle


def derived_constructs(
    row: pd.Series,
    variables: Sequence[str],
    mapping: InglehartWelzelMapping = DEFAULT_MAPPING,
) -> dict[str, object]:
    selected = set(variables)
    grouped: dict[str, list[tuple[str, float]]] = {}
    for item in mapping.items:
        if (
            item.variable not in selected
            or item.variable not in row.index
            or pd.isna(row[item.variable])
        ):
            continue
        score = item.scores.get(str(row[item.variable]))
        if score is not None:
            grouped.setdefault(item.dimension, []).append((item.variable, score))
    return {
        dimension: {
            "score": round(mean, 3),
            "label": _label(dimension, mean),
            "source_variables": [variable for variable, _ in scored],
        }
        for dimension, scored in grouped.items()
        for mean in [sum(score for _, score in scored) / len(scored)]
    }