from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import pandas as pd

from machine_bias_reproduction.prompts import CATEGORY_ANSWERS
from machine_bias_reproduction.questions import Question

from .base import ContextPayload, ContextSpec, make_payload
from .leakage import filter_variables
from .theories.inglehart_welzel import (
    DEFAULT_MAPPING,
    InglehartWelzelMapping,
    derived_constructs,
    source_variables,
    warning,
)

RAW_LABELS: dict[str, str] = {
    "i_surveyyear": "Survey year",
    "i_country": "Country",
    "i_age": "Age",
    "i_sex": "Gender",
    "i_education": "Education",
    "i_employment": "Employment",
    "i_marstat": "Marital status",
}


def _value(variable: str, value: Any) -> str | None:
    if pd.isna(value):
        return None
    if variable in CATEGORY_ANSWERS:
        return CATEGORY_ANSWERS[variable].get(str(value), str(value))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _items(row: pd.Series, variables: Sequence[str]) -> list[tuple[str, str, str]]:
    items: list[tuple[str, str, str]] = []
    for variable in variables:
        if variable not in row.index:
            continue
        value = _value(variable, row[variable])
        if value is not None:
            items.append((variable, RAW_LABELS.get(variable, variable), value))
    return items


def _raw_text(title: str, items: Sequence[tuple[str, str, str]]) -> str:
    if not items:
        return ""
    return "\n".join([f"{title}:"] + [f"- {label}: {value}" for _, label, value in items])


def _semantic_text(items: Sequence[tuple[str, str, str]]) -> str:
    if not items:
        return ""
    clauses = [f"{value} for {label.lower()}" for _, label, value in items]
    return "The respondent reports " + "; ".join(clauses) + "."


def _construct_text(constructs: dict[str, object]) -> str:
    if not constructs:
        return ""
    lines = ["Theory-structured respondent context:"]
    for dimension, construct in constructs.items():
        if isinstance(construct, dict):
            label = construct.get("label")
            score = construct.get("score")
            lines.append(f"- {dimension.replace('_', '-')} orientation: {label} ({score})")
    return "\n".join(lines)


def build_context(
    row: pd.Series,
    question: str | Question,
    spec: ContextSpec | None = None,
    *,
    mapping: InglehartWelzelMapping = DEFAULT_MAPPING,
) -> ContextPayload:
    spec = spec or ContextSpec()
    if spec.strategy == "baseline":
        return make_payload(spec)

    requested = spec.source_variables
    if spec.strategy in {"selected_vars", "theory_structured"} and not requested:
        requested = source_variables(mapping)

    kept, excluded = filter_variables(requested, question, spec.equivalent_variables)
    items = _items(row, kept)

    if spec.strategy == "raw":
        return make_payload(
            spec,
            source_variables=kept,
            excluded_variables=excluded,
            context_text=_raw_text("Additional respondent context", items),
        )
    if spec.strategy == "semantic":
        return make_payload(
            spec,
            source_variables=kept,
            excluded_variables=excluded,
            context_text=_semantic_text(items),
        )
    if spec.strategy == "matched_random":
        blocked_kept, blocked_excluded = filter_variables(
            tuple(str(column) for column in row.index),
            question,
            spec.equivalent_variables,
        )
        count = spec.variable_count or len(spec.source_variables)
        rng = random.Random(f"{spec.random_seed}:{question}:{row.get('id', '')}")
        sample = tuple(rng.sample(sorted(blocked_kept), min(count, len(blocked_kept))))
        sample_items = _items(row, sample)
        return make_payload(
            spec,
            source_variables=sample,
            excluded_variables=blocked_excluded,
            context_text=_raw_text("Matched random respondent context", sample_items),
        )
    if spec.strategy == "selected_vars":
        return make_payload(
            spec,
            source_variables=kept,
            excluded_variables=excluded,
            context_text=_raw_text("Selected-vars respondent context", items),
            warnings=warning(mapping),
        )
    if spec.strategy == "theory_structured":
        constructs = derived_constructs(row, kept, mapping)
        text = _construct_text(constructs)
        return make_payload(
            spec,
            source_variables=kept,
            excluded_variables=excluded,
            context_text=text,
            derived_constructs=constructs,
            warnings=warning(mapping),
        )
    raise ValueError(f"unknown context strategy: {spec.strategy!r}")
