from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from machine_bias_reproduction.questions import Question

from .base import ContextPayload, ContextSpec
from .strategies import build_context
from .theories.inglehart_welzel import DEFAULT_MAPPING, InglehartWelzelMapping

FIRST_PILOT_CONDITIONS: tuple[str, ...] = ("C0", "C1", "C2", "C3", "C4")


def first_pilot_specs(
    *,
    raw_variables: Sequence[str],
    selected_variables: Sequence[str],
    equivalent_variables: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, ContextSpec]:
    theory = "inglehart_welzel"
    raw = tuple(raw_variables)
    selected = tuple(selected_variables)
    return {
        "C0": ContextSpec("baseline", condition="C0_baseline"),
        "C1": ContextSpec(
            "raw",
            condition="C1_raw",
            source_variables=raw,
            equivalent_variables=equivalent_variables,
        ),
        "C2": ContextSpec(
            "semantic",
            condition="C2_semantic",
            source_variables=raw,
            equivalent_variables=equivalent_variables,
        ),
        "C3": ContextSpec(
            "selected_vars",
            condition="C3_selected_vars",
            theory=theory,
            source_variables=selected,
            equivalent_variables=equivalent_variables,
        ),
        "C4": ContextSpec(
            "theory_structured",
            condition="C4_theory_structured",
            theory=theory,
            source_variables=selected,
            equivalent_variables=equivalent_variables,
        ),
    }


def build_first_pilot(
    row: pd.Series,
    question: str | Question,
    *,
    raw_variables: Sequence[str],
    selected_variables: Sequence[str],
    equivalent_variables: Mapping[str, Sequence[str]] | None = None,
    mapping: InglehartWelzelMapping = DEFAULT_MAPPING,
) -> dict[str, ContextPayload]:
    specs = first_pilot_specs(
        raw_variables=raw_variables,
        selected_variables=selected_variables,
        equivalent_variables=equivalent_variables,
    )
    payloads = {
        name: build_context(row, question, spec, mapping=mapping) for name, spec in specs.items()
    }
    validate_first_pilot(payloads)
    return payloads


def validate_first_pilot(payloads: Mapping[str, ContextPayload]) -> None:
    missing = set(FIRST_PILOT_CONDITIONS) - set(payloads)
    if missing:
        raise ValueError(f"missing context conditions: {', '.join(sorted(missing))}")
    if payloads["C0"].context_text:
        raise ValueError("C0 must not add injected context")
    if payloads["C1"].source_variables != payloads["C2"].source_variables:
        raise ValueError("C1 and C2 must use the same source variables")
    if payloads["C3"].source_variables != payloads["C4"].source_variables:
        raise ValueError("C3 and C4 must use the same source variables")
