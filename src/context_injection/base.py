from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

ContextStrategy = Literal[
    "baseline",
    "raw",
    "semantic",
    "selected_vars",
    "theory_structured",
    "matched_random",
]

CONTEXT_STRATEGIES: tuple[ContextStrategy, ...] = (
    "baseline",
    "raw",
    "semantic",
    "selected_vars",
    "theory_structured",
    "matched_random",
)


@dataclass(frozen=True, slots=True)
class ContextSpec:
    strategy: ContextStrategy = "baseline"
    condition: str | None = None
    theory: str | None = None
    source_variables: tuple[str, ...] = ()
    equivalent_variables: Mapping[str, Sequence[str]] | None = None
    variable_count: int | None = None
    random_seed: int = 0

    @property
    def condition_label(self) -> str:
        return self.condition or self.strategy


@dataclass(frozen=True, slots=True)
class ContextPayload:
    condition: str
    strategy: ContextStrategy
    theory: str | None
    source_variables: tuple[str, ...]
    excluded_variables: tuple[str, ...]
    derived_constructs: Mapping[str, Any]
    context_text: str
    context_hash: str
    context_token_count: int
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "strategy": self.strategy,
            "theory": self.theory,
            "source_variables": list(self.source_variables),
            "excluded_variables": list(self.excluded_variables),
            "derived_constructs": dict(self.derived_constructs),
            "context_text": self.context_text,
            "context_hash": self.context_hash,
            "context_token_count": self.context_token_count,
            "warnings": list(self.warnings),
        }


def resolve_strategy(name: str) -> ContextStrategy:
    if name == "theory_selected":
        name = "selected_vars"
    if name not in CONTEXT_STRATEGIES:
        raise ValueError(f"unknown context strategy: {name!r}")
    return cast(ContextStrategy, name)


def make_payload(
    spec: ContextSpec,
    *,
    source_variables: Sequence[str] = (),
    excluded_variables: Sequence[str] = (),
    context_text: str = "",
    derived_constructs: Mapping[str, Any] | None = None,
    warnings: Sequence[str] = (),
) -> ContextPayload:
    constructs = dict(derived_constructs or {})
    body = {
        "condition": spec.condition_label,
        "strategy": spec.strategy,
        "theory": spec.theory,
        "source_variables": list(source_variables),
        "excluded_variables": list(excluded_variables),
        "derived_constructs": constructs,
        "context_text": context_text,
    }
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return ContextPayload(
        condition=spec.condition_label,
        strategy=spec.strategy,
        theory=spec.theory,
        source_variables=tuple(source_variables),
        excluded_variables=tuple(excluded_variables),
        derived_constructs=constructs,
        context_text=context_text,
        context_hash=digest,
        context_token_count=len(context_text.split()),
        warnings=tuple(warnings),
    )
