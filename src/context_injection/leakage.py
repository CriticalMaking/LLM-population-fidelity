from __future__ import annotations

from collections.abc import Mapping, Sequence

from machine_bias_reproduction.questions import Question, resolve_question

DEFAULT_EQUIVALENTS: dict[str, tuple[str, ...]] = {
    "d_happy": ("d_happy",),
    "d_polpos": ("d_polpos",),
    "d_religiousp": ("d_religiousp",),
    "d_trust": ("d_trust",),
}


def blocked_variables(
    question: str | Question,
    equivalents: Mapping[str, Sequence[str]] | None = None,
) -> set[str]:
    target = resolve_question(question).var
    blocked = set((equivalents or DEFAULT_EQUIVALENTS).get(target, ()))
    blocked.add(target)
    return blocked


def filter_variables(
    variables: Sequence[str],
    question: str | Question,
    equivalents: Mapping[str, Sequence[str]] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    blocked = blocked_variables(question, equivalents)
    kept = tuple(variable for variable in variables if variable not in blocked)
    removed = tuple(sorted(blocked))
    return kept, removed
