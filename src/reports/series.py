"""The model series the paper reports, and the baselines it compares them to."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

NTP_MODELS = ("GPT-4T", "Llama-3-70B", "Mixtral-8x7B")
FA_MODELS = ("GPT-3", "Llama-3-70B", "Mixtral-8x7B")


@dataclass(frozen=True, slots=True)
class Series:
    """One model under one generation strategy, as ``6-results.R`` names it."""

    name: str
    model: str
    strategy: str

    @property
    def label(self) -> str:
        """Return the display label used on axes and in tables."""
        return self.name


SERIES: dict[str, Series] = {
    **{f"NTP-{model}": Series(f"NTP-{model}", model, "NTP") for model in NTP_MODELS},
    **{f"FA-{model}": Series(f"FA-{model}", model, "FA") for model in FA_MODELS},
}

SERIES_NAMES: tuple[str, ...] = tuple(SERIES)

MAIN_MODELS: tuple[str, ...] = tuple(f"NTP-{model}" for model in NTP_MODELS)
"""``main_models`` in ``6-results.R:41`` — the three NTP series the main text leads with."""

COEFFICIENT_SERIES: tuple[str, ...] = (*MAIN_MODELS, "FA-Llama-3-70B")
"""The series Figures 3 and 6 are drawn for upstream (``6-results.R:757``, ``:1038``)."""

BASELINES = ("Linear", "Random")
SMOKE_SERIES: tuple[str, ...] = ("NTP-Mixtral-8x7B", "FA-Mixtral-8x7B")


def resolve_series(selected: Sequence[str] | None) -> list[Series]:
    """Resolve series names, defaulting to all six; ``all`` expands."""
    names = list(selected) if selected else list(SERIES_NAMES)
    if names == ["all"]:
        names = list(SERIES_NAMES)
    unknown = [name for name in names if name not in SERIES]
    if unknown:
        raise ValueError(f"unknown series: {', '.join(unknown)} (known: {', '.join(SERIES_NAMES)})")
    return [SERIES[name] for name in names]
