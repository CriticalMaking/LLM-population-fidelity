from __future__ import annotations

from typing import Any, cast

from culture.runner import CultureBackend, effective_modes
from machine_bias_reproduction.prompts import PromptMode


class Local:
    pass


class Served:
    ntp_degrades_to_fa = True


def backend(served: bool) -> CultureBackend:
    return cast(CultureBackend, cast(Any, Served() if served else Local()))


def modes(*names: str) -> list[PromptMode]:
    return [cast(PromptMode, name) for name in names]


def probe(informative: bool | None) -> dict[str, Any]:
    return {"prompts": 32, "mean_mass": 0.0, "informative": informative}


def test_an_informative_probe_keeps_both_modes() -> None:
    assert effective_modes(modes("ntp", "fa"), probe(True), backend(True)) == ["ntp", "fa"]


def test_a_local_model_keeps_both_modes_even_when_the_probe_is_low() -> None:
    assert effective_modes(modes("ntp", "fa"), probe(False), backend(False)) == ["ntp", "fa"]


def test_a_served_model_drops_next_token_probability_when_the_probe_is_low() -> None:
    assert effective_modes(modes("ntp", "fa"), probe(False), backend(True)) == ["fa"]


def test_a_single_requested_mode_is_never_dropped() -> None:
    assert effective_modes(modes("ntp"), probe(False), backend(True)) == ["ntp"]
    assert effective_modes(modes("fa"), probe(False), backend(True)) == ["fa"]


def test_an_empty_probe_changes_nothing() -> None:
    assert effective_modes(modes("ntp", "fa"), None, backend(True)) == ["ntp", "fa"]
    assert effective_modes(modes("ntp", "fa"), probe(None), backend(True)) == ["ntp", "fa"]
