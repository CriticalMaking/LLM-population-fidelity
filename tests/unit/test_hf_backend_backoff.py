from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

torch = pytest.importorskip("torch", reason="requires the culture extra")

from culture.backend import TransformersBackend, _answer_mass  # noqa: E402


def _backend(batch_size: int = 4) -> TransformersBackend:
    backend = object.__new__(TransformersBackend)
    backend._torch = torch
    backend._oom_backoffs = 0
    backend._batch_size = batch_size
    return backend


class _FailsAboveWidth:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.widths: list[int] = []

    def __call__(self, prompts: Sequence[str], **kwargs: Any) -> list[str]:
        if len(prompts) > self.limit:
            raise torch.OutOfMemoryError("CUDA out of memory")
        self.widths.append(len(prompts))
        seeds = kwargs.get("seeds")
        if seeds is not None:
            return [f"{prompt}:{seed}" for prompt, seed in zip(prompts, seeds, strict=True)]
        return list(prompts)


def test_backoff_halves_until_the_batch_fits_and_keeps_order() -> None:
    backend = _backend()
    run = _FailsAboveWidth(limit=2)
    prompts = [f"p{index}" for index in range(8)]

    result = backend._with_oom_backoff(prompts, run)

    assert result == prompts
    assert max(run.widths) <= 2
    assert sum(run.widths) == len(prompts)
    assert backend._oom_backoffs > 0


def test_backoff_splits_seeds_alongside_their_prompts() -> None:
    backend = _backend()
    run = _FailsAboveWidth(limit=1)
    prompts = [f"p{index}" for index in range(4)]
    seeds = [10, 20, 30, 40]

    result = backend._with_oom_backoff(prompts, run, seeds=seeds)

    assert result == ["p0:10", "p1:20", "p2:30", "p3:40"]


def test_backoff_reraises_when_a_single_prompt_cannot_fit() -> None:
    backend = _backend()
    run = _FailsAboveWidth(limit=0)

    with pytest.raises(torch.OutOfMemoryError):
        backend._with_oom_backoff(["only"], run)


def test_backoff_is_not_entered_when_the_batch_fits() -> None:
    backend = _backend()
    run = _FailsAboveWidth(limit=99)

    assert backend._with_oom_backoff([f"p{index}" for index in range(8)], run)
    assert run.widths == [8]
    assert backend._oom_backoffs == 0


def test_mass_probe_chunks_to_the_configured_batch_size() -> None:
    backend = _backend(batch_size=4)
    seen: list[int] = []

    def fake_probe(prompts: Sequence[str]) -> list[float]:
        seen.append(len(prompts))
        return [0.5] * len(prompts)

    backend._ntp_mass_probe = fake_probe  # type: ignore[method-assign]
    masses = TransformersBackend.ntp_mass_probe(backend, [f"p{index}" for index in range(32)])

    assert len(masses) == 32
    assert max(seen) <= 4
    assert sum(seen) == 32


def test_mass_probe_handles_an_empty_sample() -> None:
    assert TransformersBackend.ntp_mass_probe(_backend(), []) == []


def test_answer_mass_sums_the_letters_it_is_given() -> None:
    assert _answer_mass([0.25, 0.25, 0.4, 0.05]) == pytest.approx(0.95)


def test_answer_mass_reads_a_non_finite_row_as_no_answer() -> None:
    assert _answer_mass([float("nan"), 0.2, 0.2, 0.2]) == 0.0
    assert _answer_mass([float("inf"), 0.2, 0.2, 0.2]) == 0.0
    assert _answer_mass([0.0, 0.0, 0.0, 0.0]) == 0.0
