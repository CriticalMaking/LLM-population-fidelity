from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from culture import served_smoke
from culture.registry import CULTURE_MODELS
from machine_bias_reproduction.prompts import PromptRecord
from machine_bias_reproduction.questions import QUESTIONS


class StubBackend:
    def __init__(self, replies: Sequence[str], masses: Sequence[float] | None = None) -> None:
        self.replies = list(replies)
        self.masses = list(masses) if masses is not None else None
        self.seeds: list[int | None] = []
        self.culture: str | None = None

    def set_culture(self, culture: str) -> None:
        self.culture = culture

    def ntp_mass_probe(self, prompts: Sequence[str]) -> list[float]:
        if self.masses is None:
            return [0.0 for _ in prompts]
        return self.masses[: len(prompts)]

    def full_answer_batch(
        self,
        prompts: Sequence[str],
        *,
        seeds: Sequence[int | None],
    ) -> list[str]:
        self.seeds.extend(seeds)
        texts = [self.replies.pop(0) if self.replies else "" for _ in prompts]
        self.last_reasoning_tokens = [128 for _ in texts]
        return texts

    def describe(self) -> dict[str, Any]:
        return {
            "model_id": "stub-id",
            "logprobs_supported": True,
            "reasoning_effort": "medium",
            "reasoning_effort_supported": True,
            "mean_reasoning_tokens": 128.0,
            "fa_temperature": 0.7,
            "fa_max_new_tokens": 300,
        }


def records(count: int) -> list[PromptRecord]:
    return [
        PromptRecord(prompt_id=f"p{index}", profile="Germany", mode="fa", text="Answer:")
        for index in range(count)
    ]


@pytest.mark.parametrize(
    ("reply", "verdict"),
    [
        ("B. Quite happy", "answered"),
        ("Answer: B. Quite happy", "wrapped"),
        ("**B. Quite happy**", "wrapped"),
        ("The answer is B. Quite happy.", "wrapped"),
        ("Quite happy", "wrapped"),
        ("B", "wrapped"),
        ("Please choose one:\n\nA. Very happy\nB. Quite happy", "deflected"),
        ("Answer: Cannot be determined from the information provided.", "refused"),
        ("I don't have enough information to determine that.", "refused"),
        ("Guten Tag!", "unreadable"),
        ("", "unreadable"),
    ],
)
def test_the_four_ways_a_served_reply_lands(reply: str, verdict: str) -> None:
    assert served_smoke.classify(reply, "d_happy") == verdict


def test_the_paper_parser_and_the_tolerant_one_agree_on_the_bare_form() -> None:
    assert served_smoke.lenient_answer("B. Quite happy", "d_happy") == "B. Quite happy"
    assert served_smoke.lenient_answer("Answer: 7", "d_polpos") == "7"
    assert served_smoke.lenient_answer("Please choose one", "d_happy") is None


def test_a_wrapped_answer_never_counts_toward_the_strict_rate() -> None:
    backend = StubBackend(["Answer: B. Quite happy"] * 3)
    result = served_smoke.probe_model(
        CULTURE_MODELS["terra"],
        QUESTIONS["d_happy"],
        records(1),
        backend=backend,
    )

    assert result["answered"] == 0
    assert result["wrapped"] == 1
    assert result["strict_rate"] == 0.0
    assert result["tolerant_rate"] == 1.0
    assert backend.culture == "base"


def test_a_prompt_stops_retrying_once_the_paper_format_lands() -> None:
    backend = StubBackend(["Please choose one:", "B. Quite happy", "B. Quite happy"])
    result = served_smoke.probe_model(
        CULTURE_MODELS["sol"],
        QUESTIONS["d_happy"],
        records(1),
        backend=backend,
    )

    assert result["answered"] == 1
    assert result["first_attempt"] == 0
    assert len(backend.seeds) == 2


def test_the_same_prompt_gets_a_different_seed_each_attempt() -> None:
    backend = StubBackend(["Guten Tag!"] * 3)
    served_smoke.probe_model(
        CULTURE_MODELS["luna"],
        QUESTIONS["d_happy"],
        records(1),
        backend=backend,
    )

    assert len(backend.seeds) == 3
    assert len(set(backend.seeds)) == 3


def test_the_smoke_writes_beside_the_runs_and_never_into_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replies = {
        "luna": ["Answer: B. Quite happy"] * 3,
        "terra": ["B. Quite happy"],
        "sol": ["Cannot be determined from the information provided."] * 3,
    }
    monkeypatch.setattr(
        served_smoke,
        "probe_records",
        lambda wvs, question, prompts: records(1),
    )
    monkeypatch.setattr(
        served_smoke,
        "OpenAIBackend",
        lambda model, question, **kwargs: StubBackend(replies[model.key]),
    )

    result = served_smoke.run_smoke(
        [CULTURE_MODELS[key] for key in ("luna", "terra", "sol")],
        "d_happy",
        pd.DataFrame(),
        root=tmp_path,
    )

    directory = tmp_path / "d_happy"
    assert Path(result["directory"]) == directory
    summary = pd.read_csv(directory / "served_smoke.csv")
    assert list(summary["model"]) == ["luna", "terra", "sol"]
    assert list(summary["answered"]) == [0, 1, 0]
    assert list(summary["wrapped"]) == [1, 0, 0]
    assert list(summary["refused"]) == [0, 0, 1]
    assert (directory / "raw" / "luna_attempts.csv").is_file()
    assert not list(tmp_path.glob("**/capacity.csv"))


def test_every_model_is_compared_at_the_same_reasoning_effort() -> None:
    backend = StubBackend(["B. Quite happy"])
    result = served_smoke.probe_model(
        CULTURE_MODELS["terra"],
        QUESTIONS["d_happy"],
        records(1),
        backend=backend,
    )

    assert result["reasoning_effort"] == "medium"
    assert result["reasoning_effort_supported"] is True
    assert result["mean_reasoning_tokens"] == 128.0


def test_the_attempt_trail_carries_what_each_reply_cost_in_reasoning() -> None:
    backend = StubBackend(["Please choose one:", "B. Quite happy"])
    result = served_smoke.probe_model(
        CULTURE_MODELS["sol"],
        QUESTIONS["d_happy"],
        records(1),
        backend=backend,
    )

    assert [entry["reasoning_tokens"] for entry in result["trail"]] == [128, 128]
