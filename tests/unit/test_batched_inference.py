from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import machine_bias_reproduction.config as config
from machine_bias_reproduction.config import GLOBAL_SEED, RunPaths, replicate_seed
from machine_bias_reproduction.inference import (
    EVENT_LOG_NAME,
    RECORD_SCHEMA_VERSION,
    RunContext,
    audit_traces,
    consolidate_fa,
    consolidate_ntp,
    generate_records_batched,
)
from machine_bias_reproduction.io_utils import sha256_text
from machine_bias_reproduction.prompts import PromptMode, PromptRecord

ANSWERS = ("A. Very happy", "B. Quite happy", "C. Not very happy", "D. Not at all happy")


class FakeBatchBackend:
    batch_size = 3

    def __init__(self, *, fail_until: int = 0, always_fail: bool = False) -> None:
        self.fail_until = fail_until
        self.always_fail = always_fail
        self.batches: list[int] = []
        self.seeds_seen: list[int | None] = []
        self.calls_per_prompt: dict[str, int] = {}

    def token_length(self, prompt: str) -> int:
        return len(prompt)

    def describe(self) -> dict[str, Any]:
        return {"name": "fake-batch", "version": "0.0.0", "batch_size": self.batch_size}

    def ntp_batch(self, prompts: Sequence[str]) -> list[dict[str, float]]:
        self.batches.append(len(prompts))
        results = []
        for prompt in prompts:
            weight = (len(prompt) % 4) / 100.0
            results.append({"mass": 0.9 + weight, "A": 0.1, "B": 0.7, "C": 0.15, "D": 0.05})
        return results

    def ntp(self, prompt: str) -> dict[str, float]:
        return self.ntp_batch([prompt])[0]

    def full_answer_batch(
        self,
        prompts: Sequence[str],
        *,
        seeds: Sequence[int | None],
    ) -> list[str]:
        self.batches.append(len(prompts))
        self.seeds_seen.extend(seeds)
        candidates = []
        for prompt, seed in zip(prompts, seeds, strict=True):
            attempt = self.calls_per_prompt.get(prompt, 0)
            self.calls_per_prompt[prompt] = attempt + 1
            if self.always_fail or attempt < self.fail_until:
                candidates.append("nonsense")
                continue
            digest = int(sha256_text(f"{prompt}|{seed}")[:8], 16)
            candidates.append(ANSWERS[digest % len(ANSWERS)])
        return candidates

    def full_answer(self, prompt: str, *, seed: int | None) -> str:
        return self.full_answer_batch([prompt], seeds=[seed])[0]


def _trace() -> RunContext:
    return RunContext(
        run_id="20240110T000000Z-abcdef01",
        started_at="2024-01-10T00:00:00+00:00",
        model={"adapter_sha256": "1" * 64, "sha256": "1" * 64},
        backend={"name": "fake-batch", "version": "0.0.0"},
        code_revision="deadbee",
    )


def _paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunPaths:
    monkeypatch.setattr(config, "OUTPUTS_ROOT", tmp_path / "outputs")
    paths = RunPaths("culture/qwen3_vl_8b/german", "d_happy")
    paths.ensure()
    return paths


def _records(mode: PromptMode, count: int) -> list[PromptRecord]:
    return [
        PromptRecord(f"{mode}-{index}", f"profile-{index}", mode, f"Question {index}?\nAnswer:\n")
        for index in range(count)
    ]


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload: dict[str, Any] = json.load(stream)
    return payload


def test_batched_ntp_writes_one_traced_record_per_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    records = _records("ntp", 7)
    backend = FakeBatchBackend()

    counts = generate_records_batched(
        records, backend, paths, trace=_trace(), batch_size=3, progress_every=0
    )

    assert counts == {"generated": 7, "reused": 0, "reused_untraced": 0, "failed": 0}
    assert backend.batches == [3, 3, 1]
    written = sorted((paths.raw / "ntp").glob("*.json"))
    assert len(written) == 7
    payload = _read(written[0])
    assert payload["schema_version"] == RECORD_SCHEMA_VERSION
    assert payload["trace"]["run_id"] == "20240110T000000Z-abcdef01"
    assert payload["trace"]["sampling"]["batch_size"] == 3
    assert payload["trace"]["sampling"]["max_tokens"] == 1
    assert payload["prompt"]["sha256"] == sha256_text(payload["prompt"]["text"])

    csv_path = tmp_path / "ntp.csv"
    consolidate_ntp(records, paths, csv_path)
    assert csv_path.is_file()

    _, summary = audit_traces({"ntp": records}, paths)
    assert summary["ok"] is True
    assert summary["indexed"] == summary["traced"] == 7


def test_batched_generation_resumes_without_regenerating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    records = _records("ntp", 5)
    backend = FakeBatchBackend()
    generate_records_batched(
        records[:2], backend, paths, trace=_trace(), batch_size=3, progress_every=0
    )
    backend.batches.clear()

    counts = generate_records_batched(
        records, backend, paths, trace=_trace(), batch_size=3, progress_every=0
    )
    assert counts == {"generated": 3, "reused": 2, "reused_untraced": 0, "failed": 0}
    assert sum(backend.batches) == 3


def test_fa_answers_do_not_depend_on_batch_composition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records("fa", 6)

    def run(batch_size: int, directory: str) -> list[str]:
        monkeypatch.setattr(config, "OUTPUTS_ROOT", tmp_path / directory)
        paths = RunPaths("culture/qwen3_vl_8b/german", "d_happy")
        paths.ensure()
        generate_records_batched(
            records,
            FakeBatchBackend(),
            paths,
            trace=_trace(),
            batch_size=batch_size,
            progress_every=0,
        )
        csv_path = tmp_path / f"{directory}.csv"
        consolidate_fa(records, paths, csv_path)
        return csv_path.read_text(encoding="utf-8").splitlines()

    assert run(2, "outputs-a") == run(5, "outputs-b")


def test_a_replicate_resamples_every_prompt_under_its_own_seed_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records("fa", 3)

    def seeds(directory: str, global_seed: int = GLOBAL_SEED) -> list[int | None]:
        monkeypatch.setattr(config, "OUTPUTS_ROOT", tmp_path / directory)
        paths = RunPaths("culture/qwen3_vl_8b/german", "d_happy")
        paths.ensure()
        backend = FakeBatchBackend()
        generate_records_batched(
            records,
            backend,
            paths,
            trace=_trace(),
            batch_size=3,
            progress_every=0,
            global_seed=global_seed,
        )
        return backend.seeds_seen

    first = seeds("first")
    assert seeds("first-again", global_seed=replicate_seed(1)) == first
    second = seeds("second", global_seed=replicate_seed(2))
    assert len(second) == len(first) == 3
    assert not set(second) & set(first)
    assert all(seed is not None for seed in second)


def test_fa_retries_are_batched_and_fully_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    records = _records("fa", 4)
    backend = FakeBatchBackend(fail_until=1)

    counts = generate_records_batched(
        records, backend, paths, trace=_trace(), batch_size=4, progress_every=0
    )
    assert counts["generated"] == 4
    assert counts["failed"] == 0

    payload = _read(next((paths.raw / "fa").glob("*.json")))
    assert payload["attempt_count"] == 2
    assert [attempt["accepted"] for attempt in payload["attempts"]] == [False, True]
    seeds = [attempt["seed"] for attempt in payload["attempts"]]
    assert all(seed is not None for seed in seeds)
    assert len(set(seeds)) == 2
    assert payload["seed"] == seeds[0]
    assert payload["trace"]["sampling"]["temperature"] == 0.7
    assert payload["trace"]["sampling"]["max_retries"] == 50


def test_legacy_unseeded_fa_reproduces_the_papers_unrepeatable_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    records = _records("fa", 2)
    backend = FakeBatchBackend()
    generate_records_batched(
        records,
        backend,
        paths,
        trace=_trace(),
        batch_size=2,
        legacy_unseeded_fa=True,
        progress_every=0,
    )
    assert backend.seeds_seen == [None, None]
    payload = _read(next((paths.raw / "fa").glob("*.json")))
    assert payload["seed"] is None


def test_exhausted_fa_retries_fail_the_record_and_log_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    records = _records("fa", 2)

    counts = generate_records_batched(
        records,
        FakeBatchBackend(always_fail=True),
        paths,
        trace=_trace(),
        batch_size=2,
        progress_every=0,
    )
    assert counts["failed"] == 2
    assert counts["generated"] == 0
    failures = sorted((paths.raw / "fa").glob("*.failed.json"))
    assert len(failures) == 2
    assert "no valid FA answer" in _read(failures[0])["error"]

    events = [
        json.loads(line)
        for line in (paths.logs / EVENT_LOG_NAME).read_text(encoding="utf-8").splitlines()
    ]
    assert {event["status"] for event in events} == {"failed"}

    _, summary = audit_traces({"fa": records}, paths)
    assert summary["ok"] is False
    assert len(summary["failed_records"]) == 2


def test_a_backend_error_fails_the_batch_without_losing_the_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    records = _records("ntp", 2)

    class Broken(FakeBatchBackend):
        def ntp_batch(self, prompts: Sequence[str]) -> list[dict[str, float]]:
            raise RuntimeError("cuda out of memory")

    counts = generate_records_batched(
        records, Broken(), paths, trace=_trace(), batch_size=2, progress_every=0
    )
    assert counts["failed"] == 2
    failure = _read(next((paths.raw / "ntp").glob("*.failed.json")))
    assert failure["error"] == "RuntimeError: cuda out of memory"
    assert failure["trace"]["run_id"] == "20240110T000000Z-abcdef01"


def test_batched_generation_refuses_to_mix_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    mixed = [*_records("ntp", 1), *_records("fa", 1)]
    with pytest.raises(ValueError, match="one mode at a time"):
        generate_records_batched(
            mixed, FakeBatchBackend(), paths, trace=_trace(), batch_size=2, progress_every=0
        )


@pytest.mark.parametrize(
    ("question", "candidate", "expected"),
    [
        ("d_happy", "A. Very happy", "A. Very happy"),
        ("d_happy", "1. Very happy", "A. Very happy"),
        ("d_happy", "a. very happy", "A. Very happy"),
        ("d_happy", "A. Very happy\nand some rambling", "A. Very happy"),
        ("d_happy", "  B. Quite happy  ", "B. Quite happy"),
        ("d_happy", "1", None),
        ("d_happy", "C.C.C.C.", None),
        ("d_happy", "edListedListed", None),
        ("d_happy", "", None),
        ("d_trust", "B. Need to be very careful", "B. Need to be very careful"),
        ("d_religiousp", "G. Never, practically never", "G. Never, practically never"),
        ("d_polpos", "7", "7"),
        ("d_polpos", "10", "10"),
        ("d_polpos", "11", None),
        ("d_polpos", "A. Very happy", None),
    ],
)
def test_paper_answer_parser(question: str, candidate: str, expected: str | None) -> None:
    from machine_bias_reproduction.inference import parse_full_answer

    assert parse_full_answer(candidate, question) == expected


def test_fa_batch_uses_the_injected_parser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from machine_bias_reproduction.inference import answer_parser

    paths = _paths(tmp_path, monkeypatch)
    records = _records("fa", 2)

    class TrustBackend(FakeBatchBackend):
        def full_answer_batch(
            self, prompts: Sequence[str], *, seeds: Sequence[int | None]
        ) -> list[str]:
            self.batches.append(len(prompts))
            return ["A. Most people can be trusted" for _ in prompts]

    counts = generate_records_batched(
        records,
        TrustBackend(),
        paths,
        trace=_trace(),
        batch_size=2,
        progress_every=0,
        parse=answer_parser("d_trust"),
    )
    assert counts["generated"] == 2
    payload = _read(next((paths.raw / "fa").glob("*.json")))
    assert payload["result"]["answer"] == "A. Most people can be trusted"


def test_length_bucketing_does_not_change_which_prompts_pair_up_in_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path, monkeypatch)
    records = [
        PromptRecord(f"ntp-{index}", f"profile-{index}", "ntp", "x" * (10 - index) + "\n")
        for index in range(6)
    ]

    class LengthCodedBackend(FakeBatchBackend):
        def ntp_batch(self, prompts: Sequence[str]) -> list[dict[str, float]]:
            return [
                {"mass": float(len(prompt)), "A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
                for prompt in prompts
            ]

    generate_records_batched(
        records,
        LengthCodedBackend(),
        paths,
        trace=_trace(),
        batch_size=2,
        progress_every=0,
    )
    for record in records:
        stored = _read(paths.raw / "ntp" / f"{sha256_text(record.prompt_id)[:24]}.json")
        assert stored["prompt"]["text"] == record.text
        assert stored["result"]["mass"] == float(len(record.text))
