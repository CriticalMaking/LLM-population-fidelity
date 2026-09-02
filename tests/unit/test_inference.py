from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import machine_bias_reproduction.config as config
from machine_bias_reproduction.config import RunPaths
from machine_bias_reproduction.inference import (
    EVENT_LOG_NAME,
    RECORD_SCHEMA_VERSION,
    RunContext,
    audit_traces,
    consolidate_fa,
    consolidate_ntp,
    generate_records,
    parse_full_answer,
)
from machine_bias_reproduction.io_utils import sha256_text
from machine_bias_reproduction.prompts import PromptRecord


class FakeBackend:
    def __init__(self) -> None:
        self.fa_seeds: list[int | None] = []
        self.fa_calls = 0

    def ntp(self, prompt: str) -> dict[str, float]:
        assert prompt.endswith("\n")
        return {"mass": 0.98, "A": 0.1, "B": 0.7, "C": 0.15, "D": 0.05}

    def full_answer(self, prompt: str, *, seed: int | None) -> str:
        assert prompt.endswith("\n")
        self.fa_seeds.append(seed)
        self.fa_calls += 1
        return "not valid" if self.fa_calls == 1 else "A. Very happy\nignored"

    def describe(self) -> dict[str, Any]:
        return {"name": "fake", "version": "0.0.0", "n_ctx": 512}


def _trace(run_id: str = "20240110T000000Z-abcdef01") -> RunContext:
    return RunContext(
        run_id=run_id,
        started_at="2024-01-10T00:00:00+00:00",
        model={"filename": "model.gguf", "sha256": "0" * 64},
        backend={"name": "fake", "version": "0.0.0"},
        code_revision="deadbee",
    )


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload: dict[str, Any] = json.load(stream)
    return payload


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        ("A. Very happy", "A. Very happy"),
        ("1. Very happy", "A. Very happy"),
        (" b. quite happy ", "B. Quite happy"),
        ("D. Not at all happy\nextra", "D. Not at all happy"),
        ("A", None),
        ("not an answer", None),
    ],
)
def test_full_answer_parser(candidate: str, expected: str | None) -> None:
    assert parse_full_answer(candidate) == expected


def test_atomic_generation_resume_and_consolidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "OUTPUTS_ROOT", tmp_path / "outputs")
    paths = RunPaths("fresh", "d_happy")
    paths.ensure()
    ntp_record = PromptRecord("profile", "profile", "ntp", "Answer:\n")
    fa_record = PromptRecord("respondent", "profile", "fa", "Answer:\n")
    backend = FakeBackend()
    trace = _trace()

    ntp_counts = generate_records([ntp_record], backend, paths, trace=trace)
    fa_counts = generate_records([fa_record], backend, paths, trace=trace)
    assert ntp_counts == {"generated": 1, "reused": 0, "reused_untraced": 0, "failed": 0}
    assert fa_counts == {"generated": 1, "reused": 0, "reused_untraced": 0, "failed": 0}
    assert backend.fa_seeds[0] is not None
    assert backend.fa_seeds[1] is None

    resumed = generate_records([ntp_record], backend, paths, trace=trace)
    assert resumed == {"generated": 0, "reused": 1, "reused_untraced": 0, "failed": 0}

    ntp_csv = tmp_path / "ntp.csv"
    fa_csv = tmp_path / "fa.csv"
    consolidate_ntp([ntp_record], paths, ntp_csv)
    consolidate_fa([fa_record], paths, fa_csv)
    assert pd.read_csv(ntp_csv).iloc[0]["mass"] == pytest.approx(0.98)
    assert pd.read_csv(fa_csv).iloc[0]["d_happy"] == "A. Very happy"


def test_every_record_carries_a_complete_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "OUTPUTS_ROOT", tmp_path / "outputs")
    paths = RunPaths("fresh", "d_happy")
    paths.ensure()
    ntp_record = PromptRecord("profile", "profile", "ntp", "Question: happy?\nAnswer:\n")
    fa_record = PromptRecord("respondent", "profile", "fa", "Question: happy?\nAnswer:\n")
    backend = FakeBackend()
    trace = _trace()

    generate_records([ntp_record, fa_record], backend, paths, trace=trace)

    ntp_payload = _read(next((paths.raw / "ntp").glob("*.json")))
    assert ntp_payload["schema_version"] == RECORD_SCHEMA_VERSION
    assert ntp_payload["prompt"]["text"] == ntp_record.text
    assert ntp_payload["prompt"]["sha256"] == sha256_text(ntp_record.text)
    assert ntp_payload["trace"]["run_id"] == trace.run_id
    assert ntp_payload["trace"]["model"]["sha256"] == "0" * 64
    assert ntp_payload["trace"]["backend"]["name"] == "fake"
    assert ntp_payload["trace"]["code_revision"] == "deadbee"
    assert ntp_payload["trace"]["sampling"] == {
        "max_tokens": 1,
        "logprobs": 1000,
        "temperature": 0.0,
    }
    assert ntp_payload["trace"]["duration_ms"] >= 0
    assert ntp_payload["trace"]["created_at"]

    fa_payload = _read(next((paths.raw / "fa").glob("*.json")))
    assert fa_payload["attempt_count"] == 2
    assert [attempt["raw_text"] for attempt in fa_payload["attempts"]] == [
        "not valid",
        "A. Very happy\nignored",
    ]
    assert [attempt["accepted"] for attempt in fa_payload["attempts"]] == [False, True]
    assert fa_payload["attempts"][0]["seed"] == fa_payload["seed"]
    assert fa_payload["attempts"][1]["seed"] is None
    assert fa_payload["trace"]["sampling"]["temperature"] == 0.7

    events = [
        json.loads(line)
        for line in (paths.logs / EVENT_LOG_NAME).read_text(encoding="utf-8").splitlines()
    ]
    assert [event["status"] for event in events] == ["generated", "generated"]
    assert {event["mode"] for event in events} == {"ntp", "fa"}
    assert all(event["run_id"] == trace.run_id for event in events)

    frame, summary = audit_traces({"ntp": [ntp_record], "fa": [fa_record]}, paths)
    assert summary["ok"] is True
    assert summary["indexed"] == summary["traced"] == 2
    assert summary["untraced"] == 0
    assert summary["run_ids"] == [trace.run_id]
    assert summary["missing"] == {"ntp": 0, "fa": 0}
    assert set(frame["prompt_sha256"]) == {sha256_text(ntp_record.text)}


def test_audit_flags_mismatched_prompts_and_legacy_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "OUTPUTS_ROOT", tmp_path / "outputs")
    paths = RunPaths("fresh", "d_happy")
    paths.ensure()
    record = PromptRecord("profile", "profile", "ntp", "Question: happy?\nAnswer:\n")
    backend = FakeBackend()
    generate_records([record], backend, paths, trace=_trace())

    tampered = PromptRecord(record.prompt_id, record.profile, record.mode, "Different prompt\n")
    _, summary = audit_traces({"ntp": [tampered]}, paths)
    assert summary["ok"] is False
    assert summary["problems"] == ["prompt hash mismatch for ntp profile"]

    legacy = next((paths.raw / "ntp").glob("*.json"))
    legacy.write_text(
        json.dumps({"schema_version": 1, "mode": "ntp", "prompt_id": "profile", "profile": "p"}),
        encoding="utf-8",
    )
    frame, summary = audit_traces({"ntp": [record]}, paths)
    assert summary["untraced"] == 1
    assert summary["traced"] == 0
    assert summary["ok"] is True
    assert frame.iloc[0]["run_id"] is None

    counts = generate_records([record], backend, paths, trace=_trace())
    assert counts == {"generated": 0, "reused": 1, "reused_untraced": 1, "failed": 0}


def test_failed_records_are_logged_and_audited(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "OUTPUTS_ROOT", tmp_path / "outputs")
    paths = RunPaths("fresh", "d_happy")
    paths.ensure()

    class BrokenBackend(FakeBackend):
        def ntp(self, prompt: str) -> dict[str, float]:
            raise RuntimeError("no answer tokens")

    record = PromptRecord("profile", "profile", "ntp", "Answer:\n")
    counts = generate_records([record], BrokenBackend(), paths, trace=_trace())
    assert counts == {"generated": 0, "reused": 0, "reused_untraced": 0, "failed": 1}

    failure = next((paths.raw / "ntp").glob("*.failed.json"))
    payload = _read(failure)
    assert payload["error"] == "RuntimeError: no answer tokens"
    assert payload["trace"]["run_id"] == _trace().run_id
    assert payload["prompt"]["text"] == record.text

    events = [
        json.loads(line)
        for line in (paths.logs / EVENT_LOG_NAME).read_text(encoding="utf-8").splitlines()
    ]
    assert events[0]["status"] == "failed"
    assert events[0]["error"] == "RuntimeError: no answer tokens"

    _, summary = audit_traces({"ntp": [record]}, paths)
    assert summary["ok"] is False
    assert summary["failed_records"] == [str(failure.relative_to(paths.outputs))]
    assert summary["missing"] == {"ntp": 1}
