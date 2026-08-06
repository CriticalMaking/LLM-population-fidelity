"""Resumable Mixtral NTP and full-answer generation."""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from .config import GLOBAL_SEED, RunPaths
from .io_utils import append_jsonl, atomic_write_json, sha256_text, stable_seed
from .prompts import PromptRecord
from .questions import Question, resolve_question

MAX_FA_RETRIES = 50

RECORD_SCHEMA_VERSION = 2
EVENT_LOG_NAME = "inference_events.jsonl"
TRACE_INDEX_COLUMNS = (
    "mode",
    "prompt_id",
    "profile",
    "run_id",
    "created_at",
    "duration_ms",
    "prompt_sha256",
    "model_sha256",
    "backend_version",
    "code_revision",
    "seed",
    "attempt_count",
    "answer",
    "mass",
    "record_path",
    "schema_version",
)


def strip_trailing_newline(prompt: str) -> str:
    """Drop the prompt's closing newline, as the upstream generators do."""
    return prompt[:-1] if prompt.endswith("\n") else prompt


class InferenceBackend(Protocol):
    """Minimal backend surface needed by the experiment driver."""

    def ntp(self, prompt: str) -> dict[str, float]:
        """Return valid-answer mass and normalized A-D probabilities."""

    def full_answer(self, prompt: str, *, seed: int | None) -> str:
        """Generate one candidate full answer."""

    def describe(self) -> dict[str, Any]:
        """Return the build and loading parameters stamped onto every record."""


class BatchInferenceBackend(InferenceBackend, Protocol):
    """A backend that can answer many prompts in one forward pass."""

    def ntp_batch(self, prompts: Sequence[str]) -> list[dict[str, float]]:
        """Return one NTP result per prompt."""

    def full_answer_batch(
        self,
        prompts: Sequence[str],
        *,
        seeds: Sequence[int | None],
    ) -> list[str]:
        """Generate one candidate per prompt, each from its own seed."""

    def token_length(self, prompt: str) -> int:
        """Return the tokenized length of one prompt, for length bucketing."""


@dataclass(frozen=True, slots=True)
class RunContext:
    """Identity of one fresh-inference invocation, stamped onto every record.

    Resumed runs reuse per-prompt files written by earlier invocations, so the
    run-level manifest alone cannot say which model build produced a given
    answer. Every record therefore carries its own copy of this context.
    """

    run_id: str
    started_at: str
    model: Mapping[str, Any]
    backend: Mapping[str, Any]
    code_revision: str | None
    hardware: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def new_run_id() -> str:
        """Return a sortable, unique identifier for one invocation."""
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{stamp}-{uuid.uuid4().hex[:8]}"

    def stamp(self) -> dict[str, Any]:
        """Return the per-record trace header."""
        return {
            "run_id": self.run_id,
            "run_started_at": self.started_at,
            "code_revision": self.code_revision,
            "model": dict(self.model),
            "backend": dict(self.backend),
        }


def _llama_cpp_version() -> str | None:
    """Return the installed llama-cpp-python version, if it is discoverable."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("llama_cpp_python")
    except PackageNotFoundError:
        return None


def _llama_cpp_build_info() -> str | None:
    """Return the compiled llama.cpp backend feature string.

    This is the only record of whether the wheel in use was actually built with
    a GPU backend, which a CPU-only build otherwise hides behind identical
    numbers.
    """
    try:
        from llama_cpp.llama_cpp import llama_print_system_info

        raw = llama_print_system_info()
    except Exception:
        return None
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
    return " ".join(text.split()) or None


class LlamaCppBackend:
    """Paper-compatible llama-cpp-python Mixtral backend."""

    def __init__(
        self,
        model_path: Path,
        question: str | Question = "d_happy",
        *,
        threads: int,
        gpu_layers: int,
        seed: int = GLOBAL_SEED,
    ) -> None:
        """Load the Q4_K_M model once for NTP and FA generation."""
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_cpp import llama_supports_gpu_offload
        except ImportError as error:
            raise RuntimeError("fresh inference requires: uv sync --extra inference") from error
        if gpu_layers != 0 and not llama_supports_gpu_offload():
            raise RuntimeError(
                "GPU offload requested but this llama-cpp-python build has no GPU "
                "backend compiled in (it was built CPU-only). Rebuild with, e.g., "
                'CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 uv sync --locked --group dev '
                "--extra inference --reinstall-package llama-cpp-python --no-cache "
                "(uv's build cache is keyed on package/version, not CMAKE_ARGS, so "
                "--reinstall-package --no-cache is required even on a repeat run). "
                "Pass --gpu-layers 0 to run on CPU instead."
            )
        self._model: Any = Llama(
            model_path=str(model_path),
            n_gpu_layers=gpu_layers,
            n_ctx=512,
            n_threads=threads,
            verbose=False,
            logits_all=True,
            seed=seed,
        )
        self._question = resolve_question(question)
        self._description: dict[str, Any] = {
            "name": "llama-cpp-python",
            "version": _llama_cpp_version(),
            "build_info": _llama_cpp_build_info(),
            "gpu_offload_supported": bool(llama_supports_gpu_offload()),
            "n_gpu_layers": gpu_layers,
            "n_threads": threads,
            "n_ctx": int(self._model.n_ctx()),
            "load_seed": seed,
            "question": self._question.var,
            "answer_tokens": list(self._question.ntp_tokens),
        }

    def describe(self) -> dict[str, Any]:
        """Return the resolved build and loading parameters of this backend."""
        return dict(self._description)

    @staticmethod
    def _without_final_newline(prompt: str) -> str:
        return strip_trailing_newline(prompt)

    def ntp(self, prompt: str) -> dict[str, float]:
        """Run the paper's one-token log-probability extraction."""
        result: dict[str, Any] = self._model(
            self._without_final_newline(prompt),
            max_tokens=1,
            logprobs=1000,
        )
        logprobs = result["choices"][0]["logprobs"]["top_logprobs"][0]
        raw = [
            math.exp(float(logprobs.get(token, -math.inf))) for token in self._question.ntp_tokens
        ]
        mass = sum(raw)
        if mass <= 0:
            return {"mass": 0.0, "degenerate": True}
        normalized = [value / mass for value in raw]
        columns = self._question.answer_columns
        return {"mass": mass, **dict(zip(columns, normalized, strict=True))}

    def full_answer(self, prompt: str, *, seed: int | None) -> str:
        """Generate one temperature-0.7 answer candidate."""
        if seed is not None:
            self._model.set_seed(seed)
        prompt_tokens = self._model.tokenize(self._without_final_newline(prompt).encode("utf-8"))
        result: dict[str, Any] = self._model.create_completion(
            prompt_tokens,
            max_tokens=12,
            temperature=0.7,
        )
        return str(result["choices"][0]["text"])


def parse_full_answer(candidate: str, question: str | Question = "d_happy") -> str | None:
    """Normalize one candidate using the upstream FA acceptance rules.

    Mirrors ``2b-generate-FA-Mixtral-8x7B.py``: keep the first line, strip it,
    rewrite a leading digit as the matching letter for categorical questions,
    then require a case-insensitive match against an expected answer. A
    numerical question skips the digit rewrite, because its answers *are*
    digits.
    """
    outcome = resolve_question(question)
    first_line = re.sub(r"\n+.*$", "", candidate, flags=re.DOTALL).strip()
    if not outcome.numerical:
        for index, letter in enumerate(outcome.answer_columns, start=1):
            first_line = re.sub(rf"^{index}", letter, first_line)
    for expected in outcome.fa_answers:
        if first_line.casefold() == expected.casefold():
            return expected
    return None


def answer_parser(question: str | Question) -> Callable[[str], str | None]:
    """Return the FA acceptance rule bound to one question."""
    outcome = resolve_question(question)
    return lambda candidate: parse_full_answer(candidate, outcome)


def _result_path(paths: RunPaths, record: PromptRecord) -> Path:
    return paths.raw / record.mode / f"{sha256_text(record.prompt_id)[:24]}.json"


def _sampling(mode: str) -> dict[str, Any]:
    """Return the exact sampling parameters the backend applies for a mode."""
    if mode == "ntp":
        return {"max_tokens": 1, "logprobs": 1000, "temperature": 0.0}
    return {"max_tokens": 12, "temperature": 0.7, "max_retries": MAX_FA_RETRIES}


def _batched_sampling(
    mode: str,
    *,
    batch_size: int,
    fa_max_tokens: int,
    max_retries: int = MAX_FA_RETRIES,
) -> dict[str, Any]:
    """Return the sampling parameters of the batched transformers path.

    ``batch_size`` belongs in the trace because batched matmul reductions are
    not bitwise identical to batch-of-one, so a result is reproducible at the
    batch size that produced it.
    """
    if mode == "ntp":
        return {
            "max_tokens": 1,
            "logits": "full_vocabulary",
            "temperature": 0.0,
            "batch_size": batch_size,
        }
    return {
        "max_tokens": fa_max_tokens,
        "temperature": 0.7,
        "max_retries": max_retries,
        "batch_size": batch_size,
    }


def _new_payload(
    record: PromptRecord,
    trace: RunContext,
    sampling: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the common per-prompt record header."""
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "mode": record.mode,
        "prompt_id": record.prompt_id,
        "profile": record.profile,
        "prompt": {"text": record.text, "sha256": sha256_text(record.text)},
        "trace": {
            **trace.stamp(),
            "created_at": datetime.now(UTC).isoformat(),
            "sampling": dict(sampling),
        },
    }


def _attempt_seed(prompt_id: str, attempt: int, *, legacy_unseeded_fa: bool) -> int | None:
    """Derive a distinct, stable seed for one FA attempt.

    Unlike the llama.cpp path, whose retries are unseeded, every batched
    attempt gets its own derived seed. A prompt that needs three tries is then
    just as repeatable as one that succeeds immediately.
    ``--legacy-unseeded-fa`` still opts into the paper's unrepeatable sampling.
    """
    if legacy_unseeded_fa:
        return None
    return stable_seed(f"{prompt_id}#{attempt}", GLOBAL_SEED)


def _log_event(paths: RunPaths, payload: Mapping[str, Any]) -> None:
    """Append one inference event to the run's append-only log.

    The per-prompt files are keyed by prompt and replaced by ``--force``, so
    this log is what preserves the history of superseded generations.
    """
    append_jsonl(paths.logs / EVENT_LOG_NAME, payload)


def generate_records(
    records: Sequence[PromptRecord],
    backend: InferenceBackend,
    paths: RunPaths,
    *,
    trace: RunContext,
    legacy_unseeded_fa: bool = False,
    force: bool = False,
    parse: Callable[[str], str | None] = parse_full_answer,
) -> dict[str, int]:
    """Generate traced per-prompt atomic results, reusing valid existing files."""
    generated = 0
    reused = 0
    reused_untraced = 0
    failed = 0
    for record in records:
        destination = _result_path(paths, record)
        if destination.exists() and not force:
            reused += 1
            if _record_schema_version(destination) < RECORD_SCHEMA_VERSION:
                reused_untraced += 1
            continue
        prompt_sha256 = sha256_text(record.text)
        started = time.monotonic()
        payload: dict[str, Any] = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "mode": record.mode,
            "prompt_id": record.prompt_id,
            "profile": record.profile,
            "prompt": {"text": record.text, "sha256": prompt_sha256},
            "trace": {
                **trace.stamp(),
                "created_at": datetime.now(UTC).isoformat(),
                "sampling": _sampling(record.mode),
            },
        }
        try:
            if record.mode == "ntp":
                result = backend.ntp(record.text)
                if result.get("degenerate"):
                    raise RuntimeError(
                        "no probability mass on any expected answer token; "
                        "the model answered in another format"
                    )
                payload["result"] = result
            else:
                seed = None if legacy_unseeded_fa else stable_seed(record.prompt_id, GLOBAL_SEED)
                payload["seed"] = seed
                attempts: list[dict[str, Any]] = []
                answer: str | None = None
                for attempt in range(MAX_FA_RETRIES):
                    attempt_seed = seed if attempt == 0 else None
                    attempt_started = time.monotonic()
                    candidate = backend.full_answer(record.text, seed=attempt_seed)
                    answer = parse(candidate)
                    attempts.append(
                        {
                            "index": attempt,
                            "seed": attempt_seed,
                            "raw_text": candidate,
                            "parsed": answer,
                            "accepted": answer is not None,
                            "duration_ms": _elapsed_ms(attempt_started),
                        }
                    )
                    if answer is not None:
                        break
                payload["attempts"] = attempts
                payload["attempt_count"] = len(attempts)
                if answer is None:
                    raise RuntimeError(f"no valid FA answer after {MAX_FA_RETRIES} attempts")
                payload["result"] = {"answer": answer}
            payload["trace"]["duration_ms"] = _elapsed_ms(started)
            atomic_write_json(destination, payload)
            generated += 1
            _log_event(paths, _event(payload, destination, paths, status="generated"))
        except Exception as error:
            failed += 1
            payload["trace"]["duration_ms"] = _elapsed_ms(started)
            payload["error"] = f"{type(error).__name__}: {error}"
            failure_path = destination.with_suffix(".failed.json")
            atomic_write_json(failure_path, payload)
            _log_event(
                paths,
                _event(payload, failure_path, paths, status="failed"),
            )
    return {
        "generated": generated,
        "reused": reused,
        "reused_untraced": reused_untraced,
        "failed": failed,
    }


def _pending_records(
    records: Sequence[PromptRecord],
    paths: RunPaths,
    *,
    force: bool,
) -> tuple[list[PromptRecord], int, int]:
    """Split records into work to do and already-complete results to reuse."""
    pending: list[PromptRecord] = []
    reused = 0
    reused_untraced = 0
    for record in records:
        destination = _result_path(paths, record)
        if destination.exists() and not force:
            reused += 1
            if _record_schema_version(destination) < RECORD_SCHEMA_VERSION:
                reused_untraced += 1
            continue
        pending.append(record)
    return pending, reused, reused_untraced


def _write_result(
    payload: dict[str, Any],
    record: PromptRecord,
    paths: RunPaths,
    *,
    started: float,
) -> None:
    payload["trace"]["duration_ms"] = _elapsed_ms(started)
    destination = _result_path(paths, record)
    atomic_write_json(destination, payload)
    _log_event(paths, _event(payload, destination, paths, status="generated"))


def _write_failure(
    payload: dict[str, Any],
    record: PromptRecord,
    paths: RunPaths,
    *,
    started: float,
    error: BaseException,
) -> None:
    payload["trace"]["duration_ms"] = _elapsed_ms(started)
    payload["error"] = f"{type(error).__name__}: {error}"
    failure_path = _result_path(paths, record).with_suffix(".failed.json")
    atomic_write_json(failure_path, payload)
    _log_event(paths, _event(payload, failure_path, paths, status="failed"))


def _batches(
    records: Sequence[PromptRecord],
    backend: BatchInferenceBackend,
    batch_size: int,
) -> list[list[PromptRecord]]:
    """Order pending prompts by length and chunk them.

    Length bucketing keeps padding waste low. It is safe precisely because
    every result is either deterministic given the prompt (NTP) or drawn from a
    per-prompt seed (FA), so batch composition cannot change an answer.
    """
    ordered = sorted(records, key=lambda record: backend.token_length(record.text))
    return [ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)]


def _generate_ntp_batch(
    batch: Sequence[PromptRecord],
    backend: BatchInferenceBackend,
    paths: RunPaths,
    *,
    trace: RunContext,
    sampling: Mapping[str, Any],
) -> tuple[int, int]:
    started = time.monotonic()
    payloads = [_new_payload(record, trace, sampling) for record in batch]
    try:
        results = backend.ntp_batch([record.text for record in batch])
    except Exception as error:
        for record, payload in zip(batch, payloads, strict=True):
            _write_failure(payload, record, paths, started=started, error=error)
        return 0, len(batch)
    generated = 0
    failed = 0
    for record, payload, result in zip(batch, payloads, results, strict=True):
        if result.get("degenerate"):
            _write_failure(
                payload,
                record,
                paths,
                started=started,
                error=RuntimeError(
                    "no probability mass on any expected answer token; "
                    "the model answered in another format"
                ),
            )
            failed += 1
            continue
        payload["result"] = result
        _write_result(payload, record, paths, started=started)
        generated += 1
    return generated, failed


def _generate_fa_batch(
    batch: Sequence[PromptRecord],
    backend: BatchInferenceBackend,
    paths: RunPaths,
    *,
    trace: RunContext,
    sampling: Mapping[str, Any],
    legacy_unseeded_fa: bool,
    parse: Callable[[str], str | None],
    max_fa_retries: int,
) -> tuple[int, int]:
    """Generate a batch of full answers, retrying only the rows that failed.

    The retry loop runs at batch level so a handful of stubborn prompts never
    forces the whole batch through another pass, while each record still
    accumulates the complete per-attempt audit trail.
    """
    started = time.monotonic()
    payloads = [_new_payload(record, trace, sampling) for record in batch]
    attempts: list[list[dict[str, Any]]] = [[] for _ in batch]
    answers: list[str | None] = [None] * len(batch)
    outstanding = list(range(len(batch)))
    failure: BaseException | None = None

    for attempt in range(max_fa_retries):
        if not outstanding:
            break
        seeds = [
            _attempt_seed(batch[row].prompt_id, attempt, legacy_unseeded_fa=legacy_unseeded_fa)
            for row in outstanding
        ]
        attempt_started = time.monotonic()
        try:
            candidates = backend.full_answer_batch(
                [batch[row].text for row in outstanding],
                seeds=seeds,
            )
        except Exception as error:
            failure = error
            break
        duration = _elapsed_ms(attempt_started)
        still_outstanding: list[int] = []
        for row, seed, candidate in zip(outstanding, seeds, candidates, strict=True):
            parsed = parse(candidate)
            attempts[row].append(
                {
                    "index": attempt,
                    "seed": seed,
                    "raw_text": candidate,
                    "parsed": parsed,
                    "accepted": parsed is not None,
                    "duration_ms": duration,
                }
            )
            if parsed is None:
                still_outstanding.append(row)
            else:
                answers[row] = parsed
        outstanding = still_outstanding

    generated = 0
    failed = 0
    for row, (record, payload) in enumerate(zip(batch, payloads, strict=True)):
        payload["seed"] = _attempt_seed(record.prompt_id, 0, legacy_unseeded_fa=legacy_unseeded_fa)
        payload["attempts"] = attempts[row]
        payload["attempt_count"] = len(attempts[row])
        if answers[row] is None:
            reason = failure or RuntimeError(
                f"no valid FA answer after {len(attempts[row])} attempts"
            )
            _write_failure(payload, record, paths, started=started, error=reason)
            failed += 1
            continue
        payload["result"] = {"answer": answers[row]}
        _write_result(payload, record, paths, started=started)
        generated += 1
    return generated, failed


def generate_records_batched(
    records: Sequence[PromptRecord],
    backend: BatchInferenceBackend,
    paths: RunPaths,
    *,
    trace: RunContext,
    batch_size: int,
    fa_max_tokens: int = 12,
    legacy_unseeded_fa: bool = False,
    force: bool = False,
    progress_every: int = 25,
    parse: Callable[[str], str | None] = parse_full_answer,
    max_fa_retries: int = MAX_FA_RETRIES,
) -> dict[str, int]:
    """Generate traced per-prompt results in batches, reusing valid files.

    Writes exactly the records the one-at-a-time path writes: same schema, same
    trace header, same append-only event log. Only the execution shape differs.
    """
    modes = {record.mode for record in records}
    if len(modes) > 1:
        raise ValueError(f"batched generation handles one mode at a time, got {sorted(modes)}")
    pending, reused, reused_untraced = _pending_records(records, paths, force=force)
    generated = 0
    failed = 0
    if not pending:
        return {
            "generated": 0,
            "reused": reused,
            "reused_untraced": reused_untraced,
            "failed": 0,
        }

    mode = pending[0].mode
    sampling = _batched_sampling(
        mode, batch_size=batch_size, fa_max_tokens=fa_max_tokens, max_retries=max_fa_retries
    )
    batches = _batches(pending, backend, batch_size)
    started = time.monotonic()
    for index, batch in enumerate(batches, start=1):
        if mode == "ntp":
            batch_generated, batch_failed = _generate_ntp_batch(
                batch, backend, paths, trace=trace, sampling=sampling
            )
        else:
            batch_generated, batch_failed = _generate_fa_batch(
                batch,
                backend,
                paths,
                trace=trace,
                sampling=sampling,
                legacy_unseeded_fa=legacy_unseeded_fa,
                parse=parse,
                max_fa_retries=max_fa_retries,
            )
        generated += batch_generated
        failed += batch_failed
        if progress_every and (index % progress_every == 0 or index == len(batches)):
            done = generated + failed
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"  {mode}: {done}/{len(pending)} prompts, {done / elapsed:.1f}/s, {failed} failed",
                flush=True,
            )
    return {
        "generated": generated,
        "reused": reused,
        "reused_untraced": reused_untraced,
        "failed": failed,
    }


def _elapsed_ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000.0, 3)


def _event(
    payload: Mapping[str, Any],
    path: Path,
    paths: RunPaths,
    *,
    status: str,
) -> dict[str, Any]:
    """Summarize one generated or failed record for the append-only log."""
    trace = payload["trace"]
    result = payload.get("result", {})
    event: dict[str, Any] = {
        "run_id": trace["run_id"],
        "created_at": trace["created_at"],
        "status": status,
        "mode": payload["mode"],
        "prompt_id": payload["prompt_id"],
        "prompt_sha256": payload["prompt"]["sha256"],
        "model_sha256": trace["model"].get("sha256"),
        "duration_ms": trace.get("duration_ms"),
        "record": str(path.relative_to(paths.outputs)),
    }
    if payload["mode"] == "fa":
        event["seed"] = payload.get("seed")
        event["attempt_count"] = payload.get("attempt_count")
        event["answer"] = result.get("answer")
    else:
        event["mass"] = result.get("mass")
    if "error" in payload:
        event["error"] = payload["error"]
    return event


def _record_schema_version(path: Path) -> int:
    """Return a reused record's schema version, treating unreadable files as legacy."""
    try:
        with path.open(encoding="utf-8") as stream:
            payload: dict[str, Any] = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return 0
    version = payload.get("schema_version", 0)
    return version if isinstance(version, int) else 0


def _read_result(paths: RunPaths, record: PromptRecord) -> dict[str, Any]:
    path = _result_path(paths, record)
    if not path.is_file():
        raise FileNotFoundError(f"missing result for {record.mode} prompt {record.prompt_id}")
    with path.open(encoding="utf-8") as stream:
        payload: dict[str, Any] = json.load(stream)
    if payload["prompt_id"] != record.prompt_id or payload["mode"] != record.mode:
        raise ValueError(f"identity mismatch in {path}")
    return payload


def _optional_result(
    paths: RunPaths,
    record: PromptRecord,
    *,
    skip_missing: bool,
) -> dict[str, Any] | None:
    """Read one stored result, returning ``None`` for a prompt with no answer."""
    if skip_missing and not _result_path(paths, record).is_file():
        return None
    return _read_result(paths, record)


def consolidate_ntp(
    records: Sequence[PromptRecord],
    paths: RunPaths,
    destination: Path,
    question: str | Question = "d_happy",
    *,
    skip_missing: bool = False,
) -> int:
    """Consolidate per-profile NTP JSON into the paper-compatible CSV.

    ``skip_missing`` lets a run that could not answer every prompt still
    produce a table of what it did answer; the analysis reports the gap as
    coverage rather than inferring values that were never generated.
    """
    outcome = resolve_question(question)
    rows: list[dict[str, Any]] = []
    for record in records:
        payload = _optional_result(paths, record, skip_missing=skip_missing)
        if payload is None:
            continue
        rows.append({"profile": record.profile, **payload["result"]})
    frame = pd.DataFrame(rows, columns=["profile", "mass", *outcome.answer_columns])
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return len(rows)


def consolidate_fa(
    records: Sequence[PromptRecord],
    paths: RunPaths,
    destination: Path,
    question: str | Question = "d_happy",
    *,
    skip_missing: bool = False,
) -> int:
    """Consolidate per-observation FA JSON into the narrowed canonical CSV."""
    outcome = resolve_question(question)
    rows: list[dict[str, Any]] = []
    for record in records:
        payload = _optional_result(paths, record, skip_missing=skip_missing)
        if payload is None:
            continue
        rows.append(
            {
                "id": record.prompt_id,
                "profile": record.profile,
                outcome.var: payload["result"]["answer"],
            }
        )
    frame = pd.DataFrame(rows, columns=["id", "profile", outcome.var])
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return len(rows)


def _trace_row(payload: Mapping[str, Any], path: Path, paths: RunPaths) -> dict[str, Any]:
    trace = payload.get("trace", {})
    result = payload.get("result", {})
    return {
        "mode": payload["mode"],
        "prompt_id": payload["prompt_id"],
        "profile": payload["profile"],
        "run_id": trace.get("run_id"),
        "created_at": trace.get("created_at"),
        "duration_ms": trace.get("duration_ms"),
        "prompt_sha256": payload.get("prompt", {}).get("sha256"),
        "model_sha256": trace.get("model", {}).get("sha256"),
        "backend_version": trace.get("backend", {}).get("version"),
        "code_revision": trace.get("code_revision"),
        "seed": payload.get("seed"),
        "attempt_count": payload.get("attempt_count"),
        "answer": result.get("answer"),
        "mass": result.get("mass"),
        "record_path": str(path.relative_to(paths.outputs)),
        "schema_version": payload.get("schema_version", 0),
    }


def audit_traces(
    records_by_mode: Mapping[str, Sequence[PromptRecord]],
    paths: RunPaths,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Index every stored inference and verify its trace against the prompts.

    Re-hashing ``record.text`` proves that a stored answer belongs to the prompt
    the current code builds for that respondent, which is the claim a
    reproduction actually needs. Records written before the trace schema are
    reported as untraced rather than treated as failures.
    """
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    missing: dict[str, int] = {}
    untraced = 0
    failures: list[str] = []

    for mode, records in records_by_mode.items():
        absent = 0
        for record in records:
            path = _result_path(paths, record)
            failure_path = path.with_suffix(".failed.json")
            if failure_path.is_file():
                failures.append(str(failure_path.relative_to(paths.outputs)))
            if not path.is_file():
                absent += 1
                continue
            with path.open(encoding="utf-8") as stream:
                payload: dict[str, Any] = json.load(stream)
            if payload["prompt_id"] != record.prompt_id or payload["mode"] != record.mode:
                problems.append(f"identity mismatch: {path}")
                continue
            row = _trace_row(payload, path, paths)
            if row["schema_version"] < RECORD_SCHEMA_VERSION:
                untraced += 1
            elif row["prompt_sha256"] != sha256_text(record.text):
                problems.append(f"prompt hash mismatch for {mode} {record.prompt_id}")
            rows.append(row)
        missing[mode] = absent

    frame = pd.DataFrame(rows, columns=list(TRACE_INDEX_COLUMNS))
    traced = frame[frame["schema_version"] >= RECORD_SCHEMA_VERSION]
    summary: dict[str, Any] = {
        "indexed": len(frame),
        "traced": len(traced),
        "untraced": untraced,
        "missing": missing,
        "failed_records": failures,
        "run_ids": sorted(traced["run_id"].dropna().unique().tolist()),
        "model_sha256": sorted(traced["model_sha256"].dropna().unique().tolist()),
        "code_revisions": sorted(traced["code_revision"].dropna().unique().tolist()),
        "problems": problems,
        "ok": not problems and not failures,
    }
    return frame, summary
