from __future__ import annotations

import re
import statistics
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from machine_bias_reproduction.config import GLOBAL_SEED, OUTPUTS_ROOT
from machine_bias_reproduction.inference import answer_parser
from machine_bias_reproduction.io_utils import (
    atomic_write_json,
    stable_seed,
    write_csv,
)
from machine_bias_reproduction.prompts import PromptRecord, prompt_records
from machine_bias_reproduction.questions import Question, resolve_question

from .api_backend import REASONING_EFFORT, OpenAIBackend
from .registry import BASE_ARM, CultureModel

SMOKE_ROOT = OUTPUTS_ROOT / "culture" / "served_smoke"

SMOKE_PROMPTS = 5

SMOKE_ATTEMPTS = 3

SMOKE_FA_MAX_TOKENS = 300

VERDICTS: tuple[str, ...] = ("answered", "wrapped", "deflected", "refused", "unreadable")

_LABEL = re.compile(r"^\s*(?:the\s+)?answers?\s*(?:is|would\s+be)?\s*[:\-\u2013\u2014]?\s*", re.I)

_DEFLECTION = re.compile(
    r"please\s+(choose|select|pick|provide|answer|specify)"
    r"|which\s+(one|option|of\s+these)"
    r"|let\s+me\s+know"
    r"|your\s+answer\s*:",
    re.I,
)

_REFUSAL = re.compile(
    r"cannot\s+be\s+determined"
    r"|can(?:not|['\u2019]t)\s+(?:determine|tell|know|answer|say)"
    r"|unable\s+to\s+determine"
    r"|(?:not|in)sufficient\s+information"
    r"|not\s+enough\s+information"
    r"|do(?:n['\u2019]t|\s+not)\s+have\s+enough"
    r"|not\s+(?:specified|provided|stated|given)"
    r"|no\s+information",
    re.I,
)


def _first_line(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return stripped.splitlines()[0].strip()


def lenient_answer(candidate: str, question: str | Question) -> str | None:
    outcome = resolve_question(question)
    line = _LABEL.sub("", _first_line(candidate))
    line = line.strip().strip("*_`\"'\u201c\u201d ").strip()
    line = re.sub(r"[.\s]+$", "", line).strip()
    if not line:
        return None
    folded = line.casefold()
    for index, expected in enumerate(outcome.fa_answers, start=1):
        bare = re.sub(r"[.\s]+$", "", expected).casefold()
        if folded == bare:
            return expected
        if outcome.numerical:
            if folded == str(index):
                return expected
            continue
        letter, _, label = expected.partition(". ")
        if folded in {letter.casefold(), label.casefold(), f"{letter}.{label}".casefold()}:
            return expected
    return None


def classify(candidate: str, question: str | Question) -> str:
    outcome = resolve_question(question)
    if answer_parser(outcome)(candidate) is not None:
        return "answered"
    if lenient_answer(candidate, outcome) is not None:
        return "wrapped"
    if _DEFLECTION.search(candidate):
        return "deflected"
    if _REFUSAL.search(candidate):
        return "refused"
    return "unreadable"


def _best(verdicts: Sequence[str]) -> str:
    for verdict in VERDICTS:
        if verdict in verdicts:
            return verdict
    return "unreadable"


def _rate(count: float, total: int) -> float:
    return count / total if total else 0.0


def probe_records(
    wvs: pd.DataFrame,
    question: Question,
    prompts: int,
) -> list[PromptRecord]:
    return prompt_records(wvs, "fa", question)[:prompts]


def probe_model(
    model: CultureModel,
    question: Question,
    records: Sequence[PromptRecord],
    *,
    attempts: int = SMOKE_ATTEMPTS,
    fa_max_tokens: int = SMOKE_FA_MAX_TOKENS,
    batch_size: int | None = None,
    reasoning_effort: str | None = REASONING_EFFORT,
    backend: Any | None = None,
) -> dict[str, Any]:
    served = backend or OpenAIBackend(
        model,
        question,
        batch_size=batch_size,
        fa_max_new_tokens=fa_max_tokens,
        reasoning_effort=reasoning_effort,
    )
    served.set_culture(BASE_ARM)

    started = time.monotonic()
    masses = served.ntp_mass_probe([record.text for record in records])
    ntp_seconds = time.monotonic() - started

    trail: list[dict[str, Any]] = []
    verdicts: list[str] = []
    accepted_on: list[int | None] = []
    started = time.monotonic()
    outstanding = list(range(len(records)))
    seen: dict[int, list[str]] = {row: [] for row in outstanding}
    landed: dict[int, int] = {}
    for attempt in range(attempts):
        if not outstanding:
            break
        seeds = [
            stable_seed(f"{records[row].prompt_id}#{attempt}", GLOBAL_SEED) for row in outstanding
        ]
        texts = served.full_answer_batch(
            [records[row].text for row in outstanding],
            seeds=seeds,
        )
        thoughts = list(getattr(served, "last_reasoning_tokens", [])) or [0] * len(texts)
        still: list[int] = []
        for row, seed, text, thought in zip(outstanding, seeds, texts, thoughts, strict=True):
            verdict = classify(text, question)
            seen[row].append(verdict)
            trail.append(
                {
                    "prompt_id": records[row].prompt_id,
                    "profile": records[row].profile,
                    "attempt": attempt,
                    "seed": seed,
                    "verdict": verdict,
                    "reasoning_tokens": thought,
                    "raw_text": text,
                }
            )
            if verdict == "answered":
                landed[row] = attempt
            else:
                still.append(row)
        outstanding = still
    fa_seconds = time.monotonic() - started

    for row in range(len(records)):
        verdicts.append(_best(seen[row]))
        accepted_on.append(landed.get(row))

    counts = {verdict: verdicts.count(verdict) for verdict in VERDICTS}
    total = len(records)
    described = served.describe()
    return {
        "model": model.key,
        "label": model.label,
        "model_id": described["model_id"],
        "prompts": total,
        "attempts": attempts,
        **counts,
        "strict_rate": _rate(counts["answered"], total),
        "tolerant_rate": _rate(counts["answered"] + counts["wrapped"], total),
        "first_attempt": sum(1 for value in accepted_on if value == 0),
        "mean_ntp_mass": statistics.fmean(masses) if masses else 0.0,
        "logprobs_supported": described["logprobs_supported"],
        "reasoning_effort": described["reasoning_effort"],
        "reasoning_effort_supported": described["reasoning_effort_supported"],
        "mean_reasoning_tokens": described["mean_reasoning_tokens"],
        "fa_temperature": described["fa_temperature"],
        "fa_max_tokens": described["fa_max_new_tokens"],
        "ntp_seconds": ntp_seconds,
        "fa_seconds": fa_seconds,
        "seconds_per_prompt": _rate(fa_seconds, total),
        "trail": trail,
        "backend": described,
    }


def _table(rows: Sequence[dict[str, Any]]) -> str:
    header = (
        f"{'model':<8}{'prompts':>8}{'answered':>10}{'wrapped':>9}{'deflected':>11}"
        f"{'refused':>9}{'unread':>8}{'strict':>9}{'tolerant':>10}{'ntp mass':>10}"
        f"{'think':>8}{'s/prompt':>10}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        thought = row["mean_reasoning_tokens"]
        lines.append(
            f"{row['model']:<8}{row['prompts']:>8}{row['answered']:>10}{row['wrapped']:>9}"
            f"{row['deflected']:>11}{row['refused']:>9}{row['unreadable']:>8}"
            f"{row['strict_rate']:>8.1%}{row['tolerant_rate']:>10.1%}"
            f"{row['mean_ntp_mass']:>10.3f}"
            f"{'-' if thought is None else format(thought, '.0f'):>8}"
            f"{row['seconds_per_prompt']:>10.1f}"
        )
    return "\n".join(lines)


def run_smoke(
    models: Sequence[CultureModel],
    question: str | Question,
    wvs: pd.DataFrame,
    *,
    prompts: int = SMOKE_PROMPTS,
    attempts: int = SMOKE_ATTEMPTS,
    fa_max_tokens: int = SMOKE_FA_MAX_TOKENS,
    batch_size: int | None = None,
    reasoning_effort: str | None = REASONING_EFFORT,
    root: Any = SMOKE_ROOT,
) -> dict[str, Any]:
    outcome = resolve_question(question)
    records = probe_records(wvs, outcome, prompts)
    directory = root / outcome.var
    raw = directory / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    backends = [
        OpenAIBackend(
            model,
            outcome,
            batch_size=batch_size,
            fa_max_new_tokens=fa_max_tokens,
            reasoning_effort=reasoning_effort,
        )
        for model in models
    ]

    rows: list[dict[str, Any]] = []
    for model, served in zip(models, backends, strict=True):
        print(
            f"[{model.key}] {outcome.var}: {len(records)} prompts x {attempts} attempts "
            f"against {served.describe()['model_id']}",
            flush=True,
        )
        result = probe_model(
            model,
            outcome,
            records,
            attempts=attempts,
            fa_max_tokens=fa_max_tokens,
            batch_size=batch_size,
            reasoning_effort=reasoning_effort,
            backend=served,
        )
        trail = pd.DataFrame(result.pop("trail"))
        write_csv(trail, raw / f"{model.key}_attempts.csv")
        rows.append(result)
        print(
            f"[{model.key}] answered {result['answered']}/{result['prompts']} "
            f"({result['strict_rate']:.1%}), a further {result['wrapped']} carried an answer "
            f"the paper's parser rejects",
            flush=True,
        )

    frame = pd.DataFrame([{key: row[key] for key in row if key != "backend"} for row in rows])
    write_csv(frame, directory / "served_smoke.csv")
    atomic_write_json(
        directory / "served_smoke.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "question": outcome.var,
            "prompts": len(records),
            "attempts": attempts,
            "fa_max_tokens": fa_max_tokens,
            "reasoning_effort": reasoning_effort,
            "prompt_contract": "paper-as-single-user-message",
            "models": rows,
        },
    )
    print()
    print(_table(rows))
    print()
    print(f"per-attempt text: {raw}")
    return {"question": outcome.var, "directory": str(directory), "models": rows}
