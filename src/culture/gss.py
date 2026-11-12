from __future__ import annotations

import ast
import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.request import urlopen

import numpy as np
import numpy.typing as npt
import pandas as pd

from machine_bias_reproduction.config import PROJECT_ROOT, UPSTREAM_DIR
from machine_bias_reproduction.io_utils import (
    atomic_write_json,
    atomic_write_text,
    code_revision,
    hardware_summary,
    sha256_file,
)
from machine_bias_reproduction.questions import LETTERS

from .adapters import staged_adapter
from .registry import (
    BASE_ARM,
    CULTURE_ROOT,
    PREFLIGHT_MIN_VALID_ANSWER_MASS,
    SUBPOP_ARM,
    WEIGHTS_FILE,
    CultureModel,
    is_base,
)

FloatArray = npt.NDArray[np.float64]

GSS = "gss"

GSS_ARMS: tuple[str, ...] = (BASE_ARM, SUBPOP_ARM)

GSS_MODE = "ntp"

SUBPOP_DATASET = "jjssuh/subpop"

SUBPOP_REVISION = "886d78a5cb55a576bd82b6724c46ae0f34a2ac3e"

SUBPOP_CODE_COMMIT = "e0569aca48361415779aac7407aed1e2cbb67b4a"

STEERING_URL = (
    "https://raw.githubusercontent.com/JosephJeesungSuh/subpop/"
    f"{SUBPOP_CODE_COMMIT}/data/subpopulation_metadata/steering_prompts.json"
)

EVAL_FILE = "subpop_eval.jsonl"

STEERING_FILE = "steering_prompts.json"

GSS_SHA256: dict[str, str] = {
    EVAL_FILE: "0f1a184f55d2424c317adad92aaa4058560f411ac835fa831c64f563c9200d60",
    STEERING_FILE: "66fa7288c46bebff59a86d7ddcfe2b08c43ba0c6d8ed1fda07d3b1158f9cde1c",
}

EXPECTED_RECORDS = 2_926

EXPECTED_QUESTIONS = 133

EXPECTED_SUBGROUPS = 22

DEFAULT_GSS_SOURCE = PROJECT_ROOT.parent / "culture-mllm" / "data" / "raw" / "subpop"

GSS_ROOT = UPSTREAM_DIR / "subpop"

ANSWER_STUB = "Answer:"

STEERING_BRIDGE = "Answer the following question keeping in mind your previous answers.\n"

PROMPT_CONTRACT = "subpop-qa"

OPTION_TOKENS: tuple[str, ...] = tuple(f" {letter}" for letter in LETTERS)

MIN_ORDERED_OPTIONS = 2

UNIT_TOLERANCE = 1e-6

RUN_KEYS: tuple[str, ...] = ("qkey", "attribute", "group")

RUN_TABLE = "NTP-{model}-{arm}-gss.csv"

RUN_MANIFEST = "inference_manifest.json"

Fetcher = Callable[[Path], None]


@dataclass(frozen=True, slots=True)
class GssRecord:
    qkey: str
    attribute: str
    group: str
    question: str
    options: tuple[str, ...]
    responses: tuple[float, ...]
    ordinal: tuple[float, ...]
    refusal_rate: float


@dataclass(frozen=True, slots=True)
class SteeringPrompt:
    attribute: str
    question: str
    options: tuple[str, ...]


class LetterBackend(Protocol):
    @property
    def batch_size(self) -> int: ...

    def set_culture(self, culture: str) -> None: ...

    def describe(self) -> dict[str, Any]: ...

    def continuation_token_ids(self, stub: str, continuations: Sequence[str]) -> list[int]: ...

    def first_token_batch(
        self, prompts: Sequence[str], token_ids: Sequence[int]
    ) -> list[list[float]]: ...


def parse_record(raw: Mapping[str, Any]) -> GssRecord:
    qkey = str(raw["qkey"])
    options = tuple(str(option) for option in raw["options"])
    responses = tuple(float(value) for value in raw["responses"])
    ordinal = tuple(float(value) for value in raw["ordinal"])
    refusal_rate = float(raw["refusal_rate"])
    if len(options) > len(LETTERS):
        raise ValueError(f"{qkey}: {len(options)} options, more than the {len(LETTERS)} letters")
    if len(options) != len(responses) + 1:
        raise ValueError(
            f"{qkey}: {len(options)} options for {len(responses)} responses; the last option "
            "must be the refusal the responses leave out"
        )
    if len(ordinal) != len(responses):
        raise ValueError(f"{qkey}: {len(ordinal)} ordinal levels for {len(responses)} responses")
    if any(value < 0.0 for value in responses):
        raise ValueError(f"{qkey}: negative response share")
    if abs(sum(responses) - 1.0) > UNIT_TOLERANCE:
        raise ValueError(f"{qkey}: responses sum to {sum(responses):.6f}, not 1")
    if not 0.0 <= refusal_rate <= 1.0:
        raise ValueError(f"{qkey}: refusal rate {refusal_rate} is outside [0, 1]")
    return GssRecord(
        qkey=qkey,
        attribute=str(raw["attribute"]),
        group=str(raw["group"]),
        question=str(raw["question"]),
        options=options,
        responses=responses,
        ordinal=ordinal,
        refusal_rate=refusal_rate,
    )


def _staged(root: Path, name: str) -> Path:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing (run: ./run_experiment_add_culture.sh gss-stage)"
        )
    return path


def load_records(root: Path = GSS_ROOT) -> list[GssRecord]:
    with _staged(root, EVAL_FILE).open(encoding="utf-8") as stream:
        return [parse_record(json.loads(line)) for line in stream if line.strip()]


def load_steering(root: Path = GSS_ROOT) -> dict[str, SteeringPrompt]:
    with _staged(root, STEERING_FILE).open(encoding="utf-8") as stream:
        entries = json.load(stream)
    steering: dict[str, SteeringPrompt] = {}
    for entry in entries:
        options = entry["options"]
        if isinstance(options, str):
            options = ast.literal_eval(options)
        attribute = str(entry["attribute"])
        steering[attribute] = SteeringPrompt(
            attribute=attribute,
            question=str(entry["qa_prompt"]),
            options=tuple(str(option) for option in options),
        )
    return steering


def ordered_positions(ordinal: Sequence[float]) -> tuple[int, ...]:
    kept = sorted((level, position) for position, level in enumerate(ordinal) if level >= 0.0)
    levels = [level for level, _ in kept]
    if len(kept) < MIN_ORDERED_OPTIONS:
        raise ValueError(f"fewer than {MIN_ORDERED_OPTIONS} ordered options in {list(ordinal)}")
    if len(set(levels)) != len(levels):
        raise ValueError(f"two options share an ordinal level in {list(ordinal)}")
    return tuple(position for _, position in kept)


def ordered_shares(values: Sequence[float], ordinal: Sequence[float]) -> FloatArray:
    kept = np.asarray([values[position] for position in ordered_positions(ordinal)], dtype=float)
    total = float(kept.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(kept.shape, np.nan)
    return np.asarray(kept / total, dtype=np.float64)


def option_letters(count: int) -> list[str]:
    return [chr(ord("A") + index) for index in range(count)]


def format_mcq(body: str, options: Sequence[str], *, answer_forcing: bool = False) -> str:
    letters = option_letters(len(options))
    lines = "\n".join(
        f"{letter}. {option.strip()}" for letter, option in zip(letters, options, strict=True)
    ).strip()
    forcing = (
        "Answer as a choice between " + ",".join(f"{letter}." for letter in letters)
        if answer_forcing
        else ""
    )
    question = f"Question: {body.strip()}\n{lines}\n{forcing}".strip()
    return f"{question}\n\n{ANSWER_STUB}"


def build_prompt(record: GssRecord, steering: Mapping[str, SteeringPrompt]) -> str:
    prompt = steering.get(record.attribute)
    if prompt is None:
        raise ValueError(f"{record.qkey}: no steering prompt for {record.attribute!r}")
    if record.group not in prompt.options:
        raise ValueError(
            f"{record.qkey}: {record.group!r} is not a steering option of {record.attribute!r}"
        )
    letter = option_letters(len(prompt.options))[prompt.options.index(record.group)]
    return (
        format_mcq(prompt.question, prompt.options)
        + f" {letter}. {record.group}\n\n"
        + STEERING_BRIDGE
        + format_mcq(record.question, record.options, answer_forcing=True)
    )


def check_records(records: Sequence[GssRecord], steering: Mapping[str, SteeringPrompt]) -> None:
    questions: dict[str, GssRecord] = {}
    subgroups: set[tuple[str, str]] = set()
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        key = (record.qkey, record.attribute, record.group)
        if key in seen:
            raise ValueError(f"{record.qkey}: {record.attribute}/{record.group} appears twice")
        seen.add(key)
        subgroups.add((record.attribute, record.group))
        first = questions.setdefault(record.qkey, record)
        if (record.question, record.options, record.ordinal) != (
            first.question,
            first.options,
            first.ordinal,
        ):
            raise ValueError(f"{record.qkey}: the question differs between subgroups")
        ordered_positions(record.ordinal)
        build_prompt(record, steering)
    if len(seen) != len(questions) * len(subgroups):
        raise ValueError("a question is missing one of the subgroups")


def _fetch_eval(target: Path) -> None:
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(
        repo_id=SUBPOP_DATASET,
        filename=EVAL_FILE,
        repo_type="dataset",
        revision=SUBPOP_REVISION,
    )
    shutil.copyfile(cached, target)


def _fetch_steering(target: Path) -> None:
    with urlopen(STEERING_URL, timeout=60) as response:
        target.write_bytes(response.read())


FETCHERS: dict[str, Fetcher] = {EVAL_FILE: _fetch_eval, STEERING_FILE: _fetch_steering}


def stage_gss(
    source: Path = DEFAULT_GSS_SOURCE,
    root: Path = GSS_ROOT,
    *,
    download: bool = False,
    fetchers: Mapping[str, Fetcher] = FETCHERS,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    origins: dict[str, str] = {}
    for name, expected in GSS_SHA256.items():
        target = root / name
        origin = source / name
        if not download and origin.is_file():
            if origin.resolve() != target.resolve():
                shutil.copyfile(origin, target)
            origins[name] = str(origin)
        elif download or not target.is_file():
            fetchers[name](target)
            origins[name] = STEERING_URL if name == STEERING_FILE else SUBPOP_DATASET
        else:
            origins[name] = str(target)
        found = sha256_file(target)
        if found != expected:
            raise RuntimeError(
                f"{target} has SHA-256 {found}, expected {expected}: it is not {SUBPOP_DATASET} "
                f"at revision {SUBPOP_REVISION[:12]} or the steering prompts at commit "
                f"{SUBPOP_CODE_COMMIT[:12]} (pass --download to fetch the pinned files)"
            )
    records = load_records(root)
    check_records(records, load_steering(root))
    counts = {
        "records": len(records),
        "questions": len({record.qkey for record in records}),
        "subgroups": len({(record.attribute, record.group) for record in records}),
    }
    expected_counts = {
        "records": EXPECTED_RECORDS,
        "questions": EXPECTED_QUESTIONS,
        "subgroups": EXPECTED_SUBGROUPS,
    }
    if counts != expected_counts:
        raise RuntimeError(f"staged {counts}, expected {expected_counts}")
    return {
        "root": str(root),
        "origins": origins,
        "dataset": SUBPOP_DATASET,
        "revision": SUBPOP_REVISION,
        "code_commit": SUBPOP_CODE_COMMIT,
        "sha256": dict(GSS_SHA256),
        **counts,
    }


def run_directory(model_key: str, arm: str, root: Path | None = None) -> Path:
    return (root or CULTURE_ROOT) / model_key / arm / GSS


def run_table(model_key: str, arm: str, root: Path | None = None) -> Path:
    return run_directory(model_key, arm, root) / RUN_TABLE.format(model=model_key, arm=arm)


def letter_probabilities(backend: LetterBackend, prompts: Sequence[str]) -> FloatArray:
    token_ids = backend.continuation_token_ids(ANSWER_STUB, OPTION_TOKENS)
    scored = np.full((len(prompts), len(OPTION_TOKENS)), np.nan)
    order = sorted(range(len(prompts)), key=lambda index: len(prompts[index]))
    for start in range(0, len(order), backend.batch_size):
        chunk = order[start : start + backend.batch_size]
        rows = backend.first_token_batch([prompts[index] for index in chunk], token_ids)
        for index, row in zip(chunk, rows, strict=True):
            scored[index] = row
    return scored


def run_frame(records: Sequence[GssRecord], probabilities: FloatArray) -> pd.DataFrame:
    offered = np.asarray([len(record.options) for record in records])
    shown = np.arange(len(LETTERS))[None, :] < offered[:, None]
    kept = np.where(shown, probabilities, np.nan)
    frame = pd.DataFrame(
        {
            "qkey": [record.qkey for record in records],
            "attribute": [record.attribute for record in records],
            "group": [record.group for record in records],
            "options": offered,
            "mass": np.nansum(kept, axis=1),
        }
    )
    return pd.concat([frame, pd.DataFrame(kept, columns=list(LETTERS))], axis=1)


def mass_summary(frame: pd.DataFrame) -> dict[str, Any]:
    mass = frame["mass"].to_numpy(dtype=np.float64)
    mean = float(mass.mean()) if mass.size else float("nan")
    return {
        "prompts": int(mass.size),
        "mean_mass": mean,
        "min_mass": float(mass.min()) if mass.size else float("nan"),
        "max_mass": float(mass.max()) if mass.size else float("nan"),
        "threshold": PREFLIGHT_MIN_VALID_ANSWER_MASS,
        "informative": bool(mean >= PREFLIGHT_MIN_VALID_ANSWER_MASS),
    }


def _adapter_stamp(model: CultureModel, arm: str) -> dict[str, Any]:
    if is_base(arm):
        return {"adapter_path": None, "adapter_sha256": None}
    adapter = staged_adapter(model.key, arm)
    return {
        "adapter_path": str(adapter.resolve()),
        "adapter_sha256": sha256_file(adapter / WEIGHTS_FILE),
    }


def run_gss(
    model: CultureModel,
    arm: str,
    backend: LetterBackend,
    records: Sequence[GssRecord],
    steering: Mapping[str, SteeringPrompt],
    *,
    limit: int | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    started = datetime.now(UTC).isoformat()
    backend.set_culture(arm)
    selected = list(records if limit is None else records[:limit])
    prompts = [build_prompt(record, steering) for record in selected]
    frame = run_frame(selected, letter_probabilities(backend, prompts))
    summary = mass_summary(frame)
    print(
        f"[{model.key}/{arm}] gss: mean valid-answer mass {summary['mean_mass']:.4f} over "
        f"{summary['prompts']} prompts ({'ok' if summary['informative'] else 'LOW'})",
        flush=True,
    )
    result = {"model": model.key, "arm": arm, "question": GSS, "mode": GSS_MODE, **summary}
    if limit is not None:
        return {**result, "table": None}
    destination = run_table(model.key, arm, root)
    atomic_write_text(destination, frame.to_csv(index=False))
    atomic_write_json(
        destination.parent / RUN_MANIFEST,
        {
            "schema_version": 1,
            "started_at": started,
            "finished_at": datetime.now(UTC).isoformat(),
            "code_revision": code_revision(PROJECT_ROOT),
            "question": GSS,
            "mode": GSS_MODE,
            "model": {
                "model_key": model.key,
                "culture": arm,
                "base_model_id": model.base_model_id,
                **_adapter_stamp(model, arm),
            },
            "backend": {**backend.describe(), "question": GSS, "prompt_contract": PROMPT_CONTRACT},
            "data": {
                "dataset": SUBPOP_DATASET,
                "revision": SUBPOP_REVISION,
                "code_commit": SUBPOP_CODE_COMMIT,
                "sha256": dict(GSS_SHA256),
            },
            "answer_tokens": list(OPTION_TOKENS),
            "preflight": summary,
            "hardware": hardware_summary(),
        },
    )
    return {**result, "table": str(destination)}
