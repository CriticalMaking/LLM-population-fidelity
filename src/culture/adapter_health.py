"""Whether a staged adapter is worth spending GPU time on.

Two independent signals, because they answer different questions. The trainer's
own end-of-training numbers say whether the finetuning learned anything; the
norm of the weight update it produced says how it failed when it did not. A
first loss above the uniform-random ceiling separates the two cases that look
alike from the outside: a base model that was already broken when training
started, and a finetuning run that diverged from a working one.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .registry import ADAPTER_MIN_EVAL_TOKEN_ACCURACY, WEIGHTS_FILE

TRAINER_STATE_FILE = "trainer_state.json"
TOKENIZER_FILE = "tokenizer.json"

VERDICT_HEALTHY = "healthy"
VERDICT_DIVERGED = "diverged"
VERDICT_BROKEN_BASE = "diverged (broken base)"
VERDICT_UNKNOWN = "unknown"

LORA_A = ".lora_A."
LORA_B = ".lora_B."

_BFLOAT16 = "BF16"
_NUMPY_DTYPES = {"F64": np.float64, "F32": np.float32, "F16": np.float16}


def trainer_state_path(directory: Path) -> Path | None:
    """The end-of-training trainer state, staged copy first.

    Training writes it per checkpoint rather than beside the selected adapter,
    so an unstaged source directory is read through its last checkpoint.
    """
    staged = directory / TRAINER_STATE_FILE
    if staged.is_file():
        return staged
    checkpoints = [
        path
        for path in directory.glob("checkpoint-*")
        if path.name.split("-")[-1].isdigit() and (path / TRAINER_STATE_FILE).is_file()
    ]
    if not checkpoints:
        return None
    last = max(checkpoints, key=lambda path: int(path.name.split("-")[-1]))
    return last / TRAINER_STATE_FILE


def _last(history: list[dict[str, Any]], key: str) -> float | None:
    for entry in reversed(history):
        value = entry.get(key)
        if value is not None:
            return float(value)
    return None


def read_trainer_state(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        state: dict[str, Any] = json.load(stream)
    history = [entry for entry in state.get("log_history") or [] if isinstance(entry, dict)]
    first_loss = next((float(e["loss"]) for e in history if e.get("loss") is not None), None)
    return {
        "first_loss": first_loss,
        "final_loss": _last(history, "loss"),
        "final_eval_loss": _last(history, "eval_loss"),
        "token_accuracy": _last(history, "mean_token_accuracy"),
        "eval_token_accuracy": _last(history, "eval_mean_token_accuracy"),
        "eval_f1_macro": _last(history, "eval_f1_macro"),
        "epoch": state.get("epoch"),
        "global_step": state.get("global_step"),
    }


def vocab_size(directory: Path) -> int | None:
    """Vocabulary size from the tokenizer staged beside the adapter."""
    path = directory / TOKENIZER_FILE
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        tokenizer: dict[str, Any] = json.load(stream)
    vocab = (tokenizer.get("model") or {}).get("vocab") or {}
    return len(vocab) + len(tokenizer.get("added_tokens") or []) or None


def read_tensors(path: Path) -> dict[str, np.ndarray]:
    """Tensors from a safetensors file, without importing torch.

    The health check is a CPU-only sanity pass, so it must stay runnable
    without the `culture` extra installed.
    """
    size = path.stat().st_size
    with path.open("rb") as stream:
        (header_length,) = struct.unpack("<Q", stream.read(8))
        if header_length <= 0 or header_length + 8 > size:
            raise ValueError(
                f"{path}: safetensors header of {header_length} bytes does not fit {size} bytes"
            )
        header = json.loads(stream.read(header_length))
        body = stream.read()
    tensors: dict[str, np.ndarray] = {}
    for name, record in header.items():
        if name == "__metadata__":
            continue
        start, end = record["data_offsets"]
        raw = body[start:end]
        dtype = record["dtype"]
        if dtype == _BFLOAT16:
            values = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dtype in _NUMPY_DTYPES:
            values = np.frombuffer(raw, dtype=_NUMPY_DTYPES[dtype])
        else:
            raise ValueError(f"{path}: unsupported tensor dtype {dtype}")
        tensors[name] = values.reshape(record["shape"]).astype(np.float64)
    return tensors


def _frobenius_of_product(factor_a: np.ndarray, factor_b: np.ndarray) -> float:
    """‖B @ A‖_F without forming B @ A.

    ‖BA‖² = tr(AAᵀ · BᵀB), and both Gram matrices are rank-by-rank, so this
    stays small where the product would be thousands by thousands. Both are
    symmetric, so the trace of their product is their elementwise sum.
    """
    gram_a = factor_a @ factor_a.T
    gram_b = factor_b.T @ factor_b
    return float(np.sqrt(max(float(np.sum(gram_a * gram_b)), 0.0)))


_NO_UPDATE: dict[str, Any] = {
    "modules": 0,
    "update_norm_mean": None,
    "update_norm_median": None,
    "update_norm_max": None,
    "lora_a_norm_mean": None,
    "lora_b_norm_mean": None,
}


def update_norms(weights: Path, *, lora_alpha: float, rank: int) -> dict[str, Any]:
    """Size of the weight update the adapter applies, per targeted module.

    This is the quantity that actually reaches the base model: peft adds
    (alpha / r) * B @ A, so the scaling belongs in the measurement.

    A file this cannot parse reports no modules rather than raising. Staging
    re-copies a truncated adapter by comparing hashes, and it can only get that
    far if measuring one does not abort the run; the resulting record is
    unknown, which is never treated as a pass.
    """
    try:
        tensors = read_tensors(weights)
    except (OSError, ValueError, struct.error, json.JSONDecodeError, UnicodeDecodeError):
        return dict(_NO_UPDATE)
    scaling = lora_alpha / rank if rank else 1.0
    updates: list[float] = []
    a_norms: list[float] = []
    b_norms: list[float] = []
    for name, factor_a in tensors.items():
        if LORA_A not in name:
            continue
        factor_b = tensors.get(name.replace(LORA_A, LORA_B))
        if factor_b is None:
            continue
        updates.append(scaling * _frobenius_of_product(factor_a, factor_b))
        a_norms.append(float(np.linalg.norm(factor_a)))
        b_norms.append(float(np.linalg.norm(factor_b)))
    if not updates:
        return dict(_NO_UPDATE)
    return {
        "modules": len(updates),
        "update_norm_mean": float(np.mean(updates)),
        "update_norm_median": float(np.median(updates)),
        "update_norm_max": float(np.max(updates)),
        "lora_a_norm_mean": float(np.mean(a_norms)),
        "lora_b_norm_mean": float(np.mean(b_norms)),
    }


def random_guess_loss(vocabulary: int | None) -> float | None:
    """Cross-entropy of guessing uniformly over the vocabulary."""
    if not vocabulary or vocabulary < 2:
        return None
    return math.log(vocabulary)


def verdict(
    *,
    eval_token_accuracy: float | None,
    first_loss: float | None,
    guess_loss: float | None,
) -> str:
    """Healthy, diverged, or diverged from a base that was already broken.

    A missing trainer state is reported as unknown rather than passed: an
    adapter nobody measured is not an adapter known to be sound.
    """
    if eval_token_accuracy is None:
        return VERDICT_UNKNOWN
    if eval_token_accuracy >= ADAPTER_MIN_EVAL_TOKEN_ACCURACY:
        return VERDICT_HEALTHY
    if first_loss is not None and guess_loss is not None and first_loss > guess_loss:
        return VERDICT_BROKEN_BASE
    return VERDICT_DIVERGED


def measure(directory: Path, config: dict[str, Any]) -> dict[str, Any]:
    """The full health record for one staged adapter directory."""
    record: dict[str, Any] = {
        "first_loss": None,
        "final_loss": None,
        "final_eval_loss": None,
        "token_accuracy": None,
        "eval_token_accuracy": None,
        "eval_f1_macro": None,
        "epoch": None,
        "global_step": None,
    }
    state = trainer_state_path(directory)
    if state is not None:
        record.update(read_trainer_state(state))

    vocabulary = vocab_size(directory)
    guess_loss = random_guess_loss(vocabulary)
    record["vocab_size"] = vocabulary
    record["random_guess_loss"] = guess_loss
    record["first_loss_over_guess"] = (
        record["first_loss"] / guess_loss
        if record["first_loss"] is not None and guess_loss
        else None
    )
    record.update(
        update_norms(
            directory / WEIGHTS_FILE,
            lora_alpha=float(config.get("lora_alpha") or 0.0),
            rank=int(config.get("r") or 0),
        )
    )
    record["verdict"] = verdict(
        eval_token_accuracy=record["eval_token_accuracy"],
        first_loss=record["first_loss"],
        guess_loss=guess_loss,
    )
    record["trainer_state"] = str(state) if state is not None else None
    return record
