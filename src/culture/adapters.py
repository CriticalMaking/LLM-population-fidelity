from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from machine_bias_reproduction.config import PROJECT_ROOT
from machine_bias_reproduction.io_utils import atomic_write_json, sha256_file

from . import adapter_health
from .registry import (
    ADAPTERS_MANIFEST,
    ADAPTERS_ROOT,
    CHECKPOINT_CONDITION_DIRECTORY,
    CULTURE_ROOT,
    WEIGHTS_FILE,
    CultureModel,
)

ADAPTER_FILES: tuple[str, ...] = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "mlflow_run_id.txt",
    "processor_config.json",
    "README.md",
    "tokenizer.json",
    "tokenizer_config.json",
    "TRAINING_DONE",
    "training_args.bin",
)

REQUIRED_ADAPTER_FILES: tuple[str, ...] = (
    "adapter_config.json",
    WEIGHTS_FILE,
    "TRAINING_DONE",
)


def adapter_source(root: Path, model_key: str, culture: str) -> Path:
    directory = root / culture / model_key / CHECKPOINT_CONDITION_DIRECTORY
    if not directory.is_dir():
        raise FileNotFoundError(f"no adapter for {model_key}/{culture}: {directory}")
    missing = [name for name in REQUIRED_ADAPTER_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete adapter {directory}: missing {', '.join(sorted(missing))}"
        )
    return directory


def adapter_destination(model_key: str, culture: str) -> Path:
    return ADAPTERS_ROOT / model_key / culture


def read_adapter_config(directory: Path) -> dict[str, Any]:
    with (directory / "adapter_config.json").open(encoding="utf-8") as stream:
        payload: dict[str, Any] = json.load(stream)
    return payload


def _relative_to_project(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _adapter_record(
    source: Path,
    destination: Path,
    model: CultureModel,
    culture: str,
) -> dict[str, Any]:
    config = read_adapter_config(destination)
    declared_base = config.get("base_model_name_or_path")
    if declared_base != model.base_model_id:
        raise ValueError(
            f"{source}: adapter declares base {declared_base!r}, "
            f"registry expects {model.base_model_id!r}"
        )
    return {
        "model_key": model.key,
        "culture": culture,
        "source": str(source),
        "destination": _relative_to_project(destination),
        "base_model_name_or_path": declared_base,
        "sha256": sha256_file(destination / WEIGHTS_FILE),
        "peft_type": config.get("peft_type"),
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "target_modules": sorted(config.get("target_modules") or []),
        "exclude_modules": config.get("exclude_modules"),
        "copied_at": datetime.now(UTC).isoformat(),
        "health": adapter_health.measure(destination, config),
    }


def _stage_trainer_state(source: Path, destination: Path) -> None:
    origin = adapter_health.trainer_state_path(source)
    if origin is None:
        return
    target = destination / adapter_health.TRAINER_STATE_FILE
    if not target.is_file():
        shutil.copy2(origin, target)


def _copy_adapter(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ADAPTER_FILES:
        origin = source / name
        if origin.is_file():
            shutil.copy2(origin, destination / name)
    _stage_trainer_state(source, destination)


def _is_staged(destination: Path, source: Path) -> bool:
    staged = destination / WEIGHTS_FILE
    if not staged.is_file():
        return False
    return sha256_file(staged) == sha256_file(source / WEIGHTS_FILE)


def copy_adapters(
    root: Path,
    models: Sequence[CultureModel],
    cultures: Sequence[str],
    *,
    force: bool = False,
) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint root not found: {root}")
    records: list[dict[str, Any]] = []
    copied = 0
    reused = 0
    for model in models:
        for culture in cultures:
            source = adapter_source(root, model.key, culture)
            destination = adapter_destination(model.key, culture)
            if not force and _is_staged(destination, source):
                _stage_trainer_state(source, destination)
                reused += 1
            else:
                _copy_adapter(source, destination)
                copied += 1
            records.append(_adapter_record(source, destination, model, culture))

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint_root": str(root),
        "condition": CHECKPOINT_CONDITION_DIRECTORY,
        "policy": (
            "Only the end-of-training adapter is staged. Per-step checkpoint-N "
            "directories are training state and are excluded."
        ),
        "counts": {"copied": copied, "reused": reused, "adapters": len(records)},
        "adapters": records,
    }
    atomic_write_json(ADAPTERS_MANIFEST, manifest)
    return manifest


def staged_adapter(model_key: str, culture: str) -> Path:
    destination = adapter_destination(model_key, culture)
    missing = [name for name in REQUIRED_ADAPTER_FILES if not (destination / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"adapter not staged for {model_key}/{culture}: {destination} "
            "(run: ./run_experiment_add_culture.sh adapters)"
        )
    return destination


HEALTH_TABLE = "adapter_health.csv"

HEALTH_COLUMNS: tuple[str, ...] = (
    "model_key",
    "model_label",
    "culture",
    "verdict",
    "eval_token_accuracy",
    "token_accuracy",
    "final_eval_loss",
    "final_loss",
    "first_loss",
    "random_guess_loss",
    "first_loss_over_guess",
    "eval_f1_macro",
    "epoch",
    "global_step",
    "update_norm_mean",
    "update_norm_median",
    "update_norm_max",
    "lora_a_norm_mean",
    "lora_b_norm_mean",
    "modules",
    "r",
    "lora_alpha",
    "vocab_size",
    "sha256",
)


def health_table(models: Sequence[CultureModel], cultures: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in models:
        for culture in cultures:
            destination = adapter_destination(model.key, culture)
            if not (destination / WEIGHTS_FILE).is_file():
                continue
            config = read_adapter_config(destination)
            rows.append(
                {
                    "model_key": model.key,
                    "model_label": model.label,
                    "culture": culture,
                    "r": config.get("r"),
                    "lora_alpha": config.get("lora_alpha"),
                    "sha256": sha256_file(destination / WEIGHTS_FILE),
                    **adapter_health.measure(destination, config),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.reindex(columns=[name for name in HEALTH_COLUMNS if name in frame.columns])


def write_health(frame: pd.DataFrame) -> Path:
    destination = CULTURE_ROOT / HEALTH_TABLE
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def health_lines(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["no staged adapters to check"]
    lines = []
    for _, row in frame.iterrows():
        accuracy = row.get("eval_token_accuracy")
        shown = "n/a" if pd.isna(accuracy) else f"{float(accuracy):.1%}"
        norm = row.get("update_norm_mean")
        norm_shown = "n/a" if pd.isna(norm) else f"{float(norm):.4g}"
        lines.append(
            f"{row['model_key']}/{row['culture']}: {row['verdict']} "
            f"(eval token accuracy {shown}, mean update norm {norm_shown})"
        )
    return lines
