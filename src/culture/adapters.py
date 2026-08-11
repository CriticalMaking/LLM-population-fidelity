"""Staging the trained LoRA adapters into the repository."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from machine_bias_reproduction.config import PROJECT_ROOT
from machine_bias_reproduction.io_utils import atomic_write_json, sha256_file

from .registry import ADAPTERS_MANIFEST, ADAPTERS_ROOT, CONDITION, CultureModel

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
    "adapter_model.safetensors",
    "TRAINING_DONE",
)
"""Adapter files that must exist for a checkpoint to be considered complete."""

WEIGHTS_FILE = "adapter_model.safetensors"


def adapter_source(root: Path, model_key: str, culture: str) -> Path:
    """Locate one trained adapter in a checkpoint tree.

    The upstream layout is ``<root>/<culture>/<model_key>/cultural``. A
    directory without ``TRAINING_DONE`` is an unfinished run, and is rejected
    rather than silently used.
    """
    directory = root / culture / model_key / CONDITION
    if not directory.is_dir():
        raise FileNotFoundError(f"no adapter for {model_key}/{culture}: {directory}")
    missing = [name for name in REQUIRED_ADAPTER_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete adapter {directory}: missing {', '.join(sorted(missing))}"
        )
    return directory


def adapter_destination(model_key: str, culture: str) -> Path:
    """Return the staged location of one adapter under ``models/``."""
    return ADAPTERS_ROOT / model_key / culture


def read_adapter_config(directory: Path) -> dict[str, Any]:
    """Read one adapter's PEFT configuration."""
    with (directory / "adapter_config.json").open(encoding="utf-8") as stream:
        payload: dict[str, Any] = json.load(stream)
    return payload


def _relative_to_project(path: Path) -> str:
    """Return a repo-relative path when possible, else an absolute one."""
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
    }


def _copy_adapter(source: Path, destination: Path) -> None:
    """Copy one adapter's top-level files, excluding per-step checkpoints."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in ADAPTER_FILES:
        origin = source / name
        if origin.is_file():
            shutil.copy2(origin, destination / name)


def _is_staged(destination: Path, source: Path) -> bool:
    """Return whether a destination already holds this adapter's weights."""
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
    """Stage the selected adapters under ``models/culture`` and manifest them.

    A repeat run re-hashes what is already staged, so a truncated earlier copy
    is replaced rather than trusted.
    """
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
                reused += 1
            else:
                _copy_adapter(source, destination)
                copied += 1
            records.append(_adapter_record(source, destination, model, culture))

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint_root": str(root),
        "condition": CONDITION,
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
    """Return a staged adapter directory, requiring that it was copied first."""
    destination = adapter_destination(model_key, culture)
    missing = [name for name in REQUIRED_ADAPTER_FILES if not (destination / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"adapter not staged for {model_key}/{culture}: {destination} "
            "(run: ./run_experiment_add_culture.sh adapters)"
        )
    return destination
