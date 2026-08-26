from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from machine_bias_reproduction.prompts import PromptRecord
from machine_bias_reproduction.questions import PromptMode, resolve_modes


def _mode(value: str) -> PromptMode:
    modes = resolve_modes(value)
    if len(modes) != 1:
        raise ValueError(f"expected one prompt mode, got {value!r}")
    return modes[0]


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def prompt_record(row: Mapping[str, Any], *, qualify_prompt_id: bool = False) -> PromptRecord:
    condition = str(row["condition"])
    prompt_id = str(row["prompt_id"])
    if qualify_prompt_id:
        prompt_id = f"{condition}:{prompt_id}"
    return PromptRecord(
        prompt_id=prompt_id,
        profile=str(row["profile"]),
        mode=_mode(str(row["mode"])),
        text=str(row["prompt_text"]),
    )


def records_by_condition(
    path: Path,
    *,
    qualify_prompt_id: bool = False,
) -> dict[str, list[PromptRecord]]:
    grouped: dict[str, list[PromptRecord]] = {}
    for row in load_rows(path):
        condition = str(row["condition"])
        grouped.setdefault(condition, []).append(
            prompt_record(row, qualify_prompt_id=qualify_prompt_id)
        )
    return grouped
