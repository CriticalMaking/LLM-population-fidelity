"""How often an adapter can answer the paper's prompt at all."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from machine_bias_reproduction.config import RunPaths

from .registry import PREFLIGHT_MIN_MASS

MODES = ("ntp", "fa")
OUTCOMES = ("valid", "invalid", "failed")
EXAMPLE_LIMIT = 25
"""Rejected generations kept per mode, so a failure can be read, not just counted."""

CAPACITY_FILE = "capacity.csv"
EXAMPLES_FILE = "capacity_examples.jsonl"


@dataclass(frozen=True, slots=True)
class ModeCapacity:
    """One mode's answer-format compliance for one run."""

    mode: str
    prompts: int
    valid: int
    invalid: int
    failed: int
    mean_valid_mass: float | None
    median_valid_mass: float | None
    p10_valid_mass: float | None

    @property
    def valid_rate(self) -> float:
        """Return the share of prompts answered in the paper's exact format."""
        return self.valid / self.prompts if self.prompts else 0.0

    def as_row(self) -> dict[str, Any]:
        """Return the ``capacity.csv`` row."""
        return {
            "mode": self.mode.upper(),
            "prompts": self.prompts,
            "valid": self.valid,
            "invalid": self.invalid,
            "failed": self.failed,
            "valid_rate": self.valid_rate,
            "mean_valid_mass": self.mean_valid_mass,
            "median_valid_mass": self.median_valid_mass,
            "p10_valid_mass": self.p10_valid_mass,
        }


def _records(directory: Path) -> Iterator[dict[str, Any]]:
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as stream:
                yield json.load(stream)
        except (OSError, json.JSONDecodeError):
            continue


def classify(record: dict[str, Any], mode: str) -> str:
    """Classify one stored inference as valid, invalid or failed.

    ``failed`` means the backend never produced a candidate. ``invalid`` means
    it produced text no accepted answer matches — the case that matters here,
    being the model declining the paper's answer contract rather than the
    harness breaking.
    """
    if record.get("error") or record.get("failed"):
        return "failed"
    result = record.get("result")
    if result is None:
        return "failed"
    if mode == "ntp":
        mass = float(result.get("mass", 0.0)) if isinstance(result, dict) else 0.0
        return "valid" if mass >= PREFLIGHT_MIN_MASS else "invalid"
    return "valid" if result else "invalid"


def _rejected_text(record: dict[str, Any]) -> str | None:
    attempts = record.get("attempts") or []
    for attempt in reversed(attempts):
        raw = attempt.get("raw_text")
        if raw:
            return str(raw)
    return None


def measure(paths: RunPaths) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Walk a run's stored inferences and summarize answer-format compliance."""
    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for mode in MODES:
        counts = dict.fromkeys(OUTCOMES, 0)
        masses: list[float] = []
        kept = 0
        for record in _records(paths.raw / mode):
            outcome = classify(record, mode)
            counts[outcome] += 1
            result = record.get("result")
            if mode == "ntp" and isinstance(result, dict) and "mass" in result:
                masses.append(float(result["mass"]))
            if outcome != "valid" and kept < EXAMPLE_LIMIT:
                examples.append(
                    {
                        "mode": mode,
                        "prompt_id": record.get("prompt_id"),
                        "outcome": outcome,
                        "mass": result.get("mass") if isinstance(result, dict) else None,
                        "raw_text": _rejected_text(record),
                    }
                )
                kept += 1
        prompts = sum(counts.values())
        if not prompts:
            continue
        series = pd.Series(masses, dtype="float64")
        rows.append(
            ModeCapacity(
                mode=mode,
                prompts=prompts,
                valid=counts["valid"],
                invalid=counts["invalid"],
                failed=counts["failed"],
                mean_valid_mass=float(series.mean()) if len(series) else None,
                median_valid_mass=float(series.median()) if len(series) else None,
                p10_valid_mass=float(series.quantile(0.10)) if len(series) else None,
            ).as_row()
        )
    return pd.DataFrame(rows), examples


def write(paths: RunPaths) -> pd.DataFrame:
    """Measure a run and persist ``capacity.csv`` plus its rejected examples."""
    frame, examples = measure(paths)
    paths.outputs.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths.outputs / CAPACITY_FILE, index=False)
    with (paths.outputs / EXAMPLES_FILE).open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(json.dumps(example, sort_keys=True) + "\n")
    return frame


def read(paths: RunPaths) -> pd.DataFrame | None:
    """Read a run's capacity table, or ``None`` when it has not been measured."""
    path = paths.outputs / CAPACITY_FILE
    if not path.is_file():
        return None
    return pd.read_csv(path)


def progress_line(
    model_key: str, culture: str, mode: str, done: int, total: int, valid: int
) -> str:
    """Return the rolling counter the sweep prints while a pair runs."""
    share = 100.0 * valid / done if done else 0.0
    return f"[{model_key}/{culture}] {mode}: {done}/{total} prompts, {valid} valid ({share:.1f}%)"
