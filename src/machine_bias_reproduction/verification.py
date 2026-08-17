from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, paths_for
from .io_utils import sha256_file


@dataclass(frozen=True, slots=True)
class Checkpoint:
    metric: str
    method: str
    expected: float
    tolerance: float


CHECKPOINTS = (
    Checkpoint("wvs_rows", "WVS", 26_981.0, 0.0),
    Checkpoint("ntp_profiles", "NTP", 13_904.0, 0.0),
    Checkpoint("subpopulations", "WVS", 687.0, 0.0),
    Checkpoint("valid_token_mass_mean", "NTP", 0.984541778, 5e-10),
    Checkpoint("overall_nEMD", "NTP", 0.030770574, 5e-10),
    Checkpoint("overall_nEMD", "FA", 0.035423290, 5e-10),
    Checkpoint("median_pairwise_nEMD", "WVS", 0.114638448, 5e-10),
    Checkpoint("median_pairwise_nEMD", "NTP", 0.032728912, 5e-10),
)

CHECKPOINTED_QUESTION = "d_happy"


def verify_results(
    source: str = "archived", question: str = CHECKPOINTED_QUESTION
) -> dict[str, object]:
    if question != CHECKPOINTED_QUESTION:
        raise ValueError(
            f"the paper publishes checkpoints only for {CHECKPOINTED_QUESTION}, not {question}"
        )
    paths = paths_for(source, question)
    summary_path = paths.outputs / "summary_metrics.csv"
    manifest_path = paths.outputs / "run_manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"run analysis before verification: {summary_path}")
    summary = pd.read_csv(summary_path)
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    for checkpoint in CHECKPOINTS:
        selected = summary[
            (summary["metric"] == checkpoint.metric) & (summary["method"] == checkpoint.method)
        ]
        if len(selected) != 1:
            errors.append(f"missing or duplicate checkpoint: {checkpoint}")
            continue
        actual = float(selected.iloc[0]["value"])
        passed = bool(
            np.isclose(
                actual,
                checkpoint.expected,
                atol=checkpoint.tolerance,
                rtol=0,
            )
        )
        rows.append(
            {
                "metric": checkpoint.metric,
                "method": checkpoint.method,
                "expected": checkpoint.expected,
                "actual": actual,
                "absolute_error": abs(actual - checkpoint.expected),
                "tolerance": checkpoint.tolerance,
                "passed": passed,
            }
        )
        if not passed:
            errors.append(
                f"{checkpoint.metric}/{checkpoint.method}: "
                f"expected {checkpoint.expected}, got {actual}"
            )

    checkpoint_path = paths.outputs / "checkpoint_validation.csv"
    pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    regression_checkpoint_path = paths.outputs / "paper_regression_checkpoints.csv"
    if not regression_checkpoint_path.is_file():
        errors.append(f"missing regression checkpoint file: {regression_checkpoint_path}")
    else:
        regression_checks = pd.read_csv(regression_checkpoint_path)
        failed = regression_checks[~regression_checks["passed"].astype(bool)]
        if not failed.empty:
            errors.append("paper regression checkpoints failed:\n" + failed.to_string(index=False))

    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    for relative, expected_hash in manifest["artifacts"].items():
        artifact = Path(relative)
        if not artifact.is_absolute():
            artifact = PROJECT_ROOT / artifact
        if not artifact.is_file():
            errors.append(f"missing artifact: {relative}")
        elif sha256_file(artifact) != expected_hash:
            errors.append(f"artifact hash mismatch: {relative}")
    if errors:
        raise RuntimeError("\n".join(errors))
    return {
        "ok": True,
        "source": source,
        "checkpoints": len(rows),
        "checkpoint_file": str(checkpoint_path),
        "regression_checkpoint_file": str(regression_checkpoint_path),
        "manifest_artifacts": len(manifest["artifacts"]),
    }
