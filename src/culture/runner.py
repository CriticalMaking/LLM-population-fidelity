"""Driving one culture adapter through inference, capacity and analysis."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from machine_bias_reproduction.analysis import run_analysis
from machine_bias_reproduction.config import PROJECT_ROOT
from machine_bias_reproduction.data import canonical_run_paths
from machine_bias_reproduction.inference import (
    MAX_FA_RETRIES,
    RunContext,
    answer_parser,
    consolidate_fa,
    consolidate_ntp,
    generate_records_batched,
)
from machine_bias_reproduction.io_utils import atomic_write_json, code_revision, hardware_summary
from machine_bias_reproduction.prompts import PromptMode, PromptRecord, prompt_records
from machine_bias_reproduction.questions import Question

from . import capacity as capacity_module
from .adapters import WEIGHTS_FILE, staged_adapter
from .registry import (
    DEGENERATE_FA_RETRIES,
    PREFLIGHT_MIN_MASS,
    PREFLIGHT_PROMPTS,
    CultureModel,
    csv_stems,
    run_paths,
    run_slug,
)


def run_context(
    model: CultureModel,
    culture: str,
    adapter: Path,
    backend: Any,
    sha256_file: Any,
) -> RunContext:
    """Build the run identity for one culture adapter.

    With no single model file to hash, identity is the adapter weights plus the
    resolved base-model snapshot. ``sha256`` carries the adapter digest so the
    existing trace index stays populated.
    """
    described = backend.describe()
    adapter_hash = sha256_file(adapter / WEIGHTS_FILE)
    return RunContext(
        run_id=RunContext.new_run_id(),
        started_at=datetime.now(UTC).isoformat(),
        model={
            "model_key": model.key,
            "culture": culture,
            "base_model_id": model.base_model_id,
            "base_revision": described.get("base_revision"),
            "adapter_path": str(adapter.resolve()),
            "adapter_sha256": adapter_hash,
            "sha256": adapter_hash,
        },
        backend=described,
        code_revision=code_revision(PROJECT_ROOT),
        hardware=hardware_summary(),
    )


def preflight(
    model: CultureModel,
    culture: str,
    backend: Any,
    records: list[PromptRecord],
) -> dict[str, Any]:
    """Measure how much probability an adapter puts on any valid answer.

    Under the paper's prompt an instruction-tuned base may spend its whole
    distribution on other formats. Probing surfaces that in the manifest before
    hours of inference rather than in a plot afterwards. The run proceeds either
    way.
    """
    sample = list(records[:PREFLIGHT_PROMPTS])
    if not sample:
        return {"prompts": 0, "mean_mass": None, "informative": None}
    masses = backend.ntp_mass_probe([record.text for record in sample])
    mean_mass = sum(masses) / len(masses)
    informative = mean_mass >= PREFLIGHT_MIN_MASS
    print(
        f"[{model.key}/{culture}] preflight: mean valid-answer mass "
        f"{mean_mass:.4f} over {len(sample)} prompts ({'ok' if informative else 'LOW'})",
        flush=True,
    )
    if not informative:
        print(
            f"[{model.key}/{culture}] WARNING: below {PREFLIGHT_MIN_MASS:.2f}. The run continues "
            "and is reported in full, but its nEMD should not be read as an alignment result; "
            "capacity.csv is the measurement that matters for it",
            flush=True,
        )
    return {
        "prompts": len(sample),
        "mean_mass": mean_mass,
        "min_mass": min(masses),
        "max_mass": max(masses),
        "threshold": PREFLIGHT_MIN_MASS,
        "informative": informative,
    }


def _capacity_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"prompts": 0, "valid": 0, "valid_rate": 0.0}
    prompts = int(frame["prompts"].sum())
    valid = int(frame["valid"].sum())
    return {
        "prompts": prompts,
        "valid": valid,
        "invalid": int(frame["invalid"].sum()),
        "failed": int(frame["failed"].sum()),
        "valid_rate": valid / prompts if prompts else 0.0,
        "by_mode": frame.to_dict(orient="records"),
    }


def run_one(
    model: CultureModel,
    culture: str,
    question: Question,
    backend: Any,
    arguments: Any,
    wvs: pd.DataFrame,
    *,
    manifest_writer: Any,
    trace_writer: Any,
    sha256_file: Any,
) -> dict[str, Any]:
    """Run inference, capacity measurement and analysis for one pair.

    A pair whose probe finds no valid-answer mass drops to
    ``DEGENERATE_FA_RETRIES``, which stops a degenerate adapter from spending
    days on generations known in advance to fail.

    Analysis runs on whatever was answered. Paper checkpoints stay off because
    the paper's regression targets describe Mixtral, not these models.
    """
    paths = run_paths(model.key, culture, question)
    paths.ensure()
    backend.set_culture(culture)
    trace = run_context(model, culture, staged_adapter(model.key, culture), backend, sha256_file)
    print(f"[{model.key}/{culture}] {question.var} run_id: {trace.run_id}", flush=True)

    modes: list[PromptMode] = (
        ["ntp", "fa"] if arguments.mode == "all" else [cast(PromptMode, arguments.mode)]
    )
    counts: dict[str, dict[str, int]] = {}
    records_by_mode: dict[str, list[PromptRecord]] = {}
    probe: dict[str, Any] | None = None
    retries = MAX_FA_RETRIES
    for mode in modes:
        records = prompt_records(wvs, mode, question)
        if arguments.limit is not None:
            records = records[: arguments.limit]
        records_by_mode[mode] = records
        if probe is None:
            probe = preflight(model, culture, backend, records)
            if probe["informative"] is False:
                retries = DEGENERATE_FA_RETRIES
                print(
                    f"[{model.key}/{culture}] FA retries capped at {retries} for this run "
                    "because the probe found no valid-answer mass",
                    flush=True,
                )
        counts[mode] = generate_records_batched(
            records,
            backend,
            paths,
            trace=trace,
            batch_size=backend.batch_size,
            fa_max_tokens=arguments.fa_max_tokens,
            legacy_unseeded_fa=arguments.legacy_unseeded_fa,
            force=arguments.force,
            parse=answer_parser(question),
            max_fa_retries=retries,
        )
        if counts[mode]["failed"]:
            print(
                f"[{model.key}/{culture}] {mode}: {counts[mode]['failed']} prompts produced no "
                "answer the paper's rules accept; see raw/*/**.failed.json",
                flush=True,
            )

    capacity_frame = capacity_module.write(paths)
    summary = _capacity_summary(capacity_frame)
    print(
        f"[{model.key}/{culture}] capacity: {summary['valid']}/{summary['prompts']} prompts "
        f"answered in the paper's format ({summary['valid_rate']:.1%})",
        flush=True,
    )

    analysis: dict[str, Any] | None = None
    if arguments.limit is None:
        stems = csv_stems(model.key, culture, question)
        ntp_path, fa_path = canonical_run_paths(paths.outputs, question, stems)
        if "ntp" in modes:
            consolidate_ntp(records_by_mode["ntp"], paths, ntp_path, question, skip_missing=True)
        if "fa" in modes:
            consolidate_fa(records_by_mode["fa"], paths, fa_path, question, skip_missing=True)
        if ntp_path.is_file() and fa_path.is_file():
            analysis = run_analysis(
                run_slug(model.key, culture),
                question,
                label=model.run_label(culture),
                short_label=f"{model.label} ({culture})",
                model_description=f"{model.base_model_id} + {culture} LoRA",
                csv_stems=stems,
                paper_checkpoints=False,
            )

    manifest = manifest_writer(
        paths,
        trace,
        modes=modes,
        limit=arguments.limit,
        legacy_unseeded_fa=arguments.legacy_unseeded_fa,
        counts=counts,
    )
    manifest["preflight"] = probe
    manifest["capacity"] = summary
    manifest["question"] = question.var
    atomic_write_json(paths.outputs / "inference_manifest.json", manifest)
    return {
        "model": model.key,
        "culture": culture,
        "question": question.var,
        "run_id": trace.run_id,
        "preflight": probe,
        "capacity": summary,
        "counts": counts,
        "trace": trace_writer(records_by_mode, paths),
        "analysis": analysis,
    }
