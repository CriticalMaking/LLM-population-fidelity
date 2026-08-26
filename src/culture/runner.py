from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from machine_bias_reproduction.analysis import run_analysis
from machine_bias_reproduction.config import PROJECT_ROOT, RunPaths
from machine_bias_reproduction.data import canonical_run_paths
from machine_bias_reproduction.inference import (
    MAX_FA_RETRIES,
    BatchInferenceBackend,
    RunContext,
    answer_parser,
    consolidate_fa,
    consolidate_ntp,
    generate_records_batched,
)
from machine_bias_reproduction.io_utils import atomic_write_json, code_revision, hardware_summary
from machine_bias_reproduction.prompts import PromptMode, PromptRecord, prompt_records
from machine_bias_reproduction.questions import Question, resolve_modes

from . import capacity as capacity_module
from .adapters import staged_adapter
from .matching import country_of
from .registry import (
    DEGENERATE_FA_RETRIES,
    PREFLIGHT_MIN_VALID_ANSWER_MASS,
    PREFLIGHT_PROMPTS,
    WEIGHTS_FILE,
    CultureModel,
    csv_stems,
    is_base,
    run_paths,
    run_slug,
)


class CultureBackend(BatchInferenceBackend, Protocol):
    @property
    def batch_size(self) -> int: ...

    def set_culture(self, culture: str) -> None: ...

    def ntp_mass_probe(self, prompts: Sequence[str]) -> list[float]: ...


ManifestWriter = Callable[..., dict[str, Any]]
TraceWriter = Callable[[Mapping[str, Sequence[PromptRecord]], RunPaths], dict[str, Any]]
Sha256File = Callable[[Path], str]


def run_context(
    model: CultureModel,
    culture: str,
    adapter: Path | None,
    backend: CultureBackend,
    sha256_file: Sha256File,
) -> RunContext:
    described = backend.describe()
    adapter_hash = sha256_file(adapter / WEIGHTS_FILE) if adapter is not None else None
    return RunContext(
        run_id=RunContext.new_run_id(),
        started_at=datetime.now(UTC).isoformat(),
        model={
            "model_key": model.key,
            "culture": culture,
            "base_model_id": model.base_model_id,
            "base_revision": described.get("base_revision"),
            "adapter_path": str(adapter.resolve()) if adapter is not None else None,
            "adapter_sha256": adapter_hash,
            "sha256": adapter_hash or described.get("base_revision"),
        },
        backend=described,
        code_revision=code_revision(PROJECT_ROOT),
        hardware=hardware_summary(),
    )


def phase_records(
    records: list[PromptRecord],
    first_countries: Sequence[str] | None,
) -> list[list[PromptRecord]]:
    if not first_countries:
        return [records]
    wanted = set(first_countries)
    countries = country_of(pd.Series([record.profile for record in records]))
    priority = [record for record, name in zip(records, countries, strict=True) if name in wanted]
    rest = [record for record, name in zip(records, countries, strict=True) if name not in wanted]
    return [phase for phase in (priority, rest) if phase]


def preflight(
    model: CultureModel,
    culture: str,
    backend: CultureBackend,
    records: list[PromptRecord],
) -> dict[str, Any]:
    sample = list(records[:PREFLIGHT_PROMPTS])
    if not sample:
        return {"prompts": 0, "mean_mass": None, "informative": None}
    masses = backend.ntp_mass_probe([record.text for record in sample])
    mean_mass = sum(masses) / len(masses)
    informative = mean_mass >= PREFLIGHT_MIN_VALID_ANSWER_MASS
    print(
        f"[{model.key}/{culture}] preflight: mean valid-answer mass "
        f"{mean_mass:.4f} over {len(sample)} prompts ({'ok' if informative else 'LOW'})",
        flush=True,
    )
    if not informative:
        print(
            f"[{model.key}/{culture}] WARNING: below "
            f"{PREFLIGHT_MIN_VALID_ANSWER_MASS:.2f}. The run continues "
            "and is reported in full, but its nEMD should not be read as an alignment result; "
            "capacity.csv is the measurement that matters for it",
            flush=True,
        )
    return {
        "prompts": len(sample),
        "mean_mass": mean_mass,
        "min_mass": min(masses),
        "max_mass": max(masses),
        "threshold": PREFLIGHT_MIN_VALID_ANSWER_MASS,
        "informative": informative,
    }


def effective_modes(
    modes: list[PromptMode],
    probe: Mapping[str, Any] | None,
    backend: CultureBackend,
) -> list[PromptMode]:
    degrades = bool(getattr(backend, "ntp_degrades_to_fa", False))
    if not degrades or probe is None or probe.get("informative") is not False:
        return list(modes)
    if len(modes) < 2 or "ntp" not in modes:
        return list(modes)
    return [mode for mode in modes if mode != "ntp"]


def _capacity_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"prompts": 0, "valid": 0, "valid_rate": 0.0}
    prompts = int(frame["prompts"].sum())
    valid = int(frame["valid"].sum())
    ntp = frame[frame["mode"] == "NTP"]
    distinct = ntp["distinct_distributions"].iloc[0] if not ntp.empty else None
    return {
        "prompts": prompts,
        "valid": valid,
        "invalid": int(frame["invalid"].sum()),
        "failed": int(frame["failed"].sum()),
        "valid_rate": valid / prompts if prompts else 0.0,
        "ntp_distinct_distributions": None if pd.isna(distinct) else int(distinct),
        "by_mode": frame.to_dict(orient="records"),
    }


def run_one(
    model: CultureModel,
    culture: str,
    question: Question,
    backend: CultureBackend,
    arguments: argparse.Namespace,
    wvs: pd.DataFrame,
    *,
    manifest_writer: ManifestWriter,
    trace_writer: TraceWriter,
    sha256_file: Sha256File,
) -> dict[str, Any]:
    paths = run_paths(model.key, culture, question)
    paths.ensure()
    backend.set_culture(culture)
    weights = None if is_base(culture) else staged_adapter(model.key, culture)
    trace = run_context(model, culture, weights, backend, sha256_file)
    print(f"[{model.key}/{culture}] {question.var} run_id: {trace.run_id}", flush=True)

    requested: list[PromptMode] = resolve_modes(arguments.mode)
    counts: dict[str, dict[str, int]] = {}
    records_by_mode: dict[str, list[PromptRecord]] = {}
    first_countries = getattr(arguments, "first_countries", None) or []
    probe_records = prompt_records(wvs, requested[0], question)
    if arguments.limit is not None:
        probe_records = probe_records[: arguments.limit]
    probe = preflight(model, culture, backend, probe_records)
    modes = effective_modes(requested, probe, backend)
    if modes != requested:
        dropped = [mode for mode in requested if mode not in modes]
        print(
            f"[{model.key}/{culture}] {', '.join(dropped)} dropped for this run "
            "because the probe found no valid-answer mass",
            flush=True,
        )
    retries = MAX_FA_RETRIES
    if probe["informative"] is False:
        retries = DEGENERATE_FA_RETRIES
        print(
            f"[{model.key}/{culture}] FA retries capped at {retries} for this run "
            "because the probe found no valid-answer mass",
            flush=True,
        )
    for mode in modes:
        records = prompt_records(wvs, mode, question)
        if arguments.limit is not None:
            records = records[: arguments.limit]
        records_by_mode[mode] = records
        phases = phase_records(records, first_countries)
        if len(phases) > 1:
            print(
                f"[{model.key}/{culture}] {mode}: {len(phases[0])} prompts from "
                f"{', '.join(first_countries)} generated first, then "
                f"{len(phases[1])} more",
                flush=True,
            )
        counts[mode] = {"generated": 0, "reused": 0, "reused_untraced": 0, "failed": 0}
        for phase in phases:
            phase_counts = generate_records_batched(
                phase,
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
            for key, value in phase_counts.items():
                counts[mode][key] += value
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
        mode_paths: dict[str, Path] = {"ntp": ntp_path, "fa": fa_path}
        available = tuple(mode for mode in ("ntp", "fa") if mode_paths[mode].is_file())
        if available:
            described = (
                f"{model.base_model_id} (not finetuned)"
                if is_base(culture)
                else f"{model.base_model_id} + {culture} LoRA"
            )
            analysis = run_analysis(
                run_slug(model.key, culture),
                question,
                model_description=described,
                csv_stems=stems,
                paper_checkpoints=False,
                modes=available,
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
    manifest["modes_requested"] = list(requested)
    manifest["modes_run"] = list(modes)
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
