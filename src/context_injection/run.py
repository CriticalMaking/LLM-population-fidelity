from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from machine_bias_reproduction.config import GLOBAL_SEED, PROJECT_ROOT, RunPaths
from machine_bias_reproduction.data import canonical_run_paths
from machine_bias_reproduction.io_utils import (
    atomic_write_json,
    code_revision,
    hardware_summary,
    sha256_file,
    sha256_text,
)
from machine_bias_reproduction.questions import resolve_question
from machine_bias_reproduction.prompts import PromptMode, PromptRecord

from .config import DEFAULT_CONFIG, load_config
from .records import load_rows, records_by_condition

SAFE_NAME = re.compile(r"^[A-Za-z0-9_]+$")


def _safe(value: str, label: str) -> str:
    if not SAFE_NAME.match(value):
        raise ValueError(f"unsafe {label}: {value!r}; use letters, numbers, and underscores")
    return value


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def default_run_name(prompts: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", prompts.stem).strip("_") or "context_run"


def condition_source(run_name: str, condition: str) -> str:
    run_name = _safe(run_name, "run name")
    condition = _safe(condition, "condition")
    return f"culture/context_{run_name}/{condition}"


def records_by_mode(records: Sequence[PromptRecord]) -> dict[PromptMode, list[PromptRecord]]:
    grouped: dict[PromptMode, list[PromptRecord]] = {}
    for record in records:
        grouped.setdefault(record.mode, []).append(record)
    return grouped


def condition_paths(run_name: str, condition: str, question: str) -> RunPaths:
    return RunPaths(condition_source(run_name, condition), _safe(question, "question"))


def infer_question(prompts: Path) -> str:
    rows = load_rows(prompts)
    if not rows:
        raise ValueError(f"no prompt records in {prompts}")
    questions = {str(row["question"]) for row in rows}
    if len(questions) != 1:
        raise ValueError(f"expected one question, found {sorted(questions)}")
    return questions.pop()


def _result_path(paths: RunPaths, record: PromptRecord) -> Path:
    return paths.raw / record.mode / f"{sha256_text(record.prompt_id)[:24]}.json"


def _read_result(
    paths: RunPaths,
    record: PromptRecord,
    *,
    skip_missing: bool,
) -> dict[str, Any] | None:
    path = _result_path(paths, record)
    if skip_missing and not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        payload: dict[str, Any] = json.load(stream)
    return payload


def _consolidate_ntp(
    records: Sequence[PromptRecord],
    paths: RunPaths,
    destination: Path,
    question: str,
    *,
    skip_missing: bool,
) -> int:
    outcome = resolve_question(question)
    rows = []
    for record in records:
        payload = _read_result(paths, record, skip_missing=skip_missing)
        if payload is not None:
            rows.append({"profile": record.profile, **payload["result"]})
    frame = pd.DataFrame(rows, columns=["profile", "mass", *outcome.answer_columns])
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return len(rows)


def _consolidate_fa(
    records: Sequence[PromptRecord],
    paths: RunPaths,
    destination: Path,
    question: str,
    *,
    skip_missing: bool,
) -> int:
    outcome = resolve_question(question)
    rows = []
    for record in records:
        payload = _read_result(paths, record, skip_missing=skip_missing)
        if payload is not None:
            rows.append(
                {
                    "id": record.prompt_id,
                    "profile": record.profile,
                    outcome.var: payload["result"]["answer"],
                }
            )
    frame = pd.DataFrame(rows, columns=["id", "profile", outcome.var])
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return len(rows)


def _run_context(model_path: Path, model_hash: str, backend: Any) -> Any:
    from machine_bias_reproduction.inference import RunContext

    return RunContext(
        run_id=RunContext.new_run_id(),
        started_at=datetime.now(timezone.utc).isoformat(),
        model={
            "filename": model_path.name,
            "path": str(model_path.resolve()),
            "sha256": model_hash,
        },
        backend=backend.describe(),
        code_revision=code_revision(PROJECT_ROOT),
        hardware=hardware_summary(),
    )


def _write_trace_index(
    grouped: Mapping[PromptMode, Sequence[PromptRecord]],
    paths: RunPaths,
) -> dict[str, Any]:
    from machine_bias_reproduction.inference import audit_traces

    frame, summary = audit_traces(grouped, paths)
    paths.outputs.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths.outputs / "inference_trace.csv", index=False)
    atomic_write_json(paths.outputs / "inference_trace_summary.json", summary)
    return summary


def _copy_fa_csv(source: Path, destination: Path, question: str) -> int:
    frame = pd.read_csv(source).loc[:, ["id", "profile", question]]
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return len(frame)


def archived_fa_path(config_path: Path) -> Path:
    config = load_config(config_path)
    return Path(config["llm_outputs_csv_root"]) / "FA-Mixtral-8x7B.csv"


def consolidate_condition(
    records: Sequence[PromptRecord],
    paths: RunPaths,
    *,
    question: str,
    fa_csv: Path | None = None,
    skip_missing: bool = False,
) -> dict[str, int]:
    grouped = records_by_mode(records)
    ntp_path, fa_path = canonical_run_paths(paths.outputs, question)
    counts: dict[str, int] = {}
    if "ntp" in grouped:
        counts["ntp"] = _consolidate_ntp(
            grouped["ntp"],
            paths,
            ntp_path,
            question,
            skip_missing=skip_missing,
        )
    if "fa" in grouped:
        counts["fa"] = _consolidate_fa(
            grouped["fa"],
            paths,
            fa_path,
            question,
            skip_missing=skip_missing,
        )
    elif fa_csv is not None:
        counts["fa"] = _copy_fa_csv(fa_csv, fa_path, question)
    return counts


def _manifest(
    *,
    prompts: Path,
    run_name: str,
    question: str,
    condition: str,
    paths: RunPaths,
    records: Sequence[PromptRecord],
    generation: Mapping[str, Any] | None,
    consolidation: Mapping[str, int] | None,
    analysis: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompts": str(prompts),
        "run_name": run_name,
        "question": question,
        "condition": condition,
        "source": paths.source,
        "outputs": str(paths.outputs),
        "records": {mode: len(items) for mode, items in records_by_mode(records).items()},
        "generation": generation,
        "consolidation": consolidation,
        "analysis": analysis,
    }


def run_condition(
    records: Sequence[PromptRecord],
    *,
    prompts: Path,
    run_name: str,
    condition: str,
    question: str,
    backend: Any | None = None,
    trace: Any | None = None,
    force: bool = False,
    fa_csv: Path | None = None,
    skip_missing: bool = False,
    analysis: bool = False,
) -> dict[str, Any]:
    paths = condition_paths(run_name, condition, question)
    paths.ensure()
    grouped = records_by_mode(records)
    generation = None
    if backend is not None:
        from machine_bias_reproduction.inference import answer_parser, generate_records

        if trace is None:
            raise ValueError("trace is required when backend is supplied")
        generation = {}
        for mode, mode_records in grouped.items():
            generation[mode] = generate_records(
                mode_records,
                backend,
                paths,
                trace=trace,
                force=force,
                parse=answer_parser(question),
            )
            if generation[mode]["failed"]:
                raise RuntimeError(f"{condition} {mode} generation had failures")
    consolidation = consolidate_condition(
        records,
        paths,
        question=question,
        fa_csv=fa_csv,
        skip_missing=skip_missing,
    )
    trace_summary = _write_trace_index(grouped, paths) if backend is not None else None
    analysis_result = None
    if analysis:
        from machine_bias_reproduction.analysis import run_analysis

        analysis_result = run_analysis(paths.source, question, paper_checkpoints=False)
    manifest = _manifest(
        prompts=prompts,
        run_name=run_name,
        question=question,
        condition=condition,
        paths=paths,
        records=records,
        generation={"counts": generation, "trace": trace_summary} if generation else None,
        consolidation=consolidation,
        analysis=analysis_result,
    )
    atomic_write_json(paths.outputs / "context_run_manifest.json", manifest)
    return manifest


def _plan(
    grouped: Mapping[str, Sequence[PromptRecord]],
    *,
    run_name: str,
    question: str,
) -> list[dict[str, Any]]:
    rows = []
    for condition, records in grouped.items():
        paths = condition_paths(run_name, condition, question)
        rows.append(
            {
                "condition": condition,
                "source": paths.source,
                "outputs": str(paths.outputs),
                "records": {
                    mode: len(items) for mode, items in records_by_mode(records).items()
                },
            }
        )
    return rows


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run context prompt records by condition")
    parser.add_argument("--prompts", required=True, type=Path, help="Context prompt JSONL")
    parser.add_argument("--run-name", help="Safe run name; default is prompt filename")
    parser.add_argument("--question", help="Target question; default inferred from JSONL")
    parser.add_argument("--conditions", help="Comma-separated subset; default all")
    parser.add_argument("--model", type=Path, help="GGUF model path for live inference")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Records per condition, for tiny tests")
    parser.add_argument("--force", action="store_true", help="Regenerate existing raw outputs")
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Run existing analysis per condition",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Consolidate available raw files",
    )
    parser.add_argument("--fa-csv", type=Path, help="FA CSV to pair with NTP context outputs")
    parser.add_argument(
        "--use-archived-fa",
        action="store_true",
        help="Use configured archived FA CSV when no FA prompts are present",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path, help="Root config JSON")
    parser.add_argument("--dry-run", action="store_true", help="Print condition plan only")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    run_name = args.run_name or default_run_name(args.prompts)
    question = args.question or infer_question(args.prompts)
    grouped = records_by_condition(args.prompts)
    if args.conditions:
        selected = set(_csv_tuple(args.conditions))
        grouped = {key: value for key, value in grouped.items() if key in selected}
    if args.limit is not None:
        grouped = {key: value[: args.limit] for key, value in grouped.items()}
    if not grouped:
        raise ValueError("no conditions selected")

    plan = _plan(grouped, run_name=run_name, question=question)
    if args.dry_run:
        print(json.dumps({"question": question, "run_name": run_name, "plan": plan}, indent=2))
        return

    fa_csv = args.fa_csv
    if args.use_archived_fa and fa_csv is None:
        fa_csv = archived_fa_path(args.config)

    backend = None
    trace = None
    if args.model is not None:
        from machine_bias_reproduction.inference import LlamaCppBackend

        model_hash = sha256_file(args.model)
        backend = LlamaCppBackend(
            args.model,
            question,
            threads=args.threads,
            gpu_layers=args.gpu_layers,
        )
        trace = _run_context(args.model, model_hash, backend)

    manifests = [
        run_condition(
            records,
            prompts=args.prompts,
            run_name=run_name,
            condition=condition,
            question=question,
            backend=backend,
            trace=trace,
            force=args.force,
            fa_csv=fa_csv,
            skip_missing=args.skip_missing,
            analysis=args.analysis,
        )
        for condition, records in grouped.items()
    ]
    print(json.dumps({"conditions": manifests}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
