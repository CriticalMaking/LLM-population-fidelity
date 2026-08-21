from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from culture import (
    ADAPTERS_MANIFEST,
    BASE_ARM,
    DEFAULT_CHECKPOINT_ROOT,
    SERVED_BACKEND,
    copy_adapters,
    health_lines,
    health_table,
    is_base,
    resolve_cultures,
    resolve_finetuned_cultures,
    resolve_models,
    served_keys,
    served_models,
    staged_adapter,
    write_health,
)
from culture.api_backend import REASONING_EFFORT, REASONING_EFFORTS
from culture.figures import compare_cultures
from culture.mds import culture_mds
from culture.runner import run_one
from culture.served_smoke import SMOKE_ATTEMPTS, SMOKE_FA_MAX_TOKENS, SMOKE_PROMPTS
from culture.summary import culture_summary

from .analysis import run_analysis
from .config import (
    GLOBAL_SEED,
    MODEL_SHA256,
    PROJECT_ROOT,
    RunPaths,
    paths_for,
)
from .data import canonical_run_paths, load_wvs
from .inference import (
    EVENT_LOG_NAME,
    InferenceBackend,
    LlamaCppBackend,
    RunContext,
    answer_parser,
    audit_traces,
    consolidate_fa,
    consolidate_ntp,
    generate_records,
)
from .io_utils import (
    append_jsonl,
    atomic_write_json,
    code_revision,
    hardware_summary,
    sha256_file,
)
from .prompts import PromptMode, PromptRecord, prompt_records
from .provenance import MANIFEST_PATH, create_manifest, verify_upstream
from .questions import (
    DEFAULT_QUESTION,
    QUESTION_NAMES,
    resolve_modes,
    resolve_question,
    resolve_questions,
)
from .verification import verify_results


def _json_print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _gpu_layers(value: str | None) -> int:
    if value is not None:
        return int(value)
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        print("WARNING: nvidia-smi not found; using CPU inference", flush=True)
        return 0
    result = subprocess.run(
        [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        print(f"GPU detected: {result.stdout.strip()}; offloading all layers", flush=True)
        return -1
    print("WARNING: no working NVIDIA driver detected; using CPU inference", flush=True)
    return 0


def _run_context(
    model_path: Path,
    model_hash: str,
    backend: InferenceBackend,
) -> RunContext:
    return RunContext(
        run_id=RunContext.new_run_id(),
        started_at=datetime.now(UTC).isoformat(),
        model={
            "filename": model_path.name,
            "path": str(model_path.resolve()),
            "sha256": model_hash,
            "expected_sha256": MODEL_SHA256,
        },
        backend=backend.describe(),
        code_revision=code_revision(PROJECT_ROOT),
        hardware=hardware_summary(),
    )


def _inference_manifest(
    paths: RunPaths,
    trace: RunContext,
    *,
    modes: Sequence[str],
    limit: int | None,
    legacy_unseeded_fa: bool,
    counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 2,
        "run_id": trace.run_id,
        "started_at": trace.started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "code_revision": trace.code_revision,
        "model": dict(trace.model),
        "backend": dict(trace.backend),
        "parameters": {
            "modes": list(modes),
            "limit": limit,
            "global_seed": GLOBAL_SEED,
            "legacy_unseeded_fa": legacy_unseeded_fa,
        },
        "counts": counts,
        "hardware": dict(trace.hardware),
        "event_log": str((paths.logs / EVENT_LOG_NAME).relative_to(paths.outputs)),
    }
    atomic_write_json(paths.outputs / "inference_manifest.json", payload)
    append_jsonl(paths.logs / "runs.jsonl", payload)
    return payload


def command_archived(arguments: argparse.Namespace) -> None:
    if not MANIFEST_PATH.exists():
        create_manifest()
    verify_upstream(full=False)
    results = []
    for question in resolve_questions(arguments.questions):
        result = run_analysis("archived", question)
        if question.var == DEFAULT_QUESTION:
            result["verification"] = verify_results("archived", question.var)
        results.append(result)
    _json_print(results if len(results) > 1 else results[0])


def command_fresh(arguments: argparse.Namespace) -> None:
    model_path = Path(arguments.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Mixtral model not found: {model_path}")
    model_hash = sha256_file(model_path)
    if model_hash != MODEL_SHA256:
        raise RuntimeError(f"model SHA-256 mismatch: expected {MODEL_SHA256}, got {model_hash}")
    question = resolve_question(arguments.question)
    paths = paths_for("fresh", question.var)
    paths.ensure()
    modes: list[PromptMode] = resolve_modes(arguments.mode)
    threads = arguments.threads or max(1, (os.cpu_count() or 2) - 1)
    backend = LlamaCppBackend(
        model_path,
        question,
        threads=threads,
        gpu_layers=_gpu_layers(arguments.gpu_layers),
    )
    trace = _run_context(model_path, model_hash, backend)
    print(f"run_id: {trace.run_id}", flush=True)
    wvs = load_wvs()
    counts: dict[str, dict[str, int]] = {}
    records_by_mode: dict[str, Sequence[PromptRecord]] = {}
    for mode in modes:
        records = prompt_records(wvs, mode, question)
        if arguments.limit is not None:
            records = records[: arguments.limit]
        records_by_mode[mode] = records
        counts[mode] = generate_records(
            records,
            backend,
            paths,
            trace=trace,
            legacy_unseeded_fa=arguments.legacy_unseeded_fa,
            force=arguments.force,
            parse=answer_parser(question),
        )
        if counts[mode]["failed"]:
            raise RuntimeError(f"{mode} generation had {counts[mode]['failed']} failures")

    analysis = None
    if arguments.limit is None:
        ntp_path, fa_path = canonical_run_paths(paths.outputs, question)
        if "ntp" in modes:
            consolidate_ntp(records_by_mode["ntp"], paths, ntp_path, question)
        if "fa" in modes:
            consolidate_fa(records_by_mode["fa"], paths, fa_path, question)
        if ntp_path.is_file() and fa_path.is_file():
            analysis = run_analysis("fresh", question)
    _inference_manifest(
        paths,
        trace,
        modes=modes,
        limit=arguments.limit,
        legacy_unseeded_fa=arguments.legacy_unseeded_fa,
        counts=counts,
    )
    _json_print(
        {
            "run_id": trace.run_id,
            "question": question.var,
            "counts": counts,
            "trace": _write_trace_index(records_by_mode, paths),
            "analysis": analysis,
        }
    )


def _write_trace_index(
    records_by_mode: Mapping[str, Sequence[PromptRecord]],
    paths: RunPaths,
) -> dict[str, Any]:
    frame, summary = audit_traces(records_by_mode, paths)
    destination = paths.outputs / "inference_trace.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    atomic_write_json(paths.outputs / "inference_trace_summary.json", summary)
    return summary


def command_trace(arguments: argparse.Namespace) -> None:
    question = resolve_question(arguments.question)
    paths = paths_for("fresh", question.var)
    paths.ensure()
    wvs = load_wvs()
    modes: list[PromptMode] = resolve_modes(arguments.mode)
    records_by_mode: dict[str, Sequence[PromptRecord]] = {
        mode: prompt_records(wvs, mode, question) for mode in modes
    }
    summary = _write_trace_index(records_by_mode, paths)
    _json_print(summary)
    if not summary["ok"]:
        raise SystemExit(1)


def command_culture_adapters(arguments: argparse.Namespace) -> None:
    models = resolve_models(arguments.models)
    cultures = resolve_finetuned_cultures(arguments.cultures)
    manifest = copy_adapters(
        Path(arguments.source),
        models,
        cultures,
        force=arguments.force,
    )
    for line in health_lines(health_table(models, cultures)):
        print(line, flush=True)
    _json_print({"counts": manifest["counts"], "manifest": str(ADAPTERS_MANIFEST)})


def command_culture_health(arguments: argparse.Namespace) -> None:
    frame = health_table(
        resolve_models(arguments.models),
        resolve_finetuned_cultures(arguments.cultures),
    )
    for line in health_lines(frame):
        print(line, flush=True)
    if frame.empty:
        _json_print({"adapters": 0, "table": None})
        return
    scoped = bool(arguments.models or arguments.cultures)
    destination = None if scoped else write_health(frame)
    _json_print(
        {
            "adapters": len(frame),
            "verdicts": {
                str(name): int(count) for name, count in frame["verdict"].value_counts().items()
            },
            "table": None if destination is None else str(destination),
        }
    )


def _culture_groups(model_key: str, cultures: list[str]) -> list[tuple[list[str], dict[str, Path]]]:
    finetuned = [culture for culture in cultures if not is_base(culture)]
    groups: list[tuple[list[str], dict[str, Path]]] = []
    if finetuned:
        adapters = {culture: staged_adapter(model_key, culture) for culture in finetuned}
        groups.append((finetuned, adapters))
    if any(is_base(culture) for culture in cultures):
        groups.append(([BASE_ARM], {}))
    return groups


def command_culture(arguments: argparse.Namespace) -> None:
    from culture.backend import TransformersBackend

    verify_upstream(full=False)
    models = resolve_models(arguments.models)
    if arguments.models is None:
        served = [model for model in models if model.backend != "transformers"]
        if served:
            print(
                f"skipping {', '.join(model.key for model in served)}: "
                "an api model runs only when named with --models",
                flush=True,
            )
        models = [model for model in models if model.backend == "transformers"]
    cultures = resolve_cultures(arguments.cultures)
    questions = resolve_questions(arguments.questions)
    wvs = load_wvs()
    results: list[dict[str, Any]] = []
    for question in questions:
        for model in models:
            for group, adapters in _culture_groups(model.key, cultures):
                backend: Any
                if model.backend == "openai":
                    from culture.api_backend import OpenAIBackend

                    if adapters:
                        raise ValueError(
                            f"{model.key} has no culture-MLLM finetuning; run --cultures base"
                        )
                    print(
                        f"calling {model.label} through the api "
                        f"for {question.var} — {', '.join(group)}",
                        flush=True,
                    )
                    backend = OpenAIBackend(
                        model,
                        question,
                        batch_size=arguments.batch_size,
                        fa_max_new_tokens=arguments.fa_max_tokens,
                        reasoning_effort=arguments.reasoning_effort,
                    )
                else:
                    print(
                        f"loading {model.base_model_id} ({model.quantization or 'unquantized'}) "
                        f"for {question.var} — {', '.join(group)}",
                        flush=True,
                    )
                    backend = TransformersBackend(
                        model,
                        adapters,
                        question,
                        batch_size=arguments.batch_size,
                        fa_max_new_tokens=arguments.fa_max_tokens,
                    )
                for culture in group:
                    results.append(
                        run_one(
                            model,
                            culture,
                            question,
                            backend,
                            arguments,
                            wvs,
                            manifest_writer=_inference_manifest,
                            trace_writer=_write_trace_index,
                            sha256_file=sha256_file,
                        )
                    )
                del backend

    comparison: list[dict[str, Any]] | None = None
    if not arguments.skip_compare and arguments.limit is None:
        comparison = [compare_cultures(models, cultures, question) for question in questions]
    _json_print({"runs": results, "comparison": comparison})


def command_culture_served_smoke(arguments: argparse.Namespace) -> None:
    from culture.served_smoke import run_smoke

    models = resolve_models(arguments.models) if arguments.models else served_models()
    local = [model for model in models if model.backend != SERVED_BACKEND]
    if local:
        raise ValueError(
            f"{', '.join(model.key for model in local)} is not served through the api; "
            f"this smoke compares {', '.join(served_keys())}"
        )
    wvs = load_wvs()
    results = [
        run_smoke(
            models,
            question,
            wvs,
            prompts=arguments.prompts,
            attempts=arguments.attempts,
            fa_max_tokens=arguments.fa_max_tokens,
            batch_size=arguments.batch_size,
            reasoning_effort=arguments.reasoning_effort,
        )
        for question in resolve_questions(arguments.questions)
    ]
    _json_print(results)


def command_culture_compare(arguments: argparse.Namespace) -> None:
    models = resolve_models(arguments.models)
    cultures = resolve_cultures(arguments.cultures)
    _json_print(
        [
            compare_cultures(models, cultures, question)
            for question in resolve_questions(arguments.questions)
        ]
    )


def command_culture_mds(arguments: argparse.Namespace) -> None:
    models = resolve_models(arguments.models)
    cultures = resolve_cultures(arguments.cultures)
    questions = resolve_questions(arguments.questions)
    _json_print(
        [
            culture_mds(
                models,
                cultures,
                question,
                all_countries=arguments.all_countries,
                outcomes=arguments.outcomes and question is questions[0],
            )
            for question in questions
        ]
    )


def command_culture_summary(arguments: argparse.Namespace) -> None:
    models = resolve_models(arguments.models)
    cultures = resolve_cultures(arguments.cultures)
    _json_print(culture_summary(models, cultures, arguments.questions))


def command_reports(arguments: argparse.Namespace) -> None:
    from reports.report import build_reports

    _json_print(
        build_reports(
            arguments.section,
            questions=arguments.questions,
            series=arguments.series,
        )
    )


def command_verify(arguments: argparse.Namespace) -> None:
    upstream = verify_upstream(full=arguments.full)
    payload: dict[str, object] = {"upstream": upstream}
    archived_summary = paths_for("archived", DEFAULT_QUESTION).outputs / "summary_metrics.csv"
    if archived_summary.is_file():
        payload["results"] = verify_results("archived", DEFAULT_QUESTION)
    _json_print(payload)


def command_manifest(_: argparse.Namespace) -> None:
    _json_print(create_manifest())


def command_prompt_check(arguments: argparse.Namespace) -> None:
    from .config import UPSTREAM_DATA

    wvs = load_wvs()
    report: dict[str, Any] = {}
    for question in resolve_questions(arguments.questions):
        entry: dict[str, Any] = {}
        for mode, directory in (("ntp", "NTP"), ("fa", "FA")):
            records = prompt_records(wvs, cast(PromptMode, mode), question)
            root = UPSTREAM_DATA / "prompts" / directory / question.var
            mismatched = [
                record.prompt_id
                for record in records
                if (root / f"{record.prompt_id}.txt").is_file()
                and record.text.encode() != (root / f"{record.prompt_id}.txt").read_bytes()
            ]
            entry[mode] = {"prompts": len(records), "mismatched": mismatched[:10]}
            if mismatched:
                entry[mode]["mismatched_count"] = len(mismatched)
        report[question.var] = entry
    _json_print(report)
    if any(modes[mode].get("mismatched") for modes in report.values() for mode in ("ntp", "fa")):
        raise SystemExit(1)


def _add_question_argument(parser: argparse.ArgumentParser, *, plural: bool) -> None:
    if plural:
        parser.add_argument(
            "--questions",
            nargs="+",
            choices=(*QUESTION_NAMES, "all"),
            default=[DEFAULT_QUESTION],
            help=f"outcome questions (default: {DEFAULT_QUESTION}; 'all' for every question)",
        )
    else:
        parser.add_argument(
            "--question",
            choices=QUESTION_NAMES,
            default=DEFAULT_QUESTION,
            help=f"outcome question (default: {DEFAULT_QUESTION})",
        )


def _add_selection_arguments(parser: argparse.ArgumentParser, *, cultures_help: str) -> None:
    parser.add_argument("--models", nargs="+", help="model keys (default: all)")
    parser.add_argument("--cultures", nargs="+", help=cultures_help)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="machine-bias-reproduction",
        description="Reproduce Machine Bias: four WVS questions, NTP and FA.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    archived = subparsers.add_parser("archived", help="analyze archived NTP and FA outputs")
    archived.add_argument("--force", action="store_true", help="accepted for shell compatibility")
    _add_question_argument(archived, plural=True)
    archived.set_defaults(handler=command_archived)

    fresh = subparsers.add_parser("fresh", help="run fresh resumable Mixtral inference")
    fresh.add_argument("--model", required=True, help="path to the pinned Q4_K_M GGUF")
    fresh.add_argument("--mode", choices=("ntp", "fa", "all"), default="all")
    fresh.add_argument("--limit", type=int, help="smoke-test only the first N prompts")
    fresh.add_argument("--force", action="store_true", help="overwrite completed prompt results")
    fresh.add_argument("--threads", type=int, help="CPU inference threads")
    fresh.add_argument(
        "--gpu-layers", help="-1 for all layers, 0 for CPU; auto-detected by default"
    )
    fresh.add_argument(
        "--legacy-unseeded-fa",
        action="store_true",
        help="mirror the paper's non-repeatable FA sampling",
    )
    _add_question_argument(fresh, plural=False)
    fresh.set_defaults(handler=command_fresh)

    adapters = subparsers.add_parser(
        "culture-adapters",
        help="stage culture-finetuned LoRA weights under models/culture",
    )
    adapters.add_argument(
        "--source",
        default=str(DEFAULT_CHECKPOINT_ROOT),
        help="checkpoint root holding <culture>/<model>/cultural directories",
    )
    _add_selection_arguments(adapters, cultures_help="cultures (default: all nine)")
    adapters.add_argument("--force", action="store_true", help="re-copy staged weights")
    adapters.set_defaults(handler=command_culture_adapters)

    culture_health = subparsers.add_parser(
        "culture-health",
        help="check staged adapters for divergence before spending GPU time",
    )
    _add_selection_arguments(culture_health, cultures_help="cultures (default: all nine)")
    culture_health.set_defaults(handler=command_culture_health)

    culture = subparsers.add_parser(
        "culture",
        help="run culture-finetuned LLM inference and analysis",
    )
    _add_selection_arguments(
        culture,
        cultures_help="arms: the nine cultures and 'base' (default: base and all nine)",
    )
    culture.add_argument(
        "--first-countries",
        nargs="+",
        help="generate these countries' prompts first; ordering only, same final result",
    )
    culture.add_argument("--mode", choices=("ntp", "fa", "all"), default="all")
    culture.add_argument("--limit", type=int, help="smoke-test only the first N prompts")
    culture.add_argument("--batch-size", type=int, help="override the model's default batch size")
    culture.add_argument("--fa-max-tokens", type=int, default=12, help="FA generation budget")
    culture.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default=REASONING_EFFORT,
        help=f"reasoning budget for a served model; ignored by the local ones "
        f"(default: {REASONING_EFFORT})",
    )
    culture.add_argument("--force", action="store_true", help="overwrite completed prompt results")
    culture.add_argument(
        "--legacy-unseeded-fa",
        action="store_true",
        help="mirror the paper's non-repeatable FA sampling",
    )
    culture.add_argument(
        "--skip-compare",
        action="store_true",
        help="do not rebuild the cross-culture comparison afterwards",
    )
    _add_question_argument(culture, plural=True)
    culture.set_defaults(handler=command_culture)

    culture_served_smoke = subparsers.add_parser(
        "culture-served-smoke",
        help="compare the served models on the same prompts before paying for a run",
    )
    culture_served_smoke.add_argument(
        "--models",
        nargs="+",
        help=f"served model keys (default: {', '.join(served_keys())})",
    )
    culture_served_smoke.add_argument(
        "--prompts",
        type=int,
        default=SMOKE_PROMPTS,
        help=f"prompts per model (default: {SMOKE_PROMPTS})",
    )
    culture_served_smoke.add_argument(
        "--attempts",
        type=int,
        default=SMOKE_ATTEMPTS,
        help=f"attempts per prompt (default: {SMOKE_ATTEMPTS}, the run's own cap)",
    )
    culture_served_smoke.add_argument(
        "--fa-max-tokens",
        type=int,
        default=SMOKE_FA_MAX_TOKENS,
        help=f"FA generation budget (default: {SMOKE_FA_MAX_TOKENS})",
    )
    culture_served_smoke.add_argument(
        "--batch-size",
        type=int,
        help="override the model's default concurrency",
    )
    culture_served_smoke.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default=REASONING_EFFORT,
        help=(
            f"reasoning budget, held equal across the models compared (default: {REASONING_EFFORT})"
        ),
    )
    _add_question_argument(culture_served_smoke, plural=True)
    culture_served_smoke.set_defaults(handler=command_culture_served_smoke)

    culture_compare = subparsers.add_parser(
        "culture-compare",
        help="build cross-culture figures and reports from existing outputs",
    )
    _add_selection_arguments(culture_compare, cultures_help="arms (default: base and all nine)")
    _add_question_argument(culture_compare, plural=True)
    culture_compare.set_defaults(handler=command_culture_compare)

    culture_mds_parser = subparsers.add_parser(
        "culture-mds",
        help="build the culture MDS plates from existing outputs",
    )
    _add_selection_arguments(
        culture_mds_parser,
        cultures_help="arms (default: all; only matched cultures are drawn)",
    )
    culture_mds_parser.add_argument(
        "--all-countries",
        action="store_true",
        help="also draw the unrestricted plate over every WVS country",
    )
    culture_mds_parser.add_argument(
        "--outcomes",
        action="store_true",
        help="also draw the per-culture row across happiness, politics and religion",
    )
    _add_question_argument(culture_mds_parser, plural=True)
    culture_mds_parser.set_defaults(handler=command_culture_mds)

    culture_summary_parser = subparsers.add_parser(
        "culture-summary",
        help="build the cross-question summary figures and tables",
    )
    _add_selection_arguments(
        culture_summary_parser, cultures_help="arms (default: base and all nine)"
    )
    _add_question_argument(culture_summary_parser, plural=True)
    culture_summary_parser.set_defaults(handler=command_culture_summary)

    reports = subparsers.add_parser("reports", help="reproduce the paper's tables and figures")
    reports.add_argument(
        "section",
        nargs="?",
        default="all",
        choices=("all", "main", "appendix", "robustness", "tables", "figures", "smoke"),
    )
    reports.add_argument("--series", nargs="+", help="model series (default: all six)")
    _add_question_argument(reports, plural=True)
    reports.set_defaults(handler=command_reports)

    trace = subparsers.add_parser("trace", help="audit per-inference traces of a fresh run")
    trace.add_argument("--mode", choices=("ntp", "fa", "all"), default="all")
    _add_question_argument(trace, plural=False)
    trace.set_defaults(handler=command_trace)

    verify = subparsers.add_parser("verify", help="verify upstream and generated artifacts")
    verify.add_argument(
        "--full",
        action="store_true",
        help="CRC-check the ZIP and stat every extracted payload file",
    )
    verify.set_defaults(handler=command_verify)

    manifest = subparsers.add_parser("manifest", help="regenerate upstream/MANIFEST.json")
    manifest.set_defaults(handler=command_manifest)

    prompt_check = subparsers.add_parser(
        "prompt-check",
        help="verify prompts byte-for-byte against the upstream files",
    )
    _add_question_argument(prompt_check, plural=True)
    prompt_check.set_defaults(handler=command_prompt_check)
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
