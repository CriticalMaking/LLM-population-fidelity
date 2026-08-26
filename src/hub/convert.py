from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

MODES = ("fa", "ntp")
PROFILE_FIELDS = ("country", "wave", "age", "gender", "education", "employment", "marital")
PROFILE_SEPARATOR = "§"
BATCH_ROWS = 4096

ATTEMPT_TYPE = pa.struct(
    [
        ("index", pa.int32()),
        ("accepted", pa.bool_()),
        ("seed", pa.int64()),
        ("duration_ms", pa.float64()),
        ("parsed", pa.string()),
        ("raw_text", pa.string()),
    ]
)

RAW_SCHEMA = pa.schema(
    [
        ("record_path", pa.string()),
        ("mode", pa.string()),
        ("prompt_id", pa.string()),
        ("profile", pa.string()),
        *((field, pa.string()) for field in PROFILE_FIELDS),
        ("prompt_sha256", pa.string()),
        ("prompt_text", pa.string()),
        ("answer", pa.string()),
        ("mass", pa.float64()),
        ("result_json", pa.string()),
        ("seed", pa.int64()),
        ("attempt_count", pa.int32()),
        ("attempts", pa.list_(ATTEMPT_TYPE)),
        ("run_id", pa.string()),
        ("run_started_at", pa.string()),
        ("created_at", pa.string()),
        ("duration_ms", pa.float64()),
        ("code_revision", pa.string()),
        ("model_ref", pa.string()),
        ("model_key", pa.string()),
        ("culture", pa.string()),
        ("model_sha256", pa.string()),
        ("adapter_sha256", pa.string()),
        ("backend_name", pa.string()),
        ("backend_version", pa.string()),
        ("sampling_json", pa.string()),
        ("schema_version", pa.int32()),
    ]
)


def record_row(record: dict[str, Any], record_path: str) -> dict[str, Any]:
    prompt = record.get("prompt") or {}
    result = record.get("result") or {}
    trace = record.get("trace") or {}
    backend = trace.get("backend") or {}
    model = trace.get("model") or {}
    sampling = trace.get("sampling") or {}
    profile = record.get("profile") or ""
    parts = dict(zip(PROFILE_FIELDS, profile.split(PROFILE_SEPARATOR), strict=False))
    attempts = [
        {
            "index": attempt.get("index"),
            "accepted": attempt.get("accepted"),
            "seed": attempt.get("seed"),
            "duration_ms": attempt.get("duration_ms"),
            "parsed": attempt.get("parsed"),
            "raw_text": attempt.get("raw_text"),
        }
        for attempt in record.get("attempts") or []
    ]
    return {
        "record_path": record_path,
        "mode": record.get("mode"),
        "prompt_id": record.get("prompt_id"),
        "profile": profile,
        **{field: parts.get(field) for field in PROFILE_FIELDS},
        "prompt_sha256": prompt.get("sha256"),
        "prompt_text": prompt.get("text"),
        "answer": result.get("answer"),
        "mass": result.get("mass"),
        "result_json": json.dumps(result, ensure_ascii=False, sort_keys=True),
        "seed": record.get("seed"),
        "attempt_count": record.get("attempt_count"),
        "attempts": attempts or None,
        "run_id": trace.get("run_id"),
        "run_started_at": trace.get("run_started_at"),
        "created_at": trace.get("created_at"),
        "duration_ms": trace.get("duration_ms"),
        "code_revision": trace.get("code_revision"),
        "model_ref": model.get("filename") or model.get("base_model_id"),
        "model_key": model.get("model_key"),
        "culture": model.get("culture"),
        "model_sha256": model.get("sha256"),
        "adapter_sha256": model.get("adapter_sha256"),
        "backend_name": backend.get("name"),
        "backend_version": backend.get("version"),
        "sampling_json": json.dumps(sampling, ensure_ascii=False, sort_keys=True),
        "schema_version": record.get("schema_version"),
    }


def convert_mode(run_dir: Path, mode: str, destination: Path, *, force: bool = False) -> str:
    source_dir = run_dir / "raw" / mode
    if not source_dir.is_dir():
        return "absent"
    entries = [entry for entry in os.scandir(source_dir) if entry.name.endswith(".json")]
    if not entries:
        return "empty"
    sources = sorted(entry.name for entry in entries)
    newest = max(entry.stat().st_mtime_ns for entry in entries)
    if not force and destination.exists():
        current = (
            pq.ParquetFile(destination).metadata.num_rows == len(sources)
            and newest <= destination.stat().st_mtime_ns
        )
        if current:
            return "current"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".parquet.partial")
    writer = pq.ParquetWriter(partial, RAW_SCHEMA, compression="zstd")
    batch: list[dict[str, Any]] = []
    for name in sources:
        with open(source_dir / name, encoding="utf-8") as handle:
            record = json.load(handle)
        batch.append(record_row(record, f"raw/{mode}/{name}"))
        if len(batch) >= BATCH_ROWS:
            writer.write_table(pa.Table.from_pylist(batch, schema=RAW_SCHEMA))
            batch = []
    if batch:
        writer.write_table(pa.Table.from_pylist(batch, schema=RAW_SCHEMA))
    writer.close()
    partial.replace(destination)
    return f"written ({len(sources)} rows)"


def run_dirs(outputs: Path) -> list[Path]:
    return sorted(path.parent for path in outputs.rglob("raw") if path.is_dir())


def convert_all(
    outputs: Path,
    data_dir: Path,
    *,
    force: bool = False,
    on_event: Callable[[str], None] = print,
) -> dict[str, int]:
    counts = {"runs": 0, "written": 0, "current": 0, "skipped": 0}
    for run_dir in run_dirs(outputs):
        relative = run_dir.relative_to(outputs)
        counts["runs"] += 1
        for mode in MODES:
            destination = data_dir / relative / f"raw-{mode}.parquet"
            status = convert_mode(run_dir, mode, destination, force=force)
            if status.startswith("written"):
                counts["written"] += 1
            elif status == "current":
                counts["current"] += 1
            else:
                counts["skipped"] += 1
            on_event(f"{relative} raw-{mode}: {status}")
    return counts
