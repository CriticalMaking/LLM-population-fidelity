from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"
DEFAULT_REPO = Path.home() / "TowardsSociallyGroundedAISafety"
REMOTE_URL = "git@hf.co:datasets/Neemias/TowardsSociallyGroundedAISafety"

EXCLUDED_DIRS = {"raw", "logs", "tex"}
MIRRORED_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".md"}
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

GITATTRIBUTES = """*.csv filter=lfs diff=lfs merge=lfs -text
*.tsv filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
"""


def record_row(record: dict, record_path: str) -> dict:
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


def convert_mode(run_dir: Path, mode: str, destination: Path) -> str:
    source_dir = run_dir / "raw" / mode
    if not source_dir.is_dir():
        return "absent"
    sources = sorted(entry.name for entry in os.scandir(source_dir) if entry.name.endswith(".json"))
    if not sources:
        return "empty"
    if destination.exists() and pq.ParquetFile(destination).metadata.num_rows == len(sources):
        return "current"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".parquet.partial")
    writer = pq.ParquetWriter(partial, RAW_SCHEMA, compression="zstd")
    batch: list[dict] = []
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


def find_run_dirs(outputs: Path) -> list[Path]:
    return sorted(path.parent for path in outputs.rglob("raw") if path.is_dir())


def mirror_files(outputs: Path, data_dir: Path) -> int:
    copied = 0
    for dirpath, dirnames, filenames in os.walk(outputs):
        dirnames[:] = sorted(name for name in dirnames if name not in EXCLUDED_DIRS)
        for filename in sorted(filenames):
            source = Path(dirpath) / filename
            if source.suffix not in MIRRORED_SUFFIXES:
                continue
            target = data_dir / source.relative_to(outputs)
            if target.exists():
                source_stat, target_stat = source.stat(), target.stat()
                if (
                    source_stat.st_size == target_stat.st_size
                    and source_stat.st_mtime_ns == target_stat.st_mtime_ns
                ):
                    continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
    return copied


def prune_files(outputs: Path, data_dir: Path) -> int:
    pruned = 0
    if not data_dir.is_dir():
        return pruned
    for dirpath, _, filenames in os.walk(data_dir, topdown=False):
        for filename in filenames:
            staged = Path(dirpath) / filename
            relative = staged.relative_to(data_dir)
            if filename.startswith("raw-") and filename.endswith(".parquet"):
                if (outputs / relative.parent / "raw" / filename[4:-8]).is_dir():
                    continue
            elif (outputs / relative).exists():
                continue
            staged.unlink()
            pruned += 1
        if not os.listdir(dirpath):
            os.rmdir(dirpath)
    return pruned


def write_card(repo: Path) -> None:
    (repo / "README.md").write_text(CARD, encoding="utf-8")


def git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments], check=False, capture_output=True, text=True
    )


def ensure_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    if not (repo / ".git").exists():
        subprocess.run(["git", "init", "-b", "main", str(repo)], check=True)
    if REMOTE_URL not in git(repo, "remote", "-v").stdout:
        git(repo, "remote", "add", "origin", REMOTE_URL)
    git(repo, "lfs", "install", "--local")
    attributes = repo / ".gitattributes"
    existing = attributes.read_text(encoding="utf-8") if attributes.exists() else ""
    present = set(existing.splitlines())
    missing = [line for line in GITATTRIBUTES.splitlines() if line not in present]
    if missing:
        prefix = existing if not existing or existing.endswith("\n") else existing + "\n"
        attributes.write_text(prefix + "\n".join(missing) + "\n", encoding="utf-8")


def push(repo: Path, message: str) -> None:
    git(repo, "add", "-A")
    if git(repo, "diff", "--cached", "--quiet").returncode != 0:
        commit = git(repo, "commit", "-m", message)
        print(commit.stdout.strip() or commit.stderr.strip())
    else:
        print("nothing to commit")
    result = git(repo, "push", "-u", "origin", "main")
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode != 0:
        sys.exit("push failed - check SSH key registration and remote history")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--push", action="store_true")
    parser.add_argument(
        "--message", default=f"data: sync outputs snapshot {date.today().isoformat()}"
    )
    arguments = parser.parse_args()

    ensure_repo(arguments.repo)
    data_dir = arguments.repo / "data"
    copied = mirror_files(arguments.outputs, data_dir)
    pruned = prune_files(arguments.outputs, data_dir)
    print(f"mirrored {copied} files, pruned {pruned}")
    for run_dir in find_run_dirs(arguments.outputs):
        relative = run_dir.relative_to(arguments.outputs)
        for mode in MODES:
            status = convert_mode(run_dir, mode, data_dir / relative / f"raw-{mode}.parquet")
            print(f"{relative} raw-{mode}: {status}")
    write_card(arguments.repo)
    if arguments.push:
        push(arguments.repo, arguments.message)


CARD = """\
---
pretty_name: Towards Socially Grounded AI Safety
license: cc-by-4.0
language:
  - en
tags:
  - synthetic
  - survey-simulation
  - social-bias
  - llm-outputs
configs:
  - config_name: raw
    data_files:
      - split: fa
        path: data/**/raw-fa.parquet
      - split: ntp
        path: data/**/raw-ntp.parquet
---

# Towards Socially Grounded AI Safety

LLM-generated survey responses from a reproduction of Boelaert et al. (2025), *Machine Bias*,
extended with culture-finetuned (German QLoRA) model arms. Each model simulates World Values
Survey respondents defined by country, survey wave, age, gender, education, employment, and
marital status, answering four opinion questions.

## Experimental arms

| Arm | Model | Variants |
| --- | --- | --- |
| `fresh` | Mixtral-8x7B-v0.1 Q4_K_M (llama.cpp) | reproduction of the original protocol |
| `culture/gemma4_31b` | google/gemma-4-31B-it | `base`, `german` (QLoRA) |
| `culture/gemma4_e4b` | google/gemma-4-e4b-it | `base`, `german` (QLoRA) |
| `culture/qwen3_vl_8b` | Qwen/Qwen3-VL-8B-Thinking | `base`, `german` (QLoRA) |
| `culture/muse_glimmer_30b` | meta-models/Muse-Glimmer-30B | `base`, `german` (QLoRA) |
| `archived` | upstream Mixtral outputs, reanalysed | derived tables only (no raw records) |

Questions: `d_happy` (happiness), `d_polpos` (political position, 10-point),
`d_religiousp` (religiosity), `d_trust` (interpersonal trust).
Modes: `fa` (free answer, sampled completions) and `ntp` (next-token probabilities over the
answer options).

## Layout

```
data/<arm>/<question>/
  raw-fa.parquet        per-prompt FA records (one row per simulated respondent)
  raw-ntp.parquet       per-prompt NTP records
  FA-*.csv, NTP-*.csv   consolidated response tables (id, profile, answer)
  inference_trace.csv   flat provenance index; record_path joins to the parquet rows
  subpopulation_distances.csv, regression_fit.csv, full_coefficients.csv,
  full_standardized_coefficients.csv, ...   derived per-run analysis tables
data/culture/model_standardized_coefficients.csv   pooled coefficient table behind the
                                                   fig_model_standardized_coefficients plates
data/reports/           cross-arm paper tables (nEMD, regressions, F-tests)
```

## Raw parquet schema

One row per prompt. `profile` is the `§`-delimited respondent cell, also split into
`country`, `wave`, `age`, `gender`, `education`, `employment`, `marital`. FA rows carry
`answer`, `attempts` (list of `{index, accepted, seed, duration_ms, parsed, raw_text}`), and
`seed`; NTP rows carry `mass` plus the per-option probability distribution in `result_json`.
Provenance columns: `run_id`, `created_at`, `model_ref` (GGUF filename or Hub model id),
`model_sha256` / `adapter_sha256`, `backend_name`, `backend_version`, `sampling_json`,
`code_revision`, `schema_version`. `record_path` matches the `record_path` column of the
sibling `inference_trace.csv`. Local filesystem paths from the trace blocks are omitted.
"""


if __name__ == "__main__":
    main()
