from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

from machine_bias_reproduction.prompts import CONTEXT_ORDER
from machine_bias_reproduction.questions import PromptMode, resolve_question

from .config import DEFAULT_CONFIG, wvs_csv
from .conditions import FIRST_PILOT_CONDITIONS, first_pilot_specs
from .prompts import ContextPromptRecord, prompt_records
from .theories.inglehart_welzel import DEFAULT_MAPPING, InglehartWelzelMapping, load_mapping

REQUIRED_COLUMNS = ("profile", *CONTEXT_ORDER)


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _record_dict(
    record: ContextPromptRecord,
    *,
    question: str,
    condition: str,
) -> dict[str, Any]:
    return {
        "question": question,
        "condition": condition,
        "mode": record.mode,
        "prompt_id": record.prompt_id,
        "profile": record.profile,
        "prompt_sha256": hashlib.sha256(record.text.encode("utf-8")).hexdigest(),
        "prompt_text": record.text,
        "context": record.context.as_dict(),
    }


def validate_columns(
    wvs: pd.DataFrame,
    *,
    mode: PromptMode,
    raw_variables: Sequence[str],
    selected_variables: Sequence[str],
) -> None:
    required = set(REQUIRED_COLUMNS)
    required.update(raw_variables)
    required.update(selected_variables)
    if mode == "fa":
        required.add("id")
    missing = sorted(required - set(wvs.columns))
    if missing:
        raise ValueError(f"WVS frame is missing columns: {', '.join(missing)}")


def build_rows(
    wvs: pd.DataFrame,
    *,
    question: str,
    mode: PromptMode,
    raw_variables: Sequence[str],
    selected_variables: Sequence[str],
    conditions: Sequence[str] = FIRST_PILOT_CONDITIONS,
    mapping: InglehartWelzelMapping = DEFAULT_MAPPING,
) -> Iterable[dict[str, Any]]:
    validate_columns(
        wvs,
        mode=mode,
        raw_variables=raw_variables,
        selected_variables=selected_variables,
    )
    outcome = resolve_question(question)
    specs = first_pilot_specs(
        raw_variables=raw_variables,
        selected_variables=selected_variables,
    )
    unknown = set(conditions) - set(specs)
    if unknown:
        raise ValueError(f"unknown conditions: {', '.join(sorted(unknown))}")
    for condition in conditions:
        for record in prompt_records(wvs, mode, outcome, specs[condition], mapping=mapping):
            yield _record_dict(record, question=outcome.var, condition=condition)


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            count += 1
    return count


def write_manifest(
    path: Path,
    *,
    count: int,
    wvs_path: Path,
    out_path: Path,
    question: str,
    mode: PromptMode,
    raw_variables: Sequence[str],
    selected_variables: Sequence[str],
    conditions: Sequence[str],
    theory_map: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "record_count": count,
        "wvs_path": str(wvs_path),
        "out_path": str(out_path),
        "question": question,
        "mode": mode,
        "raw_variables": list(raw_variables),
        "selected_variables": list(selected_variables),
        "conditions": list(conditions),
        "theory_map": None if theory_map is None else str(theory_map),
    }
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export context-injected prompt records as JSONL")
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path, help="Root config JSON")
    parser.add_argument("--wvs", type=Path, help="Path to WVS CSV")
    parser.add_argument("--out", required=True, type=Path, help="Output JSONL path")
    parser.add_argument("--manifest", type=Path, help="Manifest JSON path")
    parser.add_argument("--theory-map", type=Path, help="Verified Inglehart-Welzel mapping JSON")
    parser.add_argument("--question", required=True, help="Machine Bias target question")
    parser.add_argument("--mode", choices=("ntp", "fa"), default="ntp")
    parser.add_argument("--raw-vars", required=True, help="Comma-separated C1/C2 source variables")
    parser.add_argument("--selected-vars", dest="selected_vars", help="C3/C4 source variables")
    parser.add_argument("--theory-vars", dest="selected_vars", help=argparse.SUPPRESS)
    parser.add_argument(
        "--conditions",
        default=",".join(FIRST_PILOT_CONDITIONS),
        help="Comma-separated subset of C0,C1,C2,C3,C4",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    if not args.selected_vars:
        arg_parser.error("--selected-vars is required")
    wvs_path = args.wvs or wvs_csv(args.config)
    wvs = pd.read_csv(wvs_path)
    raw_variables = _csv_tuple(args.raw_vars)
    selected_variables = _csv_tuple(args.selected_vars)
    conditions = _csv_tuple(args.conditions)
    mapping = load_mapping(args.theory_map) if args.theory_map else DEFAULT_MAPPING
    rows = build_rows(
        wvs,
        question=args.question,
        mode=cast(PromptMode, args.mode),
        raw_variables=raw_variables,
        selected_variables=selected_variables,
        conditions=conditions,
        mapping=mapping,
    )
    count = write_jsonl(rows, args.out)
    manifest = args.manifest or args.out.with_suffix(args.out.suffix + ".manifest.json")
    write_manifest(
        manifest,
        count=count,
        wvs_path=wvs_path,
        out_path=args.out,
        question=args.question,
        mode=cast(PromptMode, args.mode),
        raw_variables=raw_variables,
        selected_variables=selected_variables,
        conditions=conditions,
        theory_map=args.theory_map,
    )
    print(f"wrote {count} prompt records to {args.out}")


if __name__ == "__main__":
    main()
