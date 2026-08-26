from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_CONFIG, wvs_csv
from .leakage import blocked_variables
from .theories.inglehart_welzel import InglehartWelzelMapping, load_mapping, source_variables


def audit_mapping(
    wvs: pd.DataFrame,
    mapping: InglehartWelzelMapping,
    *,
    question: str,
) -> dict[str, Any]:
    variables = source_variables(mapping)
    blocked = blocked_variables(question)
    missing = sorted(variable for variable in variables if variable not in wvs.columns)
    leaks = sorted(set(variables).intersection(blocked))
    items: list[dict[str, Any]] = []
    for item in mapping.items:
        observed = set()
        if item.variable in wvs.columns:
            observed = set(wvs[item.variable].dropna().astype(str).unique())
        score_keys = set(item.scores)
        items.append(
            {
                "variable": item.variable,
                "dimension": item.dimension,
                "present": item.variable in wvs.columns,
                "observed_values": sorted(observed),
                "scored_values": sorted(score_keys),
                "unscored_observed_values": sorted(observed - score_keys),
                "unused_score_values": sorted(score_keys - observed),
            }
        )
    return {
        "question": question,
        "source_variables": list(variables),
        "missing_variables": missing,
        "target_leakage_variables": leaks,
        "items": items,
        "ok": not missing and not leaks,
    }


def write_report(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a theory mapping against a WVS CSV")
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path, help="Root config JSON")
    parser.add_argument("--wvs", type=Path, help="Path to WVS CSV")
    parser.add_argument("--theory-map", required=True, type=Path, help="Theory mapping JSON")
    parser.add_argument("--question", required=True, help="Target question for leakage audit")
    parser.add_argument("--out", type=Path, help="Output audit JSON path")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    wvs_path = args.wvs or wvs_csv(args.config)
    report = audit_mapping(
        pd.read_csv(wvs_path),
        load_mapping(args.theory_map),
        question=args.question,
    )
    if args.out:
        write_report(report, args.out)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
