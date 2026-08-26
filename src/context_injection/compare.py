from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd


def _pairs(values: Sequence[str]) -> dict[str, Path]:
    pairs: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator:
            raise ValueError(f"expected CONDITION=PATH, got {value!r}")
        pairs[name] = Path(path)
    return pairs


def _read(condition: str, root: Path, filename: str) -> pd.DataFrame:
    path = root / filename
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame.insert(0, "condition", condition)
    return frame


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    present = [frame for frame in frames if not frame.empty]
    return pd.concat(present, ignore_index=True) if present else pd.DataFrame()


def stack_analysis(conditions: Mapping[str, Path]) -> dict[str, pd.DataFrame]:
    return {
        "summary_metrics": _concat(
            [_read(name, root, "summary_metrics.csv") for name, root in conditions.items()]
        ),
        "subpopulation_distances": _concat(
            [
                _read(name, root, "subpopulation_distances.csv")
                for name, root in conditions.items()
            ]
        ),
    }


def write_tables(tables: Mapping[str, pd.DataFrame], out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, frame in tables.items():
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stack Machine Bias outputs across conditions")
    parser.add_argument("--condition", action="append", required=True, help="CONDITION=output/path")
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    written = write_tables(stack_analysis(_pairs(args.condition)), args.out)
    print({"written": [str(path) for path in written]})


if __name__ == "__main__":
    main()
