from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path

EXCLUDED_DIRS = {"raw", "logs", "tex", "served_smoke"}
EXCLUDED_NAMES = {"inference_manifest.json", "comparison_manifest.json"}
MIRRORED_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".md"}
CHUNK_DEPTH = 2


def is_mirrored(relative: Path) -> bool:
    return (
        relative.suffix in MIRRORED_SUFFIXES
        and relative.name not in EXCLUDED_NAMES
        and not EXCLUDED_DIRS.intersection(relative.parts[:-1])
    )


def mirror_files(outputs: Path, data_dir: Path) -> int:
    copied = 0
    for dirpath, dirnames, filenames in os.walk(outputs):
        dirnames[:] = sorted(name for name in dirnames if name not in EXCLUDED_DIRS)
        for filename in sorted(filenames):
            source = Path(dirpath) / filename
            if not is_mirrored(source.relative_to(outputs)):
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
            elif (outputs / relative).exists() and is_mirrored(relative):
                continue
            staged.unlink()
            pruned += 1
        if not os.listdir(dirpath):
            os.rmdir(dirpath)
    return pruned


def build_staging(
    outputs: Path,
    staging: Path,
    *,
    force: bool = False,
    on_event: Callable[[str], None] = print,
) -> dict[str, int]:
    from .convert import convert_all

    data_dir = staging / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    mirrored = mirror_files(outputs, data_dir)
    counts = convert_all(outputs, data_dir, force=force, on_event=on_event)
    pruned = prune_files(outputs, data_dir)
    on_event(f"mirrored {mirrored} files, pruned {pruned}")
    staged = [path for path in data_dir.rglob("*") if path.is_file()]
    return {
        **counts,
        "mirrored": mirrored,
        "pruned": pruned,
        "tables": sum(1 for path in staged if path.suffix in MIRRORED_SUFFIXES),
        "parquet": sum(1 for path in staged if path.suffix == ".parquet"),
    }


def dataset_chunks(staging: Path) -> list[Path]:
    return sorted(_walk_chunks(staging / "data", CHUNK_DEPTH))


def _walk_chunks(directory: Path, depth: int) -> list[Path]:
    if not directory.is_dir():
        return []
    children = sorted(entry for entry in directory.iterdir() if entry.is_dir())
    if depth == 0 or not children:
        return [directory]
    return [chunk for child in children for chunk in _walk_chunks(child, depth - 1)]


def dataset_files(staging: Path) -> list[Path]:
    chunks = dataset_chunks(staging)
    return sorted(
        path
        for path in staging.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(staging).parts)
        and not any(chunk in path.parents for chunk in chunks)
    )
