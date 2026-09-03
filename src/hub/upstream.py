from __future__ import annotations

import gzip
import tarfile
from pathlib import Path

from machine_bias_reproduction.config import PROJECT_ROOT
from machine_bias_reproduction.provenance import is_substantive

ARCHIVE_NAME = "redraw-subset.tar.gz"
PACKAGE = Path("upstream") / "extracted" / "Machine-Bias-replication"
SUBSET: tuple[str, ...] = (
    "code",
    "data/WVS",
    "data/Linear",
    "data/subpops.csv",
    "data/LLM-outputs/csv",
)


def subset_roots(project_root: Path = PROJECT_ROOT) -> list[Path]:
    return [project_root / PACKAGE / part for part in SUBSET]


def subset_files(project_root: Path = PROJECT_ROOT) -> list[Path]:
    roots = subset_roots(project_root)
    missing = [root.relative_to(project_root).as_posix() for root in roots if not root.exists()]
    if missing:
        raise FileNotFoundError(f"upstream subset incomplete, missing {', '.join(missing)}")
    found: list[Path] = []
    for root in roots:
        found.extend([root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()])
    return sorted(p for p in found if is_substantive(p.relative_to(project_root).as_posix()))


def staged_archive(staging: Path) -> Path:
    return staging / "upstream" / ARCHIVE_NAME


def staged_archive_bytes(staging: Path) -> int:
    archive = staged_archive(staging)
    return archive.stat().st_size if archive.is_file() else 0


def build_upstream_archive(staging: Path, project_root: Path = PROJECT_ROOT) -> Path:
    files = subset_files(project_root)
    destination = staged_archive(staging)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        destination.open("wb") as handle,
        gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path in files:
            archive.add(path, arcname=path.relative_to(project_root).as_posix(), recursive=False)
    return destination
