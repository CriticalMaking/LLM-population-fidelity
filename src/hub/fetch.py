from __future__ import annotations

from pathlib import Path
from typing import Any

GROUPS = {
    "tables": ["data/**/*.csv", "data/**/*.tsv", "data/**/*.json", "data/**/*.jsonl"],
    "raw": ["data/**/raw-*.parquet"],
    "card": ["README.md"],
    "upstream": ["upstream/*.tar.gz"],
}


def allow_patterns(groups: list[str] | None) -> list[str]:
    selected = groups or list(GROUPS)
    unknown = sorted(set(selected) - set(GROUPS))
    if unknown:
        raise SystemExit(f"unknown groups: {', '.join(unknown)}; choose from {', '.join(GROUPS)}")
    return [pattern for group in selected for pattern in GROUPS[group]]


def download_dataset(
    repo_id: str,
    destination: Path,
    *,
    groups: list[str] | None = None,
    revision: str | None = None,
    force: bool = False,
) -> Path:
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    path: Any = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(destination),
        allow_patterns=allow_patterns(groups),
        force_download=force,
    )
    return Path(path)
