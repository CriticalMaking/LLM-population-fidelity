from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

IGNORE_PATTERNS = ["*.partial", "*.tmp", ".DS_Store"]
DIGEST_BLOCK = 1 << 20


class UploadState:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[str, str] = {}
        if path.is_file():
            try:
                self.entries = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.entries = {}

    def is_current(self, key: str, digest: str) -> bool:
        return self.entries.get(key) == digest

    def record(self, key: str, digest: str) -> None:
        self.entries[key] = digest
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.entries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def uploaded_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and not any(fnmatch(path.name, pattern) for pattern in IGNORE_PATTERNS)
    )


def folder_digest(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in uploaded_files(folder):
        digest.update(f"{path.relative_to(folder)}:{path.stat().st_size}\n".encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(DIGEST_BLOCK), b""):
                digest.update(block)
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(DIGEST_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def folder_size(folder: Path) -> int:
    return sum(path.stat().st_size for path in uploaded_files(folder))


def upload_chunks(
    api: Any,
    repo_id: str,
    staging: Path,
    chunks: list[Path],
    state: UploadState,
    *,
    dry_run: bool = False,
    on_event: Callable[[str], None] = print,
) -> dict[str, int]:
    if not dry_run:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)

    uploaded = skipped = 0
    for chunk in chunks:
        relative = chunk.relative_to(staging)
        key = f"dataset:{repo_id}:{relative}"
        digest = folder_digest(chunk)
        size_gb = folder_size(chunk) / 2**30

        if state.is_current(key, digest):
            skipped += 1
            on_event(f"skip   {relative}  ({size_gb:.2f} GB)")
            continue

        on_event(f"upload {relative}  ({size_gb:.2f} GB)")
        uploaded += 1
        if dry_run:
            continue

        api.upload_folder(
            folder_path=str(chunk),
            path_in_repo=str(relative),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add {relative}",
            ignore_patterns=list(IGNORE_PATTERNS),
            delete_patterns="**",
        )
        state.record(key, digest)

    return {"uploaded": uploaded, "skipped": skipped, "chunks": len(chunks)}


def upload_archive(
    api: Any,
    repo_id: str,
    staging: Path,
    archive: Path,
    state: UploadState,
    *,
    dry_run: bool = False,
    on_event: Callable[[str], None] = print,
) -> bool:
    relative = archive.relative_to(staging).as_posix()
    key = f"dataset:{repo_id}:{relative}"
    digest = file_digest(archive)
    size_mb = archive.stat().st_size / 2**20
    if state.is_current(key, digest):
        on_event(f"skip   {relative}  ({size_mb:.0f} MB)")
        return False
    on_event(f"upload {relative}  ({size_mb:.0f} MB)")
    if not dry_run:
        upload_file(api, repo_id, archive, relative)
        state.record(key, digest)
    return True


def upload_file(api: Any, repo_id: str, path: Path, path_in_repo: str) -> None:
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Update {path_in_repo}",
    )
