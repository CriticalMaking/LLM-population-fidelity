"""Safe filesystem and hashing helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Return the lowercase SHA-256 digest for a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_seed(identifier: str, global_seed: int) -> int:
    """Derive a stable nonzero llama.cpp seed from a prompt identifier."""
    payload = f"{global_seed}\0{identifier}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF
    return value or 1


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace a UTF-8 text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a consistently formatted JSON object."""
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    """Append one durable record to an append-only JSON Lines log.

    Single ``write`` calls of one complete line keep concurrent appends from
    interleaving, so a crashed or resumed run never truncates earlier events.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8", newline="") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def hash_paths(paths: Iterable[Path], root: Path) -> dict[str, str]:
    """Hash paths and key them relative to a common root."""
    return {str(path.relative_to(root)): sha256_file(path) for path in paths}


def hardware_summary() -> dict[str, Any]:
    """Return stable, non-sensitive runtime hardware metadata."""
    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


def code_revision(root: Path) -> str | None:
    """Return the short commit of a repository, suffixed when the tree is dirty."""

    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )

    head = git("rev-parse", "--short", "HEAD")
    if head.returncode != 0:
        return None
    revision = head.stdout.strip()
    if not revision:
        return None
    status = git("status", "--porcelain")
    return f"{revision}-dirty" if status.stdout.strip() else revision
