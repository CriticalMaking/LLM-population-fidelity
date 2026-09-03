from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .config import (
    ARCHIVE_PATH,
    ARCHIVE_SHA256,
    PROJECT_ROOT,
    UPSTREAM_CODE,
    UPSTREAM_DATA,
    UPSTREAM_DIR,
    UPSTREAM_ROOT,
)
from .io_utils import atomic_write_json, hash_paths, sha256_file

MANIFEST_PATH = UPSTREAM_DIR / "MANIFEST.json"

CONSUMED_PATHS = (
    UPSTREAM_DATA / "WVS" / "wvs-data.csv",
    UPSTREAM_DATA / "WVS" / "wvs-levels.csv",
    UPSTREAM_DATA / "WVS" / "wvs-questions.csv",
    UPSTREAM_DATA / "subpops.csv",
    UPSTREAM_DATA / "Linear" / "Linear-baseline-d_happy.csv",
    UPSTREAM_DATA / "LLM-outputs" / "csv" / "NTP-Mixtral-8x7B-d_happy.csv",
    UPSTREAM_DATA / "LLM-outputs" / "csv" / "FA-Mixtral-8x7B.csv",
    UPSTREAM_CODE / "1-create-prompts.R",
    UPSTREAM_CODE / "2a-generate-NTP-Mixtral-8x7B.py",
    UPSTREAM_CODE / "2b-generate-FA-Mixtral-8x7B.py",
    UPSTREAM_CODE / "4-create-subpopulations.R",
    UPSTREAM_CODE / "6-results.R",
    UPSTREAM_CODE / "EMD.cpp",
)


def is_substantive(name: str, *, is_directory: bool = False) -> bool:
    if is_directory:
        return False
    path = PurePosixPath(name)
    if not path.parts or path.parts[0] == "__MACOSX":
        return False
    return path.name != ".DS_Store" and not path.name.startswith("._")


def extraction_name(name: str) -> str:
    try:
        return name.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def archive_inventory() -> dict[str, int]:
    total_entries = 0
    total_files = 0
    total_bytes = 0
    payload_files = 0
    payload_bytes = 0
    metadata_files = 0
    metadata_bytes = 0
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        for info in archive.infolist():
            total_entries += 1
            if info.is_dir():
                continue
            total_files += 1
            total_bytes += info.file_size
            if is_substantive(info.filename):
                payload_files += 1
                payload_bytes += info.file_size
            else:
                metadata_files += 1
                metadata_bytes += info.file_size
    return {
        "total_entries": total_entries,
        "total_files": total_files,
        "total_uncompressed_bytes": total_bytes,
        "substantive_files": payload_files,
        "substantive_uncompressed_bytes": payload_bytes,
        "metadata_files": metadata_files,
        "metadata_uncompressed_bytes": metadata_bytes,
    }


def create_manifest() -> dict[str, Any]:
    missing = [path for path in CONSUMED_PATHS if not path.is_file()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"required upstream files are missing:\n{joined}")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "canonical_archive": {
            "path": str(ARCHIVE_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(ARCHIVE_PATH),
            "expected_sha256": ARCHIVE_SHA256,
            "size_bytes": ARCHIVE_PATH.stat().st_size,
        },
        "working_extraction": {
            "path": str(UPSTREAM_ROOT.relative_to(PROJECT_ROOT)),
            "excluded_patterns": ["__MACOSX/**", "**/._*", "**/.DS_Store"],
            **archive_inventory(),
        },
        "consumed_files": hash_paths(CONSUMED_PATHS, PROJECT_ROOT),
    }
    atomic_write_json(MANIFEST_PATH, manifest)
    return manifest


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open(encoding="utf-8") as stream:
        loaded: dict[str, Any] = json.load(stream)
    return loaded


def verify_upstream(*, full: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    archive_hash = sha256_file(ARCHIVE_PATH)
    if archive_hash != ARCHIVE_SHA256:
        errors.append(f"archive SHA-256 mismatch: {archive_hash}")

    manifest = load_manifest() if MANIFEST_PATH.exists() else create_manifest()
    consumed = manifest.get("consumed_files", {})
    for relative, expected_hash in consumed.items():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            errors.append(f"missing consumed file: {relative}")
        elif sha256_file(path) != expected_hash:
            errors.append(f"modified consumed file: {relative}")

    checked_files = len(consumed)
    checked_bytes = sum((PROJECT_ROOT / relative).stat().st_size for relative in consumed)
    if full:
        checked_files = 0
        checked_bytes = 0
        with zipfile.ZipFile(ARCHIVE_PATH) as archive:
            bad_crc = archive.testzip()
            if bad_crc is not None:
                errors.append(f"ZIP CRC failure: {bad_crc}")
            for info in archive.infolist():
                if not is_substantive(info.filename, is_directory=info.is_dir()):
                    continue
                member_name = extraction_name(info.filename)
                relative = PurePosixPath(member_name)
                extracted = UPSTREAM_DIR / "extracted" / Path(*relative.parts)
                checked_files += 1
                checked_bytes += info.file_size
                if not extracted.is_file():
                    errors.append(f"missing extracted file: {member_name}")
                elif extracted.stat().st_size != info.file_size:
                    errors.append(f"size mismatch: {member_name}")
                if len(errors) >= 100:
                    errors.append("stopped after 100 extraction errors")
                    break

    result = {
        "ok": not errors,
        "full": full,
        "archive_sha256": archive_hash,
        "checked_files": checked_files,
        "checked_bytes": checked_bytes,
        "errors": errors,
    }
    if errors:
        raise RuntimeError("\n".join(errors))
    return result
