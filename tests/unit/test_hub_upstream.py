from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Any

import pytest

from hub import cards, datasets, fetch, push, upstream

SUBSET_FILES = (
    "code/1-create-prompts.R",
    "data/WVS/wvs-data.csv",
    "data/Linear/Linear-baseline-d_happy.csv",
    "data/subpops.csv",
    "data/LLM-outputs/csv/NTP-Mixtral-8x7B-d_happy.csv",
)

OUTSIDE_FILES = (
    "data/prompts/FA/d_happy/Germa200600835.txt",
    "data/LLM-outputs/raw/answer.json",
    "data/Robustness/table.csv",
    "data/WVS/.DS_Store",
    "data/WVS/._wvs-data.csv",
)


class FakeApi:
    def __init__(self, present: bool = True) -> None:
        self.uploaded: list[str] = []
        self.present = present

    def upload_file(self, **kwargs: Any) -> None:
        self.uploaded.append(str(kwargs["path_in_repo"]))
        self.present = True

    def file_exists(self, repo_id: str, filename: str, **kwargs: Any) -> bool:
        return self.present


class RefusingApi(FakeApi):
    def file_exists(self, repo_id: str, filename: str, **kwargs: Any) -> bool:
        raise ConnectionError("hub unreachable")


def _quiet(_: str) -> None:
    return None


def _package(root: Path, names: tuple[str, ...]) -> None:
    for name in names:
        path = root / upstream.PACKAGE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")


def test_the_archive_holds_the_redraw_subset_and_nothing_else(tmp_path: Path) -> None:
    _package(tmp_path, SUBSET_FILES + OUTSIDE_FILES)
    archive = upstream.build_upstream_archive(tmp_path / "staging", tmp_path)
    with tarfile.open(archive) as handle:
        members = sorted(handle.getnames())
    assert members == sorted((upstream.PACKAGE / name).as_posix() for name in SUBSET_FILES)
    assert archive == tmp_path / "staging" / "upstream" / "redraw-subset.tar.gz"


def test_rebuilding_an_unchanged_subset_is_byte_identical(tmp_path: Path) -> None:
    _package(tmp_path, SUBSET_FILES)
    first = upstream.build_upstream_archive(tmp_path / "staging", tmp_path).read_bytes()
    second = upstream.build_upstream_archive(tmp_path / "staging", tmp_path).read_bytes()
    assert first == second


def test_an_incomplete_package_is_refused_by_name(tmp_path: Path) -> None:
    _package(tmp_path, tuple(name for name in SUBSET_FILES if not name.startswith("data/WVS")))
    with pytest.raises(FileNotFoundError, match="data/WVS"):
        upstream.build_upstream_archive(tmp_path / "staging", tmp_path)


def test_the_archive_extracts_under_the_upstream_tree(tmp_path: Path) -> None:
    _package(tmp_path, SUBSET_FILES)
    archive = upstream.build_upstream_archive(tmp_path / "staging", tmp_path)
    clone = tmp_path / "clone"
    clone.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(clone, filter="data")
    extracted = clone / upstream.PACKAGE / "data" / "WVS" / "wvs-data.csv"
    assert extracted.read_text(encoding="utf-8") == "data/WVS/wvs-data.csv"


def test_the_pull_group_selects_only_the_archive_and_is_pulled_by_default() -> None:
    assert fetch.allow_patterns(["upstream"]) == ["upstream/*.tar.gz"]
    assert "upstream/*.tar.gz" in fetch.allow_patterns(None)


def test_the_staged_archive_is_left_to_its_own_upload(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    (staging / "upstream").mkdir(parents=True)
    (staging / "upstream" / "redraw-subset.tar.gz").write_bytes(b"archive")
    (staging / "README.md").write_text("card", encoding="utf-8")
    assert datasets.dataset_files(staging) == [staging / "README.md"]
    assert upstream.staged_archive_bytes(staging) == len(b"archive")
    assert upstream.staged_archive_bytes(tmp_path / "elsewhere") == 0


def test_the_archive_uploads_once_until_its_bytes_change(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    archive = upstream.staged_archive(staging)
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"first")
    state = push.UploadState(tmp_path / "upload_state.json")
    api = FakeApi()
    events: list[str] = []

    assert push.upload_archive(api, "lab/repo", staging, archive, state, on_event=events.append)
    assert not push.upload_archive(api, "lab/repo", staging, archive, state, on_event=events.append)
    archive.write_bytes(b"second")
    assert push.upload_archive(api, "lab/repo", staging, archive, state, on_event=events.append)

    assert api.uploaded == ["upstream/redraw-subset.tar.gz"] * 2
    assert [event.split()[0] for event in events] == ["upload", "skip", "upload"]


def test_a_dry_run_reports_the_archive_without_uploading_or_recording(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    archive = upstream.staged_archive(staging)
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"first")
    state = push.UploadState(tmp_path / "upload_state.json")
    api = FakeApi()

    assert push.upload_archive(api, "lab/repo", staging, archive, state, dry_run=True)
    assert api.uploaded == []
    assert state.entries == {}


def test_the_card_describes_the_archive_only_when_it_is_staged() -> None:
    with_archive = cards.dataset_card({"runs": 1, "upstream_bytes": 53 * 2**20})
    assert "upstream/redraw-subset.tar.gz` (53 MB)" in with_archive
    assert "hub pull --groups upstream" in with_archive
    assert "replication package terms" in with_archive
    assert "upstream/redraw-subset.tar.gz" not in cards.dataset_card({"runs": 1})


def test_an_archive_deleted_from_the_remote_uploads_again(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    archive = upstream.staged_archive(staging)
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"first")
    state = push.UploadState(tmp_path / "upload_state.json")
    api = FakeApi()
    assert push.upload_archive(api, "lab/repo", staging, archive, state, on_event=_quiet)

    api.present = False
    assert push.upload_archive(api, "lab/repo", staging, archive, state, on_event=_quiet)
    assert api.uploaded == ["upstream/redraw-subset.tar.gz"] * 2


def test_an_unreachable_hub_leaves_an_unchanged_archive_alone(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    archive = upstream.staged_archive(staging)
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"first")
    state = push.UploadState(tmp_path / "upload_state.json")
    api = RefusingApi()
    assert push.upload_archive(api, "lab/repo", staging, archive, state, on_event=_quiet)
    assert not push.upload_archive(api, "lab/repo", staging, archive, state, on_event=_quiet)
    assert api.uploaded == ["upstream/redraw-subset.tar.gz"]
