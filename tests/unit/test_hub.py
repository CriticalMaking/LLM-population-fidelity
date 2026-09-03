from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow.parquet", reason="requires the hub extra")

import pyarrow.parquet as pq

from hub import cards, datasets, push
from hub.convert import convert_mode, record_row

RECORD = {
    "mode": "fa",
    "prompt_id": "Germa200600835",
    "profile": "Germany§2006§46§Female§Middle§Working§Married",
    "seed": 7,
    "attempt_count": 2,
    "schema_version": 3,
    "prompt": {"sha256": "abc", "text": "Question: ...\nAnswer:"},
    "result": {"answer": "B. Quite happy", "mass": 0.5},
    "attempts": [{"index": 0, "accepted": False, "seed": 7, "duration_ms": 1.0}],
    "trace": {
        "run_id": "run-1",
        "backend": {"name": "llama_cpp", "version": "0.3.1"},
        "model": {"filename": "mixtral.gguf", "model_key": "fresh", "sha256": "def"},
        "sampling": {"temperature": 0.7},
    },
}


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "outputs" / "culture" / "terra" / "base" / "d_happy"
    (directory / "raw" / "fa").mkdir(parents=True)
    (directory / "raw" / "fa" / "0001.json").write_text(json.dumps(RECORD), encoding="utf-8")
    (directory / "summary_metrics.csv").write_text("overall_nemd\n0.1\n", encoding="utf-8")
    return directory


def test_record_row_splits_the_profile_into_demographic_columns() -> None:
    row = record_row(RECORD, "raw/fa/0001.json")
    assert row["country"] == "Germany"
    assert row["wave"] == "2006"
    assert row["marital"] == "Married"
    assert row["model_ref"] == "mixtral.gguf"
    assert json.loads(row["result_json"])["answer"] == "B. Quite happy"


def test_convert_mode_writes_parquet_then_reports_it_current(run_dir: Path, tmp_path: Path) -> None:
    destination = tmp_path / "staging" / "raw-fa.parquet"
    assert convert_mode(run_dir, "fa", destination) == "written (1 rows)"
    assert pq.ParquetFile(destination).metadata.num_rows == 1
    assert convert_mode(run_dir, "fa", destination) == "current"
    assert convert_mode(run_dir, "ntp", tmp_path / "staging" / "raw-ntp.parquet") == "absent"


def test_build_staging_mirrors_tables_and_converts_records(run_dir: Path, tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    staging = tmp_path / "staging"
    counts = datasets.build_staging(outputs, staging, on_event=lambda message: None)

    assert counts["runs"] == 1
    assert counts["parquet"] == 1
    staged = staging / "data" / "culture" / "terra" / "base" / "d_happy"
    assert (staged / "raw-fa.parquet").is_file()
    assert (staged / "summary_metrics.csv").is_file()


def test_build_staging_prunes_files_no_longer_in_outputs(run_dir: Path, tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    staging = tmp_path / "staging"
    datasets.build_staging(outputs, staging, on_event=lambda message: None)
    stale = staging / "data" / "culture" / "terra" / "base" / "d_happy" / "gone.csv"
    stale.write_text("x\n", encoding="utf-8")

    counts = datasets.build_staging(outputs, staging, on_event=lambda message: None)

    assert counts["pruned"] == 1
    assert not stale.exists()


def test_chunks_split_by_model_and_leave_shallow_files_to_upload_singly(
    run_dir: Path, tmp_path: Path
) -> None:
    outputs = tmp_path / "outputs"
    (outputs / "culture" / "pooled.csv").write_text("a\n", encoding="utf-8")
    staging = tmp_path / "staging"
    datasets.build_staging(outputs, staging, on_event=lambda message: None)
    cards.write_dataset_card(staging, {"runs": 1})

    chunks = [str(chunk.relative_to(staging)) for chunk in datasets.dataset_chunks(staging)]
    loose = [str(path.relative_to(staging)) for path in datasets.dataset_files(staging)]

    assert chunks == ["data/culture/terra"]
    assert loose == ["README.md", "data/culture/pooled.csv"]


def test_upload_state_skips_a_chunk_until_its_contents_change(tmp_path: Path) -> None:
    chunk = tmp_path / "data" / "culture" / "terra"
    chunk.mkdir(parents=True)
    (chunk / "a.csv").write_text("one\n", encoding="utf-8")
    state = push.UploadState(tmp_path / "upload_state.json")

    digest = push.folder_digest(chunk)
    assert not state.is_current("dataset:lab/repo:data/culture/terra", digest)
    state.record("dataset:lab/repo:data/culture/terra", digest)
    assert state.is_current("dataset:lab/repo:data/culture/terra", digest)

    (chunk / "a.csv").write_text("two\n", encoding="utf-8")
    assert not state.is_current("dataset:lab/repo:data/culture/terra", push.folder_digest(chunk))


def test_folder_digest_reads_contents_not_just_sizes(tmp_path: Path) -> None:
    chunk = tmp_path / "chunk"
    chunk.mkdir()
    (chunk / "regression_fit.csv").write_text("fa,0.512\n", encoding="utf-8")
    before = push.folder_digest(chunk)

    (chunk / "regression_fit.csv").write_text("fa,0.498\n", encoding="utf-8")

    assert push.folder_digest(chunk) != before


def test_folder_digest_ignores_what_the_upload_ignores(tmp_path: Path) -> None:
    chunk = tmp_path / "chunk"
    chunk.mkdir()
    (chunk / "kept.csv").write_text("a\n", encoding="utf-8")
    before = push.folder_digest(chunk)

    (chunk / "raw-fa.parquet.partial").write_text("half a file\n", encoding="utf-8")

    assert push.folder_digest(chunk) == before


def test_a_second_repo_does_not_inherit_the_first_repos_uploads(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    chunk = staging / "data" / "culture" / "terra"
    chunk.mkdir(parents=True)
    (chunk / "a.csv").write_text("one\n", encoding="utf-8")
    state = push.UploadState(tmp_path / "upload_state.json")
    chunks = [chunk]

    first = push.upload_chunks(None, "lab/repo", staging, chunks, state, dry_run=True)
    state.record(f"dataset:lab/repo:{chunk.relative_to(staging)}", push.folder_digest(chunk))
    second = push.upload_chunks(None, "lab/repo", staging, chunks, state, dry_run=True)
    third = push.upload_chunks(None, "fork/repo", staging, chunks, state, dry_run=True)

    assert first["uploaded"] == 1
    assert second["skipped"] == 1
    assert third["uploaded"] == 1


def test_the_served_models_real_identity_never_reaches_the_staging_tree(
    run_dir: Path, tmp_path: Path
) -> None:
    outputs = tmp_path / "outputs"
    manifest = run_dir / "inference_manifest.json"
    manifest.write_text(
        json.dumps({"model": "gpt-5.6-terra", "root": "/home/someone/outputs"}),
        encoding="utf-8",
    )
    smoke = outputs / "culture" / "served_smoke" / "d_happy"
    smoke.mkdir(parents=True)
    (smoke / "served_smoke.csv").write_text("model\ngpt-5.6-terra\n", encoding="utf-8")
    staging = tmp_path / "staging"

    datasets.build_staging(outputs, staging, on_event=lambda message: None)

    staged = [path for path in staging.rglob("*") if path.is_file()]
    assert staged, "expected the run's own tables to be staged"
    for path in staged:
        if path.suffix == ".parquet":
            continue
        assert "gpt-5.6" not in path.read_text(encoding="utf-8")


def test_a_manifest_already_staged_is_pruned_on_the_next_build(
    run_dir: Path, tmp_path: Path
) -> None:
    outputs = tmp_path / "outputs"
    staging = tmp_path / "staging"
    datasets.build_staging(outputs, staging, on_event=lambda message: None)
    staged = staging / "data" / "culture" / "terra" / "base" / "d_happy"
    (run_dir / "inference_manifest.json").write_text("{}", encoding="utf-8")
    (staged / "inference_manifest.json").write_text("{}", encoding="utf-8")

    counts = datasets.build_staging(outputs, staging, on_event=lambda message: None)

    assert counts["pruned"] == 1
    assert not (staged / "inference_manifest.json").exists()


def test_a_pulled_snapshots_bookkeeping_is_never_offered_for_upload(
    run_dir: Path, tmp_path: Path
) -> None:
    outputs = tmp_path / "outputs"
    staging = tmp_path / "staging"
    counts = datasets.build_staging(outputs, staging, on_event=lambda message: None)
    cards.write_dataset_card(staging, counts)
    cache = staging / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (cache / "raw-fa.parquet.metadata").write_text("etag\n", encoding="utf-8")

    loose = [str(path.relative_to(staging)) for path in datasets.dataset_files(staging)]

    assert loose == ["README.md"]


def test_dataset_card_names_the_lab_repository_and_the_served_arm() -> None:
    card = cards.dataset_card({"runs": 61, "parquet": 61, "tables": 400})

    assert cards.DATASET_REPO_ID == "MInDS-lab-UTFPR/TowardsSociallyGroundedAISafety"
    assert cards.DATASET_URL in card
    assert "`culture/terra`" in card
    assert "the API exposes no" in card
    assert "| run directories | 61 |" in card
