from __future__ import annotations

from machine_bias_reproduction.provenance import extraction_name, is_substantive


def test_mojibake_zip_name_decodes_to_infozip_extraction_name() -> None:
    raw = "folder/en┬ºUnited_States.txt"
    assert extraction_name(raw) == "folder/en§United_States.txt"


def test_metadata_entries_are_not_substantive() -> None:
    assert not is_substantive("__MACOSX/package/._data.csv")
    assert not is_substantive("package/data/._answers.csv")
    assert not is_substantive("package/.DS_Store")
    assert is_substantive("package/data/answers.csv")
