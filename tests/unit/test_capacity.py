from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from culture import capacity
from machine_bias_reproduction import config
from machine_bias_reproduction.config import RunPaths


def _write(paths: RunPaths, mode: str, name: str, payload: dict[str, object]) -> None:
    directory = paths.raw / mode
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunPaths:
    monkeypatch.setattr(config, "OUTPUTS_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(config, "FIGURES_ROOT", tmp_path / "figures")
    run = RunPaths("culture/gemma4_31b/english", "d_happy")
    run.ensure()
    return run


def test_ntp_below_the_threshold_is_invalid_not_valid(paths: RunPaths) -> None:
    assert capacity.classify({"result": {"mass": 0.9999}}, "ntp") == "valid"
    assert capacity.classify({"result": {"mass": 0.0004}}, "ntp") == "invalid"
    assert capacity.classify({"result": {"mass": 0.0}}, "ntp") == "invalid"


def test_a_record_with_no_result_is_failed(paths: RunPaths) -> None:
    assert capacity.classify({"error": "RuntimeError: boom"}, "fa") == "failed"
    assert capacity.classify({"result": None}, "fa") == "failed"
    assert capacity.classify({"result": {"answer": "A. Very happy"}}, "fa") == "valid"


def test_measure_counts_each_mode_and_keeps_rejected_text(paths: RunPaths) -> None:
    _write(paths, "ntp", "a", {"result": {"mass": 0.99}})
    _write(paths, "ntp", "b", {"result": {"mass": 0.0003}})
    _write(
        paths,
        "fa",
        "c",
        {
            "result": {"answer": "A. Very happy"},
            "attempts": [{"raw_text": "A. Very happy", "accepted": True}],
        },
    )
    _write(
        paths,
        "fa",
        "d",
        {"error": "no valid FA answer", "attempts": [{"raw_text": "C.C.C.C.", "accepted": False}]},
    )

    frame = capacity.write(paths)
    ntp = frame[frame["mode"] == "NTP"].iloc[0]
    fa = frame[frame["mode"] == "FA"].iloc[0]
    assert (int(ntp["valid"]), int(ntp["invalid"])) == (1, 1)
    assert ntp["valid_rate"] == pytest.approx(0.5)
    assert (int(fa["valid"]), int(fa["failed"])) == (1, 1)

    examples = [
        json.loads(line)
        for line in (paths.outputs / capacity.EXAMPLES_FILE)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(entry["raw_text"] == "C.C.C.C." for entry in examples)
    assert capacity.read(paths) is not None


def test_a_run_that_answered_nothing_still_reports(paths: RunPaths) -> None:
    for index in range(5):
        _write(paths, "ntp", f"n{index}", {"result": {"mass": 1e-9}})
    frame = capacity.write(paths)
    row = frame[frame["mode"] == "NTP"].iloc[0]
    assert int(row["valid"]) == 0
    assert row["valid_rate"] == 0.0
    assert int(row["prompts"]) == 5


def test_a_model_repeating_one_answer_is_visible_as_one_distinct_distribution(
    paths: RunPaths,
) -> None:
    for index in range(20):
        _write(
            paths,
            "ntp",
            f"n{index}",
            {"result": {"mass": 1e-30, "A": 0.1, "B": 0.4, "C": 0.25, "D": 0.25}},
        )

    row = capacity.write(paths)[lambda frame: frame["mode"] == "NTP"].iloc[0]

    assert int(row["distinct_distributions"]) == 1
    assert int(row["tied_answers"]) == 20
    assert row["response_std"] == pytest.approx(0.0)


def test_answers_that_move_with_the_profile_count_separately(paths: RunPaths) -> None:
    for index in range(20):
        shift = index / 1000.0
        _write(
            paths,
            "ntp",
            f"n{index}",
            {"result": {"mass": 0.99, "A": 0.05 + shift, "B": 0.40 - shift, "C": 0.30, "D": 0.25}},
        )

    row = capacity.write(paths)[lambda frame: frame["mode"] == "NTP"].iloc[0]

    assert int(row["distinct_distributions"]) == 20
    assert int(row["tied_answers"]) == 0
    assert row["response_std"] > 0.0


def test_degeneracy_is_not_reported_for_full_answer_generation(paths: RunPaths) -> None:
    _write(paths, "ntp", "a", {"result": {"mass": 0.99, "A": 0.7, "B": 0.3}})
    _write(paths, "fa", "b", {"result": {"answer": "A. Very happy"}})

    frame = capacity.write(paths)

    assert frame[frame["mode"] == "FA"].iloc[0]["distinct_distributions"] is None or pd.isna(
        frame[frame["mode"] == "FA"].iloc[0]["distinct_distributions"]
    )
    assert int(frame[frame["mode"] == "NTP"].iloc[0]["distinct_distributions"]) == 1
