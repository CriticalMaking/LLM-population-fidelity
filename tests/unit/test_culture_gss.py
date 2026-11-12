from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

import culture.backend
from culture import gss
from culture.registry import CULTURE_MODELS
from machine_bias_reproduction import cli
from machine_bias_reproduction.cli import build_parser
from machine_bias_reproduction.io_utils import sha256_file

MODEL = CULTURE_MODELS["qwen3_vl_2b"]

STEERING_ENTRIES: list[dict[str, str]] = [
    {
        "attribute": "SEX",
        "qa_prompt": "What is the sex that you were assigned at birth?",
        "options": "['Male', 'Female']",
    },
    {
        "attribute": "RELIG",
        "qa_prompt": "What is your present religion, if any?",
        "options": str([f"Faith {index}" for index in range(11)] + ["Nothing in particular"]),
    },
]

SUBGROUPS: tuple[tuple[str, str], ...] = (
    ("SEX", "Male"),
    ("SEX", "Female"),
    ("RELIG", "Nothing in particular"),
)

QUESTIONS: dict[str, dict[str, Any]] = {
    "happy": {
        "question": "Would you say that you are very happy, pretty happy, or not too happy?",
        "options": ["Very happy", "Pretty happy", "Not too happy", "Refused"],
        "ordinal": [1.0, 2.0, 3.0],
        "responses": ([0.3, 0.5, 0.2], [0.2, 0.6, 0.2], [0.4, 0.4, 0.2]),
    },
    "trust": {
        "question": "Can most people be trusted?",
        "options": ["Can trust", "Can't be too careful", "Depends", "Refused"],
        "ordinal": [1.0, 2.0, 1.5],
        "responses": ([0.3, 0.6, 0.1], [0.2, 0.7, 0.1], [0.5, 0.4, 0.1]),
    },
}


def _raw(qkey: str = "trust", subgroup: int = 1, **changes: Any) -> dict[str, Any]:
    question = QUESTIONS[qkey]
    attribute, group = SUBGROUPS[subgroup]
    return {
        "qkey": qkey,
        "attribute": attribute,
        "group": group,
        "question": question["question"],
        "options": list(question["options"]),
        "responses": list(question["responses"][subgroup]),
        "refusal_rate": 0.01,
        "ordinal": list(question["ordinal"]),
        **changes,
    }


def _raw_table() -> list[dict[str, Any]]:
    return [_raw(qkey, subgroup) for qkey in QUESTIONS for subgroup in range(len(SUBGROUPS))]


def _write(root: Path, rows: Sequence[dict[str, Any]] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    table = _raw_table() if rows is None else rows
    (root / gss.EVAL_FILE).write_text(
        "".join(json.dumps(row) + "\n" for row in table), encoding="utf-8"
    )
    (root / gss.STEERING_FILE).write_text(json.dumps(STEERING_ENTRIES), encoding="utf-8")
    return root


def _pin(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(
        gss, "GSS_SHA256", {name: sha256_file(root / name) for name in gss.GSS_SHA256}
    )
    monkeypatch.setattr(gss, "EXPECTED_RECORDS", len(_raw_table()))
    monkeypatch.setattr(gss, "EXPECTED_QUESTIONS", len(QUESTIONS))
    monkeypatch.setattr(gss, "EXPECTED_SUBGROUPS", len(SUBGROUPS))


class FakeBackend:
    def __init__(self, batch_size: int = 2) -> None:
        self._batch_size = batch_size
        self.batches: list[list[str]] = []
        self.cultures: list[str] = []
        self.stubs: list[tuple[str, tuple[str, ...]]] = []

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def set_culture(self, culture: str) -> None:
        self.cultures.append(culture)

    def describe(self) -> dict[str, Any]:
        return {"name": "fake", "question": "d_happy", "prompt_contract": "paper"}

    def continuation_token_ids(self, stub: str, continuations: Sequence[str]) -> list[int]:
        self.stubs.append((stub, tuple(continuations)))
        return list(range(100, 100 + len(continuations)))

    def first_token_batch(
        self, prompts: Sequence[str], token_ids: Sequence[int]
    ) -> list[list[float]]:
        self.batches.append(list(prompts))
        return [
            [len(prompt) / 10_000.0 + position / 100.0 for position in range(len(token_ids))]
            for prompt in prompts
        ]


def test_a_record_keeps_its_shares_apart_from_the_refusal_option() -> None:
    record = gss.parse_record(_raw())

    assert record.options[-1] == "Refused"
    assert len(record.responses) == len(record.options) - 1
    assert record.ordinal == (1.0, 2.0, 1.5)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"options": ["Can trust", "Can't be too careful", "Depends"]}, "refusal"),
        ({"ordinal": [1.0, 2.0]}, "ordinal levels"),
        ({"responses": [0.5, 0.6, -0.1]}, "negative"),
        ({"responses": [0.3, 0.3, 0.3]}, "sum to"),
        ({"refusal_rate": 1.5}, "refusal rate"),
        (
            {"options": [str(index) for index in range(11)], "responses": [0.1] * 10},
            "more than the 10 letters",
        ),
    ],
)
def test_a_malformed_record_is_refused(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        gss.parse_record(_raw(**changes))


def test_options_are_ordered_by_their_level_and_those_without_one_are_dropped() -> None:
    assert gss.ordered_positions([1.0, 2.0, 1.5]) == (0, 2, 1)
    assert gss.ordered_positions([2.0, 3.0, 1.0]) == (2, 0, 1)
    assert gss.ordered_positions([1.0, 2.0, -1.0, -1.0]) == (0, 1)


def test_a_scale_needs_two_distinct_levels() -> None:
    with pytest.raises(ValueError, match="fewer than 2"):
        gss.ordered_positions([1.0, -1.0])
    with pytest.raises(ValueError, match="share an ordinal level"):
        gss.ordered_positions([1.0, 1.0, 2.0])


def test_shares_are_renormalised_over_the_ordered_options() -> None:
    shares = gss.ordered_shares([0.2, 0.2, 0.1, 0.5], [1.0, 2.0, -1.0])

    assert shares == pytest.approx([0.5, 0.5])
    assert gss.ordered_shares([0.3, 0.6, 0.1], [1.0, 2.0, 1.5]) == pytest.approx([0.3, 0.1, 0.6])


def test_shares_without_any_mass_are_blank() -> None:
    assert np.isnan(gss.ordered_shares([0.0, 0.0, 0.4], [1.0, 2.0, -1.0])).all()
    assert np.isnan(gss.ordered_shares([float("nan"), 0.2], [1.0, 2.0])).all()


def test_the_prompt_is_the_steering_answer_then_the_question_with_its_letters(
    tmp_path: Path,
) -> None:
    steering = gss.load_steering(_write(tmp_path))

    prompt = gss.build_prompt(gss.parse_record(_raw()), steering)

    assert prompt == (
        "Question: What is the sex that you were assigned at birth?\n"
        "A. Male\n"
        "B. Female\n"
        "\n"
        "Answer: B. Female\n"
        "\n"
        "Answer the following question keeping in mind your previous answers.\n"
        "Question: Can most people be trusted?\n"
        "A. Can trust\n"
        "B. Can't be too careful\n"
        "C. Depends\n"
        "D. Refused\n"
        "Answer as a choice between A.,B.,C.,D.\n"
        "\n"
        "Answer:"
    )


def test_a_steering_question_letters_its_options_past_the_tenth(tmp_path: Path) -> None:
    steering = gss.load_steering(_write(tmp_path))

    prompt = gss.build_prompt(gss.parse_record(_raw(subgroup=2)), steering)

    assert steering["RELIG"].options[-1] == "Nothing in particular"
    assert "L. Nothing in particular\n\nAnswer: L. Nothing in particular\n\n" in prompt


def test_a_subgroup_the_steering_prompts_do_not_offer_is_refused(tmp_path: Path) -> None:
    steering = gss.load_steering(_write(tmp_path))

    with pytest.raises(ValueError, match="no steering prompt"):
        gss.build_prompt(gss.parse_record(_raw(attribute="AGE")), steering)
    with pytest.raises(ValueError, match="not a steering option"):
        gss.build_prompt(gss.parse_record(_raw(group="Other")), steering)


def test_the_table_must_put_every_question_to_every_subgroup(tmp_path: Path) -> None:
    steering = gss.load_steering(_write(tmp_path))
    records = [gss.parse_record(row) for row in _raw_table()]

    gss.check_records(records, steering)
    with pytest.raises(ValueError, match="missing one of the subgroups"):
        gss.check_records(records[:-1], steering)
    with pytest.raises(ValueError, match="appears twice"):
        gss.check_records([*records, records[0]], steering)
    reworded = gss.parse_record(_raw(question="Another wording?"))
    with pytest.raises(ValueError, match="differs between subgroups"):
        gss.check_records([records[3], reworded, records[5]], steering)


def test_loading_before_staging_names_the_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="gss-stage"):
        gss.load_records(tmp_path)


def test_staging_copies_the_two_files_and_counts_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write(tmp_path / "source")
    _pin(monkeypatch, source)

    staged = gss.stage_gss(source, tmp_path / "staged")

    assert (tmp_path / "staged" / gss.EVAL_FILE).read_bytes() == (
        source / gss.EVAL_FILE
    ).read_bytes()
    assert staged["records"] == len(_raw_table())
    assert staged["questions"] == len(QUESTIONS)
    assert staged["subgroups"] == len(SUBGROUPS)
    assert staged["origins"][gss.EVAL_FILE] == str(source / gss.EVAL_FILE)
    assert gss.stage_gss(tmp_path / "staged", tmp_path / "staged")["records"] == len(_raw_table())


def test_staging_refuses_a_file_that_is_not_the_pinned_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write(tmp_path / "source")
    _pin(monkeypatch, source)
    _write(tmp_path / "other", _raw_table()[:-3])

    with pytest.raises(RuntimeError, match="expected"):
        gss.stage_gss(tmp_path / "other", tmp_path / "staged")


def test_staging_fetches_what_the_source_does_not_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin = _write(tmp_path / "origin")
    _pin(monkeypatch, origin)
    fetched: list[str] = []

    def fetcher(name: str) -> gss.Fetcher:
        def fetch(target: Path) -> None:
            fetched.append(name)
            target.write_bytes((origin / name).read_bytes())

        return fetch

    fetchers = {name: fetcher(name) for name in gss.GSS_SHA256}

    staged = gss.stage_gss(tmp_path / "absent", tmp_path / "staged", fetchers=fetchers)
    again = gss.stage_gss(tmp_path / "absent", tmp_path / "staged", fetchers=fetchers)
    forced = gss.stage_gss(origin, tmp_path / "staged", download=True, fetchers=fetchers)

    assert staged["origins"][gss.EVAL_FILE] == gss.SUBPOP_DATASET
    assert again["origins"][gss.EVAL_FILE] == str(tmp_path / "staged" / gss.EVAL_FILE)
    assert forced["origins"][gss.STEERING_FILE] == gss.STEERING_URL
    assert fetched == [gss.EVAL_FILE, gss.STEERING_FILE] * 2


def test_staging_refuses_a_table_of_another_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write(tmp_path / "source")
    _pin(monkeypatch, source)
    monkeypatch.setattr(gss, "EXPECTED_QUESTIONS", len(QUESTIONS) + 1)

    with pytest.raises(RuntimeError, match="expected"):
        gss.stage_gss(source, tmp_path / "staged")


def test_prompts_are_scored_in_batches_by_length_and_returned_in_their_order() -> None:
    backend = FakeBackend(batch_size=2)
    prompts = ["x" * length for length in (30, 10, 50, 20, 40)]

    scored = gss.letter_probabilities(backend, prompts)

    assert [[len(prompt) for prompt in batch] for batch in backend.batches] == [
        [10, 20],
        [30, 40],
        [50],
    ]
    assert backend.stubs == [("Answer:", gss.OPTION_TOKENS)]
    assert gss.OPTION_TOKENS[:2] == (" A", " B")
    assert scored.shape == (5, 10)
    assert scored[:, 0] == pytest.approx([0.003, 0.001, 0.005, 0.002, 0.004])
    assert scored[2, 9] == pytest.approx(0.005 + 0.09)


def test_a_run_table_blanks_the_letters_a_question_does_not_offer() -> None:
    records = [gss.parse_record(_raw()), gss.parse_record(_raw("happy", 0))]
    probabilities = np.tile(np.arange(1, 11) / 100.0, (2, 1))

    frame = gss.run_frame(records, probabilities)

    assert list(frame.columns) == ["qkey", "attribute", "group", "options", "mass", *"ABCDEFGHIJ"]
    assert frame["options"].tolist() == [4, 4]
    assert frame["mass"].tolist() == pytest.approx([0.1, 0.1])
    assert frame[list("ABCD")].iloc[0].tolist() == pytest.approx([0.01, 0.02, 0.03, 0.04])
    assert frame[list("EFGHIJ")].isna().all().all()


def test_a_run_writes_its_table_and_a_manifest_that_names_the_prompt_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    steering = gss.load_steering(_write(tmp_path / "data"))
    records = gss.load_records(tmp_path / "data")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(gss, "staged_adapter", lambda key, arm: adapter)
    backend = FakeBackend(batch_size=4)

    result = gss.run_gss(MODEL, "subpop", backend, records, steering, root=tmp_path / "out")

    table = gss.run_table(MODEL.key, "subpop", tmp_path / "out")
    assert (
        table == tmp_path / "out" / MODEL.key / "subpop" / "gss" / "NTP-qwen3_vl_2b-subpop-gss.csv"
    )
    assert result["table"] == str(table)
    assert result["prompts"] == len(records)
    assert backend.cultures == ["subpop"]
    frame = pd.read_csv(table)
    assert frame[["qkey", "attribute", "group"]].to_records(index=False).tolist() == [
        (record.qkey, record.attribute, record.group) for record in records
    ]
    manifest = json.loads((table.parent / gss.RUN_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["backend"]["prompt_contract"] == "subpop-qa"
    assert manifest["backend"]["question"] == "gss"
    assert manifest["model"]["adapter_sha256"] == sha256_file(adapter / "adapter_model.safetensors")
    assert manifest["data"]["revision"] == gss.SUBPOP_REVISION
    assert manifest["preflight"]["prompts"] == len(records)


def test_the_released_model_carries_no_adapter_in_its_manifest(tmp_path: Path) -> None:
    steering = gss.load_steering(_write(tmp_path / "data"))
    records = gss.load_records(tmp_path / "data")

    gss.run_gss(MODEL, "base", FakeBackend(), records, steering, root=tmp_path / "out")

    directory = gss.run_directory(MODEL.key, "base", tmp_path / "out")
    manifest = json.loads((directory / gss.RUN_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["model"]["adapter_path"] is None
    assert manifest["model"]["adapter_sha256"] is None


def test_a_limited_run_reports_its_mass_and_writes_nothing(tmp_path: Path) -> None:
    steering = gss.load_steering(_write(tmp_path / "data"))
    records = gss.load_records(tmp_path / "data")

    result = gss.run_gss(
        MODEL, "base", FakeBackend(), records, steering, limit=2, root=tmp_path / "out"
    )

    assert result["prompts"] == 2
    assert result["table"] is None
    assert not (tmp_path / "out").exists()


def test_a_run_is_flagged_when_the_letters_hold_too_little_mass() -> None:
    low = gss.mass_summary(pd.DataFrame({"mass": [0.01, 0.05]}))
    high = gss.mass_summary(pd.DataFrame({"mass": [0.4, 0.9]}))

    assert low["informative"] is False
    assert high["informative"] is True
    assert high["mean_mass"] == pytest.approx(0.65)


def test_the_cli_takes_the_stage_and_the_run() -> None:
    parser = build_parser()

    stage = parser.parse_args(["culture-gss-stage", "--download"])
    run = parser.parse_args(["culture-gss", "--models", "qwen3_vl_2b", "--limit", "8"])

    assert stage.download is True
    assert stage.source == str(gss.DEFAULT_GSS_SOURCE)
    assert run.models == ["qwen3_vl_2b"]
    assert run.arms is None
    assert run.limit == 8
    assert gss.GSS_ARMS == ("base", "subpop")


class LoadedBackend(FakeBackend):
    loads: ClassVar[list[tuple[str, tuple[str, ...]]]] = []

    def __init__(
        self,
        model: Any,
        adapters: dict[str, Path],
        question: str,
        *,
        batch_size: int | None = None,
    ) -> None:
        super().__init__(batch_size or 4)
        LoadedBackend.loads.append((model.key, tuple(adapters)))


def _command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = _write(tmp_path / "data")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    records, steering = gss.load_records(data), gss.load_steering(data)
    monkeypatch.setattr(gss, "load_records", lambda: records)
    monkeypatch.setattr(gss, "load_steering", lambda: steering)
    monkeypatch.setattr(gss, "CULTURE_ROOT", tmp_path / "out")
    monkeypatch.setattr(gss, "staged_adapter", lambda key, arm: adapter)
    monkeypatch.setattr(cli, "staged_adapter", lambda key, arm: adapter)
    monkeypatch.setattr(culture.backend, "TransformersBackend", LoadedBackend)
    monkeypatch.setattr(LoadedBackend, "loads", [])
    return tmp_path / "out"


def test_the_command_runs_both_arms_once_and_skips_them_afterwards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _command(tmp_path, monkeypatch)
    arguments = build_parser().parse_args(["culture-gss", "--models", MODEL.key])

    cli.command_culture_gss(arguments)
    cli.command_culture_gss(arguments)

    assert LoadedBackend.loads == [(MODEL.key, ("subpop",)), (MODEL.key, ())]
    assert gss.run_table(MODEL.key, "base", out).is_file()
    assert gss.run_table(MODEL.key, "subpop", out).is_file()
    assert capsys.readouterr().out.count("already run, skipping") == 2


def test_the_command_reruns_an_arm_when_forced_and_writes_nothing_under_a_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _command(tmp_path, monkeypatch)
    parser = build_parser()

    cli.command_culture_gss(
        parser.parse_args(["culture-gss", "--models", MODEL.key, "--arms", "base", "--limit", "2"])
    )
    assert not out.exists()
    cli.command_culture_gss(
        parser.parse_args(["culture-gss", "--models", MODEL.key, "--arms", "base"])
    )
    cli.command_culture_gss(
        parser.parse_args(["culture-gss", "--models", MODEL.key, "--arms", "base", "--force"])
    )

    assert LoadedBackend.loads == [(MODEL.key, ())] * 3
    assert not gss.run_table(MODEL.key, "subpop", out).exists()


def test_the_command_refuses_a_model_without_next_token_probabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _command(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="no next-token probabilities"):
        cli.command_culture_gss(build_parser().parse_args(["culture-gss", "--models", "terra"]))
    assert LoadedBackend.loads == []
