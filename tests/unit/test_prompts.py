from __future__ import annotations

import pytest

from machine_bias_reproduction.config import UPSTREAM_DATA
from machine_bias_reproduction.data import load_wvs
from machine_bias_reproduction.prompts import build_prompt, prompt_records
from machine_bias_reproduction.questions import (
    QUESTION_NAMES,
    PromptMode,
    resolve_question,
)

MODES = (("ntp", "NTP"), ("fa", "FA"))
SAMPLE = 25


@pytest.mark.parametrize("question", QUESTION_NAMES)
@pytest.mark.parametrize(("mode", "directory"), MODES)
def test_prompts_match_upstream_byte_for_byte(
    question: str, mode: PromptMode, directory: str
) -> None:
    wvs = load_wvs()
    records = prompt_records(wvs.head(SAMPLE * 4), mode, question)[:SAMPLE]
    root = UPSTREAM_DATA / "prompts" / directory / question
    assert records
    for record in records:
        upstream = root / f"{record.prompt_id}.txt"
        assert record.text.encode() == upstream.read_bytes(), (
            f"{question}/{mode} prompt {record.prompt_id} differs from upstream"
        )


def test_numerical_question_closes_with_a_trailing_space() -> None:
    wvs = load_wvs()
    row = wvs.iloc[0]
    assert build_prompt(row, "d_polpos", "fa").endswith("Answer: \n")
    assert build_prompt(row, "d_happy", "fa").endswith("Answer:\n")


def test_politics_is_rescaled_for_ntp_only() -> None:
    wvs = load_wvs()
    row = wvs.iloc[0]
    assert "scale of 0 to 9" in build_prompt(row, "d_polpos", "ntp")
    assert "scale of 1 to 10" in build_prompt(row, "d_polpos", "fa")


def test_missing_context_answer_omits_the_entire_question_pair() -> None:
    wvs = load_wvs()
    row = wvs.iloc[0].copy()
    row["i_employment"] = None
    prompt = build_prompt(row, "d_happy", "fa")
    assert "Are you employed now or not?" not in prompt
    assert "Taking all things together" in prompt


@pytest.mark.parametrize("question", QUESTION_NAMES)
def test_complete_prompt_counts(question: str) -> None:
    wvs = load_wvs()
    assert len(prompt_records(wvs, "ntp", question)) == 13_904
    assert len(prompt_records(wvs, "fa", question)) == 26_981


def test_question_registry_matches_the_upstream_metadata() -> None:
    import pandas as pd

    metadata = pd.read_csv(UPSTREAM_DATA / "WVS" / "wvs-questions.csv").set_index("var")
    levels = pd.read_csv(UPSTREAM_DATA / "WVS" / "wvs-levels.csv")
    for name in QUESTION_NAMES:
        question = resolve_question(name)
        assert question.full_q == metadata.loc[name, "full_q"]
        assert question.numerical == (metadata.loc[name, "type"] == "numerical")
        if question.numerical:
            continue
        block = levels[levels["variable"] == name]
        assert tuple(block["label"]) == question.wvs_labels
        assert tuple(block["letter_answer"]) == question.fa_answers
