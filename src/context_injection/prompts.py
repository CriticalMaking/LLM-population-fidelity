from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from machine_bias_reproduction.prompts import (
    PromptMode,
    PromptRecord,
    build_prompt as baseline_prompt,
)
from machine_bias_reproduction.questions import Question, resolve_question

from .base import ContextPayload, ContextSpec
from .strategies import build_context
from .theories.inglehart_welzel import DEFAULT_MAPPING, InglehartWelzelMapping


@dataclass(frozen=True, slots=True)
class ContextPromptRecord:
    prompt_id: str
    profile: str
    mode: PromptMode
    text: str
    context: ContextPayload

    def plain(self) -> PromptRecord:
        return PromptRecord(self.prompt_id, self.profile, self.mode, self.text)


def build_prompt(
    row: pd.Series,
    question: str | Question,
    mode: PromptMode,
    spec: ContextSpec | None = None,
    *,
    mapping: InglehartWelzelMapping = DEFAULT_MAPPING,
) -> tuple[str, ContextPayload]:
    question = resolve_question(question)
    base = baseline_prompt(row, question, mode)
    context = build_context(row, question, spec, mapping=mapping)
    if not context.context_text:
        return base, context

    marker = f"Question: {question.question_text(mode)}\nAnswer:"
    before, found, after = base.partition(marker)
    if not found:
        raise ValueError(f"target question marker not found for {question.var}")
    return f"{before.rstrip()}\n\n{context.context_text}\n\n{found}{after}", context


def prompt_records(
    wvs: pd.DataFrame,
    mode: PromptMode,
    question: str | Question,
    spec: ContextSpec | None = None,
    *,
    mapping: InglehartWelzelMapping = DEFAULT_MAPPING,
) -> list[ContextPromptRecord]:
    question = resolve_question(question)
    rows, id_column = (wvs.drop_duplicates("profile", keep="first"), "profile")
    if mode == "fa":
        rows, id_column = wvs, "id"

    records: list[ContextPromptRecord] = []
    for _, row in rows.iterrows():
        text, context = build_prompt(row, question, mode, spec, mapping=mapping)
        records.append(
            ContextPromptRecord(
                prompt_id=str(row[id_column]),
                profile=str(row["profile"]),
                mode=mode,
                text=text,
                context=context,
            )
        )
    return records
