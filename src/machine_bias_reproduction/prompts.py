"""The paper's prompts, rebuilt byte-for-byte."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .questions import PromptMode as PromptMode
from .questions import Question, resolve_question

QUESTION_TEXT = {
    "i_surveyyear": "What year is it?",
    "i_country": (
        "What country are we in? Is it 'A. Australia' or 'B. Germany' or "
        "'C. Mexico' or 'D. Russia' or 'E. United States'?"
    ),
    "i_age": "How old are you?",
    "i_sex": "What is your gender? Is it 'A. Male' or 'B. Female'?",
    "i_education": (
        "What is the highest educational level that you have attained? Is it "
        "'A. Low' or 'B. Middle' or 'C. High'?"
    ),
    "i_employment": (
        "Are you employed now or not? Are you 'A. Working' or 'B. Retired' or "
        "'C. Homemaker' or 'D. Student' or 'E. Unemployed'?"
    ),
    "i_marstat": (
        "What is your marital status? Are you 'A. Married' or 'B. Cohabiting' or "
        "'C. Divorced or separated' or 'D. Widowed' or 'E. Single'?"
    ),
}

CATEGORY_ANSWERS = {
    "i_country": {
        "Australia": "A. Australia",
        "Germany": "B. Germany",
        "Mexico": "C. Mexico",
        "Russia": "D. Russia",
        "United States": "E. United States",
    },
    "i_sex": {"Male": "A. Male", "Female": "B. Female"},
    "i_education": {"Low": "A. Low", "Middle": "B. Middle", "High": "C. High"},
    "i_employment": {
        "Working": "A. Working",
        "Retired": "B. Retired",
        "Homemaker": "C. Homemaker",
        "Student": "D. Student",
        "Unemployed": "E. Unemployed",
    },
    "i_marstat": {
        "Married": "A. Married",
        "Cohabiting": "B. Cohabiting",
        "Divorced or separated": "C. Divorced or separated",
        "Widowed": "D. Widowed",
        "Single": "E. Single",
    },
}

CONTEXT_ORDER = (
    "i_surveyyear",
    "i_country",
    "i_age",
    "i_sex",
    "i_education",
    "i_employment",
    "i_marstat",
)


@dataclass(frozen=True, slots=True)
class PromptRecord:
    """One inference prompt and its source identity."""

    prompt_id: str
    profile: str
    mode: PromptMode
    text: str


def _answer(variable: str, value: Any) -> str | None:
    if pd.isna(value):
        return None
    if variable in {"i_surveyyear", "i_age"}:
        numeric = float(value)
        return str(int(numeric)) if numeric.is_integer() else str(value)
    return CATEGORY_ANSWERS[variable][str(value)]


def build_prompt(row: pd.Series, question: str | Question, mode: PromptMode) -> str:
    """Build the upstream prompt exactly, including its final newline.

    A question whose answer is missing in the WVS drops its whole Q/A pair, and
    a numerical outcome closes with ``Answer: `` rather than ``Answer:``.
    """
    outcome = resolve_question(question)
    lines: list[str] = []
    for variable in CONTEXT_ORDER:
        answer = _answer(variable, row[variable])
        if answer is None:
            continue
        lines.extend((f"Question: {QUESTION_TEXT[variable]}", f"Answer: {answer}"))
    lines.append(f"Question: {outcome.question_text(mode)}")
    return "\n".join(lines) + f"\nAnswer:{outcome.answer_suffix}\n"


def _selection(wvs: pd.DataFrame, mode: PromptMode) -> tuple[pd.DataFrame, str]:
    """Return the rows and identity column one generation mode scores."""
    if mode == "ntp":
        return wvs.drop_duplicates("profile", keep="first"), "profile"
    return wvs, "id"


def prompt_records(
    wvs: pd.DataFrame,
    mode: PromptMode,
    question: str | Question,
) -> list[PromptRecord]:
    """Create the NTP unique-profile or FA observation-level prompt collection."""
    outcome = resolve_question(question)
    rows, id_column = _selection(wvs, mode)
    return [
        PromptRecord(
            prompt_id=str(row[id_column]),
            profile=str(row["profile"]),
            mode=mode,
            text=build_prompt(row, outcome, mode),
        )
        for _, row in rows.iterrows()
    ]
