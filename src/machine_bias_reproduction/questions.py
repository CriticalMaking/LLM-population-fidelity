"""The four WVS outcome questions, and how each one is encoded."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

PromptMode = Literal["ntp", "fa"]

LETTERS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")


@dataclass(frozen=True, slots=True)
class Question:
    """One outcome question and every encoding the reproduction needs."""

    var: str
    label: str
    numerical: bool
    answer_columns: tuple[str, ...]
    ntp_csv_columns: tuple[str, ...]
    """Answer columns as the archived NTP CSV names them: ``X0``-``X9`` for d_polpos."""
    wvs_labels: tuple[str, ...]
    fa_answers: tuple[str, ...]
    ntp_tokens: tuple[str, ...]
    """Single tokens NTP scores: space-prefixed letters, or bare digits for d_polpos."""
    full_q: str
    full_q_ntp: str
    """d_polpos is asked 0-9 for NTP and 1-10 for FA, so its answer is one digit token."""

    @property
    def levels(self) -> int:
        """Return the number of ordered answer categories."""
        return len(self.answer_columns)

    @property
    def answer_suffix(self) -> str:
        """Return what follows ``Answer:``: a space for numerical questions, else nothing."""
        return " " if self.numerical else ""

    def question_text(self, mode: PromptMode) -> str:
        """Return the closing question wording for one generation strategy."""
        return self.full_q_ntp if mode == "ntp" else self.full_q

    def normalize(self, values: pd.Series) -> pd.Series:
        """Map raw answers onto this question's canonical strings.

        ``d_polpos`` is stored as a float in the WVS and an int in the FA
        outputs, so both become ``"7"``; categorical answers already match.
        """
        if not self.numerical:
            return values.astype("object")
        return values.map(lambda value: pd.NA if pd.isna(value) else str(int(value)))

    def ntp_answers(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return an NTP frame's answer block under this question's column names.

        Two layouts appear: the archived CSVs name d_polpos' answers ``X0``-``X9``,
        while this reproduction's own runs write them as ``answer_columns``. Both
        are accepted, and renaming the archived block by position is what aligns
        the NTP 0-9 scale with the WVS 1-10 one.
        """
        for source in (self.ntp_csv_columns, self.answer_columns):
            if set(source).issubset(frame.columns):
                block = frame.loc[:, list(source)].copy()
                block.columns = list(self.answer_columns)
                return block
        raise KeyError(
            f"NTP frame for {self.var} has none of the expected answer columns "
            f"{list(self.ntp_csv_columns)} or {list(self.answer_columns)}; "
            f"found {list(frame.columns)}"
        )


def _letters(count: int) -> tuple[str, ...]:
    return LETTERS[:count]


def _lettered(labels: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        f"{letter}. {label}" for letter, label in zip(_letters(len(labels)), labels, strict=True)
    )


def _options(labels: tuple[str, ...]) -> str:
    return " or ".join(f"'{answer}'" for answer in _lettered(labels))


def _spaced(count: int) -> tuple[str, ...]:
    return tuple(f" {letter}" for letter in _letters(count))


_HAPPY = ("Very happy", "Quite happy", "Not very happy", "Not at all happy")
_RELIGIOUS = (
    "More than once a week",
    "Once a week",
    "Once a month",
    "Only on special holy days",
    "Once a year",
    "Less often",
    "Never, practically never",
)
_TRUST = ("Most people can be trusted", "Need to be very careful")
_POLPOS = tuple(str(value) for value in range(1, 11))

_HAPPY_Q = f"Taking all things together, how happy are you? Are you {_options(_HAPPY)}?"
_RELIGIOUS_Q = (
    "Apart from weddings, funerals and christenings, about how often do you attend "
    f"religious services these days? Is it {_options(_RELIGIOUS)}?"
)
_TRUST_Q = (
    "Generally speaking, would you say that most people can be trusted or that you need "
    f"to be very careful in dealing with people? Is it {_options(_TRUST)}?"
)
_POLPOS_Q = (
    'In political matters, people talk of "the left" and "the right." How would you place '
    "your views on a scale of {low} to {high}, where {low} is left and {high} is right, "
    "generally speaking?"
)


QUESTIONS: dict[str, Question] = {
    "d_happy": Question(
        var="d_happy",
        label="Happiness",
        numerical=False,
        answer_columns=_letters(4),
        ntp_csv_columns=_letters(4),
        wvs_labels=_HAPPY,
        fa_answers=_lettered(_HAPPY),
        ntp_tokens=_spaced(4),
        full_q=_HAPPY_Q,
        full_q_ntp=_HAPPY_Q,
    ),
    "d_polpos": Question(
        var="d_polpos",
        label="Politics",
        numerical=True,
        answer_columns=_letters(10),
        ntp_csv_columns=tuple(f"X{value}" for value in range(10)),
        wvs_labels=_POLPOS,
        fa_answers=_POLPOS,
        ntp_tokens=tuple(str(value) for value in range(10)),
        full_q=_POLPOS_Q.format(low=1, high=10),
        full_q_ntp=_POLPOS_Q.format(low=0, high=9),
    ),
    "d_religiousp": Question(
        var="d_religiousp",
        label="Religion",
        numerical=False,
        answer_columns=_letters(7),
        ntp_csv_columns=_letters(7),
        wvs_labels=_RELIGIOUS,
        fa_answers=_lettered(_RELIGIOUS),
        ntp_tokens=_spaced(7),
        full_q=_RELIGIOUS_Q,
        full_q_ntp=_RELIGIOUS_Q,
    ),
    "d_trust": Question(
        var="d_trust",
        label="Trust",
        numerical=False,
        answer_columns=_letters(2),
        ntp_csv_columns=_letters(2),
        wvs_labels=_TRUST,
        fa_answers=_lettered(_TRUST),
        ntp_tokens=_spaced(2),
        full_q=_TRUST_Q,
        full_q_ntp=_TRUST_Q,
    ),
}

QUESTION_NAMES: tuple[str, ...] = tuple(QUESTIONS)
DEFAULT_QUESTION = "d_happy"


def resolve_question(name: str | Question) -> Question:
    """Resolve a question name to its registry entry."""
    if isinstance(name, Question):
        return name
    try:
        return QUESTIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown question: {name!r} (known: {', '.join(QUESTION_NAMES)})"
        ) from None


def resolve_questions(selected: Sequence[str] | None) -> list[Question]:
    """Resolve question names, defaulting to all four; ``all`` expands to all four."""
    if not selected or list(selected) == ["all"]:
        return [QUESTIONS[name] for name in QUESTION_NAMES]
    return [resolve_question(name) for name in selected]


def one_hot(
    values: pd.Series,
    categories: tuple[str, ...],
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """One-hot encode answers, leaving a missing answer missing rather than zero."""
    categorical = pd.Categorical(values, categories=categories)
    frame = pd.get_dummies(categorical, dtype=np.float64)
    frame.columns = list(columns)
    frame.index = values.index
    frame.loc[values.isna(), :] = np.nan
    return frame
