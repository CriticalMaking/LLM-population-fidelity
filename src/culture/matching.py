"""Which cultures the WVS happiness slice can actually speak to."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

WVS_COUNTRIES = ("Australia", "Germany", "Mexico", "Russia", "United States")

CULTURE_COUNTRIES: dict[str, tuple[str, ...]] = {
    "english": ("Australia", "United States"),
    "german": ("Germany",),
    "spanish": ("Mexico",),
}
"""Cultures whose language is dominant in a WVS respondent country.

Russia has respondents but no russian finetuned culture MLLM, so its
subpopulations appear in no matched panel.
"""

MATCHED_CULTURES: tuple[str, ...] = tuple(CULTURE_COUNTRIES)

UNMATCHED_CULTURES: tuple[str, ...] = (
    "arabic",
    "bengali",
    "chinese",
    "korean",
    "portuguese",
    "turkish",
)

UNREPRESENTED_COUNTRIES: tuple[str, ...] = ("Russia",)


def country_of(subpopulation: pd.Series) -> pd.Series:
    """Extract the respondent country from a subpopulation name.

    Uses the pattern ``analysis._social_design`` applies, so a country label
    here can never drift from the one the regressions use.
    """
    return subpopulation.str.replace(r"^(.*?) \d.*$", r"\1", regex=True)


def countries_for(culture: str) -> tuple[str, ...]:
    """Return the WVS countries a culture is directly associated with."""
    return CULTURE_COUNTRIES.get(culture, ())


def restrict_to(
    frame: pd.DataFrame,
    countries: Sequence[str],
    column: str = "subpopulation",
) -> pd.DataFrame:
    """Keep only the rows whose respondents live in one of ``countries``.

    An empty country list returns an empty frame rather than the full one:
    silently returning everything would let "no matched respondents" be read as
    a result about all of them.
    """
    if not countries:
        return frame.iloc[0:0]
    return frame[country_of(frame[column]).isin(list(countries))]


def restrict(frame: pd.DataFrame, culture: str, column: str = "subpopulation") -> pd.DataFrame:
    """Keep only the rows whose respondents belong to this culture's countries."""
    return restrict_to(frame, countries_for(culture), column)


def home_splits(culture: str) -> list[tuple[str, tuple[str, ...]]]:
    """Return every home-country slice worth scoring a culture against.

    The pooled slice always comes first. A culture spanning more than one country
    -- only ``english``, over Australia and the United States -- also yields each
    country on its own, so a pooled home advantage can be checked against the two
    respondent blocks that make it up rather than assumed uniform across them.
    """
    countries = countries_for(culture)
    if not countries:
        return []
    pooled = [("; ".join(countries), countries)]
    if len(countries) == 1:
        return pooled
    return pooled + [(country, (country,)) for country in countries]
