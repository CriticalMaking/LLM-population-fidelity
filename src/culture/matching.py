"""Which cultures the WVS happiness slice can actually speak to."""

from __future__ import annotations

import pandas as pd

WVS_COUNTRIES = ("Australia", "Germany", "Mexico", "Russia", "United States")

CULTURE_COUNTRIES: dict[str, tuple[str, ...]] = {
    "english": ("Australia", "United States"),
    "german": ("Germany",),
    "spanish": ("Mexico",),
}
"""Cultures whose language is dominant in a WVS respondent country.

Russia has respondents but no russian adapter, so its subpopulations appear in
no matched panel.
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


def restrict(frame: pd.DataFrame, culture: str, column: str = "subpopulation") -> pd.DataFrame:
    """Keep only the rows whose respondents belong to this culture's countries.

    An unmatched culture returns an empty frame rather than the full one: there
    is no matched subset, and silently returning everything would let an
    unmatched culture be read as a matched result.
    """
    countries = countries_for(culture)
    if not countries:
        return frame.iloc[0:0]
    return frame[country_of(frame[column]).isin(countries)]
