from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

WVS_COUNTRIES = ("Australia", "Germany", "Mexico", "Russia", "United States")

CULTURE_COUNTRIES: dict[str, tuple[str, ...]] = {
    "english": ("Australia", "United States"),
    "german": ("Germany",),
    "spanish": ("Mexico",),
}

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
    return subpopulation.str.replace(r"^(.*?) \d.*$", r"\1", regex=True)


def countries_for(culture: str) -> tuple[str, ...]:
    return CULTURE_COUNTRIES.get(culture, ())


def restrict_to(
    frame: pd.DataFrame,
    countries: Sequence[str],
    column: str = "subpopulation",
) -> pd.DataFrame:
    if not countries:
        return frame.iloc[0:0]
    return frame[country_of(frame[column]).isin(list(countries))]


def restrict(frame: pd.DataFrame, culture: str, column: str = "subpopulation") -> pd.DataFrame:
    return restrict_to(frame, countries_for(culture), column)


def home_splits(culture: str) -> list[tuple[str, tuple[str, ...]]]:
    countries = countries_for(culture)
    if not countries:
        return []
    pooled = [("; ".join(countries), countries)]
    if len(countries) == 1:
        return pooled
    return pooled + [(country, (country,)) for country in countries]
