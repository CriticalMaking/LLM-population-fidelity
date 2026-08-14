from __future__ import annotations

from collections.abc import Iterable

from .registry import BASE_ARM

MODES: tuple[str, str] = ("NTP", "FA")

MODE_TITLES = {"NTP": "Next-token probabilities", "FA": "Full answer"}

CULTURE_COLORS = {
    "arabic": "#1b9e77",
    "bengali": "#d95f02",
    "chinese": "#7570b3",
    "english": "#e7298a",
    "german": "#66a61e",
    "korean": "#e6ab02",
    "portuguese": "#a6761d",
    "spanish": "#666666",
    "turkish": "#1f78b4",
}

BASE_COLOR = "#111111"

DEFAULT_ARM_COLOR = "#e7298a"

REFERENCE_COLOR = "#000000"

QUALITY_PALETTE = ("#1b9e77", "#66c2a5", "#ffd92f", "#fc8d62", "#d73027")
OUTCOME_PALETTE = {"valid": "#1b9e77", "invalid": "#fc8d62", "failed": "#d73027"}

MDS_REFERENCES: tuple[tuple[str, str], ...] = (
    ("Mixtral archived", "archived"),
    ("Mixtral fresh", "fresh"),
)

MDS_SERIES_COLORS = {"Mixtral archived": REFERENCE_COLOR, "Mixtral fresh": "#c8c8c8"}
MDS_SERIES_MARKERS = {"Mixtral archived": "o", "Mixtral fresh": "s"}

MDS_CULTURE_COLORS = {"spanish": "#1f78b4"}


def culture_color(arm: str) -> str:
    if arm == BASE_ARM:
        return BASE_COLOR
    return CULTURE_COLORS.get(arm, DEFAULT_ARM_COLOR)


def arm_order(arms: Iterable[str]) -> list[str]:
    names = sorted(arms)
    return [arm for arm in names if arm == BASE_ARM] + [arm for arm in names if arm != BASE_ARM]


def is_reference(label: str) -> bool:
    return label in MDS_SERIES_COLORS


def mds_color(label: str) -> str:
    return MDS_SERIES_COLORS.get(label, MDS_CULTURE_COLORS.get(label, culture_color(label)))
