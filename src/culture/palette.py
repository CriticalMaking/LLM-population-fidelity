from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from machine_bias_reproduction.plates import (
    METHOD_MARKERS,
    METHOD_TONES,
    MODEL_TONE,
    REFERENCE_TONE,
    RERUN_TONE,
    UNTUNED_TONE,
    Tone,
    slot_linestyle,
    slot_marker,
    slot_tone,
    tone,
)

from .registry import BASE_ARM, CULTURES

MODES: tuple[str, str] = ("NTP", "FA")

MODE_TONES = {mode: METHOD_TONES[mode] for mode in MODES}
MODE_MARKERS = dict(METHOD_MARKERS)

CULTURE_SLOTS = {culture: index for index, culture in enumerate(CULTURES)}

MIXTRAL_ARCHIVED = "Mixtral archived"
MIXTRAL_FRESH = "Mixtral fresh"

MDS_REFERENCES: tuple[tuple[str, str], ...] = (
    (MIXTRAL_ARCHIVED, "archived"),
    (MIXTRAL_FRESH, "fresh"),
)

REFERENCE_TONES: dict[str, Tone] = {
    MIXTRAL_ARCHIVED: REFERENCE_TONE,
    MIXTRAL_FRESH: RERUN_TONE,
    BASE_ARM: UNTUNED_TONE,
}

REFERENCE_MARKERS: dict[str, str] = {
    MIXTRAL_ARCHIVED: "o",
    MIXTRAL_FRESH: "s",
    BASE_ARM: "D",
}

REFERENCE_LINESTYLES: dict[str, Any] = {
    MIXTRAL_ARCHIVED: (0, (6, 3)),
    MIXTRAL_FRESH: (0, (2, 1.4)),
    BASE_ARM: (0, (7, 2, 1, 2)),
}

MODEL_SLOTS = {
    "gemma4_31b": 0,
    "gemma4_e4b": 1,
    "qwen3_vl_8b": 6,
    "qwen3_vl_2b": 2,
    "llama3_2_3b": 7,
    "muse_glimmer_30b": 3,
    "luna": 8,
    "terra": 4,
    "sol": 5,
}

MDS_ARM_TONES: dict[str, Tone] = {"spanish": tone("#1f78b4"), BASE_ARM: MODEL_TONE}


def arm_tone(arm: str) -> Tone:
    if arm in REFERENCE_TONES:
        return REFERENCE_TONES[arm]
    return slot_tone(CULTURE_SLOTS.get(arm, len(CULTURE_SLOTS)))


def mds_arm_tone(arm: str) -> Tone:
    return MDS_ARM_TONES.get(arm) or arm_tone(arm)


def arm_marker(arm: str) -> str:
    if arm in REFERENCE_MARKERS:
        return REFERENCE_MARKERS[arm]
    return slot_marker(CULTURE_SLOTS.get(arm, len(CULTURE_SLOTS)))


def arm_linestyle(arm: str) -> Any:
    if arm in REFERENCE_LINESTYLES:
        return REFERENCE_LINESTYLES[arm]
    return slot_linestyle(CULTURE_SLOTS.get(arm, len(CULTURE_SLOTS)))


def model_tone(model_key: str, index: int = 0) -> Tone:
    return slot_tone(MODEL_SLOTS.get(model_key, 2 + index))


def model_marker(model_key: str, index: int = 0) -> str:
    return slot_marker(MODEL_SLOTS.get(model_key, 2 + index))


def arm_order(arms: Iterable[str]) -> list[str]:
    names = sorted(arms)
    return [arm for arm in names if arm == BASE_ARM] + [arm for arm in names if arm != BASE_ARM]


def is_reference(label: str) -> bool:
    return label in (MIXTRAL_ARCHIVED, MIXTRAL_FRESH, BASE_ARM)
