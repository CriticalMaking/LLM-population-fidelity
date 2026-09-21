from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from machine_bias_reproduction.config import (
    FIGURES_ROOT,
    FIRST_REPLICATE,
    MODELS_ROOT,
    OUTPUTS_ROOT,
    PROJECT_ROOT,
    RunPaths,
    check_replicate,
    paths_for,
)
from machine_bias_reproduction.questions import Question, resolve_question

CULTURES: tuple[str, ...] = (
    "arabic",
    "bengali",
    "chinese",
    "english",
    "german",
    "korean",
    "portuguese",
    "spanish",
    "spanish-mx",
    "turkish",
)

BASE_ARM = "base"

GLOBAL_ARM = "global"

SUBPOP_ARM = "subpop"

DISTRIBUTION_TRAINED_ARMS: tuple[str, ...] = (GLOBAL_ARM, SUBPOP_ARM)

ARM_DISPLAY = {
    BASE_ARM: "as released",
    GLOBAL_ARM: "distribution-matched",
    SUBPOP_ARM: "subgroup-matched",
}

SERVED_BACKEND = "openai"

FINETUNED_ARMS: tuple[str, ...] = (*CULTURES, *DISTRIBUTION_TRAINED_ARMS)

ARMS: tuple[str, ...] = (BASE_ARM, *FINETUNED_ARMS)

CHECKPOINT_CONDITION_DIRECTORY = "cultural"

DISTRIBUTIONAL_CONDITION_DIRECTORY = "distributional"

SUBPOP_CONDITION_DIRECTORY = "subpop"

ARM_CONDITIONS: dict[str, str] = {
    GLOBAL_ARM: DISTRIBUTIONAL_CONDITION_DIRECTORY,
    SUBPOP_ARM: SUBPOP_CONDITION_DIRECTORY,
}

WEIGHTS_FILE = "adapter_model.safetensors"

PREFLIGHT_PROMPTS = 32

PREFLIGHT_MIN_VALID_ANSWER_MASS = 0.10

ADAPTER_MIN_EVAL_TOKEN_ACCURACY = 0.50

DEGENERATE_FA_RETRIES = 3

ADAPTERS_ROOT = MODELS_ROOT / "culture"
ADAPTERS_MANIFEST = ADAPTERS_ROOT / "ADAPTERS.json"
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT.parent / "culture-mllm" / "checkpoints"

CULTURE_ROOT = OUTPUTS_ROOT / "culture"
CULTURE_FIGURES = FIGURES_ROOT / "culture"

REPLICATE_DIRECTORY = "rep{}"
REPLICATE_PATTERN = re.compile(r"^rep([2-9][0-9]*)$")


VISION_TEXT_MODALITY = "vision_text"

TEXT_MODALITY = "text"


@dataclass(frozen=True, slots=True)
class CultureModel:
    key: str
    base_model_id: str
    label: str
    dtype: str
    quantization: str | None
    batch_size: int
    backend: str = "transformers"
    modality: str = VISION_TEXT_MODALITY

    def run_label(self, culture: str) -> str:
        if is_base(culture):
            return f"{self.label} ({arm_display(culture)})"
        return f"{self.label} + {arm_display(culture)} LoRA"


CULTURE_MODELS: dict[str, CultureModel] = {
    "gemma4_31b": CultureModel(
        key="gemma4_31b",
        base_model_id="google/gemma-4-31B-it",
        label="Gemma-4-31B-it",
        dtype="bfloat16",
        quantization="nf4",
        batch_size=4,
    ),
    "gemma4_e4b": CultureModel(
        key="gemma4_e4b",
        base_model_id="google/gemma-4-E4B-it",
        label="Gemma-4-E4B-it",
        dtype="bfloat16",
        quantization=None,
        batch_size=16,
    ),
    "qwen3_vl_8b": CultureModel(
        key="qwen3_vl_8b",
        base_model_id="Qwen/Qwen3-VL-8B-Thinking",
        label="Qwen3-VL-8B-Thinking",
        dtype="bfloat16",
        quantization=None,
        batch_size=16,
    ),
    "qwen3_vl_2b": CultureModel(
        key="qwen3_vl_2b",
        base_model_id="Qwen/Qwen3-VL-2B-Thinking",
        label="Qwen3-VL-2B-Thinking",
        dtype="bfloat16",
        quantization=None,
        batch_size=32,
    ),
    "llama3_2_3b": CultureModel(
        key="llama3_2_3b",
        base_model_id="meta-llama/Llama-3.2-3B",
        label="Llama-3.2-3B",
        dtype="bfloat16",
        quantization=None,
        batch_size=32,
        modality=TEXT_MODALITY,
    ),
    "muse_glimmer_30b": CultureModel(
        key="muse_glimmer_30b",
        base_model_id="meta-models/Muse-Glimmer-30B",
        label="Muse-Glimmer-30B",
        dtype="bfloat16",
        quantization="nf4",
        batch_size=4,
    ),
    "luna": CultureModel(
        key="luna",
        base_model_id="luna",
        label="Luna",
        dtype="api",
        quantization=None,
        batch_size=4,
        backend="openai",
    ),
    "terra": CultureModel(
        key="terra",
        base_model_id="terra",
        label="GPT-5.6-Terra",
        dtype="api",
        quantization=None,
        batch_size=4,
        backend="openai",
    ),
    "sol": CultureModel(
        key="sol",
        base_model_id="sol",
        label="Sol",
        dtype="api",
        quantization=None,
        batch_size=4,
        backend="openai",
    ),
}


SIZE_TIERS: dict[str, tuple[str, ...]] = {
    "small": ("qwen3_vl_2b", "llama3_2_3b", "gemma4_e4b"),
    "mid": ("qwen3_vl_8b",),
    "big": ("gemma4_31b", "muse_glimmer_30b"),
}


def tier_models(tier: str) -> list[CultureModel]:
    return [CULTURE_MODELS[key] for key in SIZE_TIERS[tier]]


def served_models() -> list[CultureModel]:
    return [model for model in CULTURE_MODELS.values() if model.backend == SERVED_BACKEND]


def served_keys() -> tuple[str, ...]:
    return tuple(model.key for model in served_models())


def is_served(model_key: str) -> bool:
    model = CULTURE_MODELS.get(model_key)
    return model is not None and model.backend == SERVED_BACKEND


def resolve_models(selected: Sequence[str] | None) -> list[CultureModel]:
    keys = list(selected) if selected else list(CULTURE_MODELS)
    unknown = [key for key in keys if key not in CULTURE_MODELS]
    if unknown:
        raise ValueError(f"unknown model keys: {', '.join(unknown)}")
    return [CULTURE_MODELS[key] for key in keys]


def is_base(culture: str) -> bool:
    return culture == BASE_ARM


def is_global(arm: str) -> bool:
    return arm == GLOBAL_ARM


def checkpoint_condition(arm: str) -> str:
    return ARM_CONDITIONS.get(arm, CHECKPOINT_CONDITION_DIRECTORY)


def arm_display(arm: str) -> str:
    return ARM_DISPLAY.get(arm, arm)


def resolve_cultures(selected: Sequence[str] | None) -> list[str]:
    cultures = list(selected) if selected else list(ARMS)
    if cultures == ["all"]:
        cultures = list(ARMS)
    unknown = [culture for culture in cultures if culture not in ARMS]
    if unknown:
        raise ValueError(f"unknown cultures: {', '.join(unknown)}")
    return cultures


def resolve_finetuned_cultures(selected: Sequence[str] | None) -> list[str]:
    cultures = list(selected) if selected else list(FINETUNED_ARMS)
    if cultures == ["all"]:
        cultures = list(FINETUNED_ARMS)
    if any(is_base(culture) for culture in cultures):
        raise ValueError(f"{BASE_ARM!r} is not culture-finetuned; there is nothing to stage")
    unknown = [culture for culture in cultures if culture not in FINETUNED_ARMS]
    if unknown:
        raise ValueError(f"unknown cultures: {', '.join(unknown)}")
    return cultures


def replicate_directory(replicate: int) -> str | None:
    if check_replicate(replicate) == FIRST_REPLICATE:
        return None
    return REPLICATE_DIRECTORY.format(replicate)


def run_slug(model_key: str, culture: str, replicate: int = FIRST_REPLICATE) -> str:
    slug = f"culture/{model_key}/{culture}"
    directory = replicate_directory(replicate)
    return slug if directory is None else f"{slug}/{directory}"


def run_paths(
    model_key: str,
    culture: str,
    question: str | Question,
    replicate: int = FIRST_REPLICATE,
) -> RunPaths:
    return paths_for(run_slug(model_key, culture, replicate), resolve_question(question).var)


def replicates(model_key: str, culture: str, root: Path | None = None) -> list[int]:
    arm = (root or CULTURE_ROOT) / model_key / culture
    found = [FIRST_REPLICATE]
    if arm.is_dir():
        for entry in arm.iterdir():
            matched = REPLICATE_PATTERN.match(entry.name)
            if matched and entry.is_dir():
                found.append(int(matched.group(1)))
    return sorted(found)


def csv_stems(model_key: str, culture: str, question: str | Question) -> tuple[str, str]:
    var = resolve_question(question).var
    return (
        f"NTP-{model_key}-{culture}-{var}.csv",
        f"FA-{model_key}-{culture}-{var}.csv",
    )
