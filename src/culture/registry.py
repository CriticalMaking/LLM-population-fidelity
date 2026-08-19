from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from machine_bias_reproduction.config import (
    FIGURES_ROOT,
    MODELS_ROOT,
    OUTPUTS_ROOT,
    PROJECT_ROOT,
    RunPaths,
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
    "turkish",
)

BASE_ARM = "base"

ARMS: tuple[str, ...] = (BASE_ARM, *CULTURES)

CHECKPOINT_CONDITION_DIRECTORY = "cultural"

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


@dataclass(frozen=True, slots=True)
class CultureModel:
    key: str
    base_model_id: str
    label: str
    dtype: str
    quantization: str | None
    batch_size: int

    def run_label(self, culture: str) -> str:
        if is_base(culture):
            return f"{self.label} (base, not finetuned)"
        return f"{self.label} + {culture} LoRA"


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
    "muse_glimmer_30b": CultureModel(
        key="muse_glimmer_30b",
        base_model_id="meta-models/Muse-Glimmer-30B",
        label="Muse-Glimmer-30B",
        dtype="bfloat16",
        quantization="nf4",
        batch_size=4,
    ),
}


def resolve_models(selected: Sequence[str] | None) -> list[CultureModel]:
    keys = list(selected) if selected else list(CULTURE_MODELS)
    unknown = [key for key in keys if key not in CULTURE_MODELS]
    if unknown:
        raise ValueError(f"unknown model keys: {', '.join(unknown)}")
    return [CULTURE_MODELS[key] for key in keys]


def is_base(culture: str) -> bool:
    return culture == BASE_ARM


def resolve_cultures(selected: Sequence[str] | None) -> list[str]:
    cultures = list(selected) if selected else list(ARMS)
    if cultures == ["all"]:
        cultures = list(ARMS)
    unknown = [culture for culture in cultures if culture not in ARMS]
    if unknown:
        raise ValueError(f"unknown cultures: {', '.join(unknown)}")
    return cultures


def resolve_finetuned_cultures(selected: Sequence[str] | None) -> list[str]:
    cultures = list(selected) if selected else list(CULTURES)
    if cultures == ["all"]:
        cultures = list(CULTURES)
    if any(is_base(culture) for culture in cultures):
        raise ValueError(f"{BASE_ARM!r} is not culture-finetuned; there is nothing to stage")
    unknown = [culture for culture in cultures if culture not in CULTURES]
    if unknown:
        raise ValueError(f"unknown cultures: {', '.join(unknown)}")
    return cultures


def run_slug(model_key: str, culture: str) -> str:
    return f"culture/{model_key}/{culture}"


def run_paths(model_key: str, culture: str, question: str | Question) -> RunPaths:
    return paths_for(run_slug(model_key, culture), resolve_question(question).var)


def csv_stems(model_key: str, culture: str, question: str | Question) -> tuple[str, str]:
    var = resolve_question(question).var
    return (
        f"NTP-{model_key}-{culture}-{var}.csv",
        f"FA-{model_key}-{culture}-{var}.csv",
    )
