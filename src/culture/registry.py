"""Which culture models and adapters exist, and where their artifacts live."""

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

CONDITION = "cultural"
"""Sub-directory of a checkpoint tree holding the end-of-training adapter."""

PREFLIGHT_PROMPTS = 32
"""Prompts sampled by the pre-flight probe before a run commits to inference."""

PREFLIGHT_MIN_MASS = 0.10
"""Minimum mean valid-answer probability mass for a run to be treated as informative.

Below this an adapter puts essentially no probability on any accepted answer, so
its nEMD is noise rather than a measurement. The run still executes and is still
reported: the flag says the number is not an alignment result, it does not drop
the run.
"""

DEGENERATE_FA_RETRIES = 3
"""FA retry budget for a run whose probe found no valid-answer mass.

Fifty retries cannot produce an answer the model never had probability on.
"""

ADAPTERS_ROOT = MODELS_ROOT / "culture"
ADAPTERS_MANIFEST = ADAPTERS_ROOT / "ADAPTERS.json"
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT.parent / "culture-mllm" / "checkpoints"

CULTURE_ROOT = OUTPUTS_ROOT / "culture"
CULTURE_FIGURES = FIGURES_ROOT / "culture"
"""Where every culture table and figure lands, under the repository's own trees.

Named here rather than in a figure module so a table writer and a figure writer
cannot drift apart on where a run's artifacts belong.
"""


@dataclass(frozen=True, slots=True)
class CultureModel:
    """One fine-tuned base model and how inference must load it.

    ``quantization`` and ``dtype`` mirror the training configuration, so the
    adapter composes onto weights in the numeric format it was fitted against:

    * ``nf4`` — the 31B base is ~62 GB in bfloat16, so training used 4-bit
      QLoRA to fit a 32 GB card and inference must match.
    * ``fp8-dequantized`` — the deep-gemm kernel native FP8 needs rejects this
      GPU's recipe, so weights dequantize to bfloat16 at load. 17.5 GB resident.
    """

    key: str
    base_model_id: str
    label: str
    dtype: str
    quantization: str | None
    batch_size: int

    def run_label(self, culture: str) -> str:
        """Return the human-readable model name used in reports."""
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
    "qwen3_vl_8b": CultureModel(
        key="qwen3_vl_8b",
        base_model_id="Qwen/Qwen3-VL-8B-Thinking-FP8",
        label="Qwen3-VL-8B-Thinking-FP8",
        dtype="bfloat16",
        quantization="fp8-dequantized",
        batch_size=16,
    ),
}


def resolve_models(selected: Sequence[str] | None) -> list[CultureModel]:
    """Resolve model keys to registry entries, defaulting to all of them."""
    keys = list(selected) if selected else list(CULTURE_MODELS)
    unknown = [key for key in keys if key not in CULTURE_MODELS]
    if unknown:
        raise ValueError(f"unknown model keys: {', '.join(unknown)}")
    return [CULTURE_MODELS[key] for key in keys]


def resolve_cultures(selected: Sequence[str] | None) -> list[str]:
    """Resolve culture names, defaulting to all nine; ``all`` expands."""
    cultures = list(selected) if selected else list(CULTURES)
    if cultures == ["all"]:
        cultures = list(CULTURES)
    unknown = [culture for culture in cultures if culture not in CULTURES]
    if unknown:
        raise ValueError(f"unknown cultures: {', '.join(unknown)}")
    return cultures


def run_slug(model_key: str, culture: str) -> str:
    """Return the run identifier naming this model/culture pair's artifacts."""
    return f"culture/{model_key}/{culture}"


def run_paths(model_key: str, culture: str, question: str | Question) -> RunPaths:
    """Return the output and figure layout for one model/culture/question run."""
    return paths_for(run_slug(model_key, culture), resolve_question(question).var)


def csv_stems(model_key: str, culture: str, question: str | Question) -> tuple[str, str]:
    """Return the NTP and FA consolidated CSV filenames for one run."""
    var = resolve_question(question).var
    return (
        f"NTP-{model_key}-{culture}-{var}.csv",
        f"FA-{model_key}-{culture}-{var}.csv",
    )
