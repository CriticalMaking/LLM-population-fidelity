"""Canonical paths and constants for the reproduction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = PROJECT_ROOT / "upstream"
ARCHIVE_PATH = UPSTREAM_DIR / "Machine-Bias-replication.zip"
PAPER_PATH = UPSTREAM_DIR / "paper.pdf"
UPSTREAM_ROOT = UPSTREAM_DIR / "extracted" / "Machine-Bias-replication"
UPSTREAM_DATA = UPSTREAM_ROOT / "data"
UPSTREAM_CODE = UPSTREAM_ROOT / "code"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
FIGURES_ROOT = PROJECT_ROOT / "figures"
MODELS_ROOT = PROJECT_ROOT / "models"

ARCHIVE_SHA256 = "679f0726dcd8cd6ec0fdd6d3bd727e4c6b72b4224785cdd9eabd12bc564aa5e2"
PAPER_SHA256 = "123585b09c98ec6546a8508b6d4bbf7032cd8b6f032cc560d88c5b8c9b724261"
MODEL_SHA256 = "5e066c60d89d904db46c3bf577661040578a223a5a39ba3fd23ff091549767f0"
MODEL_FILENAME = "mixtral-8x7b-v0.1.Q4_K_M.gguf"
MODEL_URL = (
    "https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF/resolve/"
    "89d949782453e711318567af765d39e77c57afb0/"
    "mixtral-8x7b-v0.1.Q4_K_M.gguf"
)

EXPECTED_WVS_ROWS = 26_981
EXPECTED_NTP_PROFILES = 13_904
EXPECTED_SUBPOPULATIONS = 687
GLOBAL_SEED = 20_240_110


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Filesystem layout for one run of one question."""

    source: str
    question: str

    @property
    def slug(self) -> str:
        """Return the run's path fragment, ``<source>/<question>``."""
        return f"{self.source}/{self.question}"

    @property
    def outputs(self) -> Path:
        """Return the run's output directory."""
        return OUTPUTS_ROOT / self.source / self.question

    @property
    def figures(self) -> Path:
        """Return the run's figure directory."""
        return FIGURES_ROOT / self.source / self.question

    @property
    def logs(self) -> Path:
        """Return the run's log directory."""
        return self.outputs / "logs"

    @property
    def raw(self) -> Path:
        """Return the per-prompt result directory."""
        return self.outputs / "raw"

    def ensure(self) -> None:
        """Create all mutable run directories."""
        self.outputs.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.raw.mkdir(parents=True, exist_ok=True)


CULTURE_SOURCE_PATTERN = re.compile(r"^culture/[A-Za-z0-9_]+/[A-Za-z0-9_]+$")
"""Match a ``culture/<model_key>/<culture>`` run name.

The components are restricted so a run name can never escape ``outputs/`` or
``figures/``.
"""

QUESTION_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def paths_for(source: str, question: str) -> RunPaths:
    """Resolve the mutable paths for one question of a named run."""
    if source not in {"archived", "fresh"} and not CULTURE_SOURCE_PATTERN.match(source):
        raise ValueError(f"unknown source: {source}")
    if not QUESTION_PATTERN.match(question):
        raise ValueError(f"unsafe question name: {question}")
    return RunPaths(source, question)
