from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = PROJECT_ROOT / "upstream"
ARCHIVE_PATH = UPSTREAM_DIR / "Machine-Bias-replication.zip"
UPSTREAM_ROOT = UPSTREAM_DIR / "extracted" / "Machine-Bias-replication"
UPSTREAM_DATA = UPSTREAM_ROOT / "data"
UPSTREAM_CODE = UPSTREAM_ROOT / "code"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
FIGURES_ROOT = PROJECT_ROOT / "figures"
MODELS_ROOT = PROJECT_ROOT / "models"

ARCHIVE_SHA256 = "679f0726dcd8cd6ec0fdd6d3bd727e4c6b72b4224785cdd9eabd12bc564aa5e2"
MODEL_SHA256 = "5e066c60d89d904db46c3bf577661040578a223a5a39ba3fd23ff091549767f0"

EXPECTED_WVS_ROWS = 26_981
EXPECTED_NTP_PROFILES = 13_904
EXPECTED_SUBPOPULATIONS = 687
GLOBAL_SEED = 20_240_110
FIRST_REPLICATE = 1


def check_replicate(replicate: int) -> int:
    if replicate < FIRST_REPLICATE:
        raise ValueError(f"replicates count from {FIRST_REPLICATE}, got {replicate}")
    return replicate


def replicate_seed(replicate: int) -> int:
    return GLOBAL_SEED + check_replicate(replicate) - FIRST_REPLICATE


@dataclass(frozen=True, slots=True)
class RunPaths:
    source: str
    question: str

    @property
    def outputs(self) -> Path:
        return OUTPUTS_ROOT / self.source / self.question

    @property
    def figures(self) -> Path:
        return FIGURES_ROOT / self.source / self.question

    @property
    def logs(self) -> Path:
        return self.outputs / "logs"

    @property
    def raw(self) -> Path:
        return self.outputs / "raw"

    def ensure(self) -> None:
        self.outputs.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.raw.mkdir(parents=True, exist_ok=True)


CULTURE_SOURCE_PATTERN = re.compile(
    r"^culture/[A-Za-z0-9_]+/[A-Za-z0-9_][A-Za-z0-9_-]*(/rep[2-9][0-9]*)?$"
)

QUESTION_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def paths_for(source: str, question: str) -> RunPaths:
    if source not in {"archived", "fresh"} and not CULTURE_SOURCE_PATTERN.match(source):
        raise ValueError(f"unknown source: {source}")
    if not QUESTION_PATTERN.match(question):
        raise ValueError(f"unsafe question name: {question}")
    return RunPaths(source, question)
