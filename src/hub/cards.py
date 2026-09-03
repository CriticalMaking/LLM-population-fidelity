from __future__ import annotations

from pathlib import Path

DATASET_REPO_ID = "MInDS-lab-UTFPR/TowardsSociallyGroundedAISafety"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_REPO_ID}"
GITHUB_URL = "https://github.com/CriticalMaking/TowardsSociallyGroundedAISafety"

FRONT_MATTER = """---
pretty_name: Towards Socially Grounded AI Safety
license: cc-by-4.0
language:
  - en
tags:
  - synthetic
  - survey-simulation
  - social-bias
  - llm-outputs
configs:
  - config_name: raw
    data_files:
      - split: fa
        path: data/**/raw-fa.parquet
      - split: ntp
        path: data/**/raw-ntp.parquet
---
"""

ARMS = (
    ("`fresh`", "Mixtral-8x7B-v0.1 Q4_K_M (llama.cpp)", "reproduction of the original protocol"),
    ("`culture/gemma4_31b`", "google/gemma-4-31B-it", "`base`, `german` (QLoRA)"),
    ("`culture/gemma4_e4b`", "google/gemma-4-e4b-it", "`base`, `german` (LoRA)"),
    ("`culture/qwen3_vl_8b`", "Qwen/Qwen3-VL-8B-Thinking", "`base`, `german` (LoRA)"),
    ("`culture/qwen3_vl_2b`", "Qwen/Qwen3-VL-2B-Thinking", "`base`, `german` (LoRA)"),
    ("`culture/llama3_2_3b`", "meta-llama/Llama-3.2-3B", "`base`, `german` (LoRA)"),
    ("`culture/muse_glimmer_30b`", "meta-models/Muse-Glimmer-30B", "`base`, `german` (QLoRA)"),
    (
        "`culture/terra`",
        "served proprietary model, OpenAI-compatible API (codename `terra`)",
        "`base` only",
    ),
    ("`archived`", "upstream Mixtral outputs, reanalysed", "derived tables only (no raw records)"),
)


def _table(rows: tuple[tuple[str, ...], ...], header: tuple[str, ...]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _upstream_section(archive_bytes: int) -> str:
    if not archive_bytes:
        return ""
    return f"""
## Upstream replication package

`upstream/redraw-subset.tar.gz` ({archive_bytes / 2**20:.0f} MB) carries the part of the
Boelaert et al. replication package that the analysis reads: the WVS respondent table and
answer levels, the subpopulation cells, the archived Mixtral, Llama-3-70B and GPT-4T response
tables, the linear baselines and the original R and Python code. Pull it with
`hub pull --groups upstream` and extract it at the repository root with
`tar xzf hub/downloads/upstream/redraw-subset.tar.gz`; it lands under the gitignored
`upstream/extracted/Machine-Bias-replication/`, which is enough for every figure and table.
The survey data are the World Values Survey's: the replication package terms, not this
dataset's licence, govern their reuse.
"""


def dataset_card(counts: dict[str, int], repo_id: str = DATASET_REPO_ID) -> str:
    arms = _table(ARMS, ("Arm", "Model", "Variants"))
    upstream = _upstream_section(counts.get("upstream_bytes", 0))
    inventory = _table(
        (
            ("run directories", str(counts.get("runs", 0))),
            ("raw parquet files", str(counts.get("parquet", 0))),
            ("analysis tables", str(counts.get("tables", 0))),
        ),
        ("Contents", "Count"),
    )
    return f"""{FRONT_MATTER}
# Towards Socially Grounded AI Safety

LLM-generated survey responses from a reproduction of Boelaert et al. (2025), *Machine Bias*,
extended with culture-finetuned (German QLoRA) model arms. Each model simulates World Values
Survey respondents defined by country, survey wave, age, gender, education, employment, and
marital status, answering four opinion questions.

- **Code, inference and analysis:** {GITHUB_URL}
- **Dataset:** https://huggingface.co/datasets/{repo_id}

## Experimental arms

{arms}

Questions: `d_happy` (happiness), `d_polpos` (political position, 10-point),
`d_religiousp` (religiosity), `d_trust` (interpersonal trust).
Modes: `fa` (free answer, sampled completions) and `ntp` (next-token probabilities over the
answer options). Served arms (e.g. `terra`) provide `fa` only — the API exposes no
token-level probabilities for `ntp`.

## Inventory

{inventory}

## Layout

```
data/<arm>/<question>/
  raw-fa.parquet        per-prompt FA records (one row per simulated respondent)
  raw-ntp.parquet       per-prompt NTP records
  FA-*.csv, NTP-*.csv   consolidated response tables (id, profile, answer)
  inference_trace.csv   flat provenance index; record_path joins to the parquet rows
  subpopulation_distances.csv, regression_fit.csv, full_coefficients.csv,
  full_standardized_coefficients.csv, ...   derived per-run analysis tables
data/culture/model_standardized_coefficients.csv   pooled coefficient table behind the
                                                   fig_model_standardized_coefficients plates
data/reports/           cross-arm paper tables (nEMD, regressions, F-tests)
```
{upstream}
## Raw parquet schema

One row per prompt. `profile` is the `§`-delimited respondent cell, also split into
`country`, `wave`, `age`, `gender`, `education`, `employment`, `marital`. FA rows carry
`answer`, `attempts` (list of `{{index, accepted, seed, duration_ms, parsed, raw_text}}`), and
`seed`; NTP rows carry `mass` plus the per-option probability distribution in `result_json`.
Provenance columns: `run_id`, `created_at`, `model_ref` (GGUF filename or Hub model id),
`model_sha256` / `adapter_sha256`, `backend_name`, `backend_version`, `sampling_json`,
`code_revision`, `schema_version`. `record_path` matches the `record_path` column of the
sibling `inference_trace.csv`. Local filesystem paths from the trace blocks are omitted.

## License

CC-BY-4.0.
"""


def write_dataset_card(
    staging: Path, counts: dict[str, int], repo_id: str = DATASET_REPO_ID
) -> Path:
    staging.mkdir(parents=True, exist_ok=True)
    path = staging / "README.md"
    path.write_text(dataset_card(counts, repo_id), encoding="utf-8")
    return path
