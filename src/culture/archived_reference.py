"""Score the archived Machine Bias series with the population-fidelity metrics.

The upstream replication package holds per-profile NTP probability vectors for GPT-4T,
Llama-3-70B and Mixtral-8x7B, and full answers for GPT-3, Llama-3-70B and Mixtral-8x7B.
The sweep's fidelity tables carry the Mixtral pair alone. This scores every published
series the same way, on the cells it retains, so a served frontier model is read against
the proprietary models of the original study and not only against Mixtral. GPT-4T was
published under NTP alone and GPT-3 under FA alone, so a comparison across them is also
a comparison across elicitation modes; the Mixtral and Llama-3-70B rows, scored under
both modes on the same cells, are the bridge.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from machine_bias_reproduction.config import FIRST_REPLICATE, OUTPUTS_ROOT, UPSTREAM_DATA
from machine_bias_reproduction.data import PreparedData, archived_fa, archived_ntp, prepare_data
from machine_bias_reproduction.questions import Question, resolve_questions

from .fidelity import (
    IDENTITY,
    answer_array,
    binding_term,
    cell_facets,
    group_fidelity,
    grouped_scores,
    lead_with,
)
from .palette import MIXTRAL_ARCHIVED
from .population import REPLICATE

ARCHIVED_MODELS: tuple[str, ...] = ("GPT-3", "GPT-4T", "Llama-3-70B", "Mixtral-8x7B")

ARCHIVED_CSV = UPSTREAM_DATA / "LLM-outputs" / "csv"

ARCHIVED_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_archived.csv"

# Mixtral keeps the label the rest of the tables already use for the same run.
ARCHIVED_SERIES: dict[str, str] = {"Mixtral-8x7B": MIXTRAL_ARCHIVED}


def archived_series(model: str) -> str:
    return ARCHIVED_SERIES.get(model, f"{model} archived")


def archived_frames(
    model: str, question: Question
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """The published NTP and FA frames of one model, each None where the archive has none."""
    ntp_path = ARCHIVED_CSV / f"NTP-{model}-{question.var}.csv"
    fa_path = ARCHIVED_CSV / f"FA-{model}.csv"
    ntp = archived_ntp(question, model) if ntp_path.is_file() else None
    fa = archived_fa(question, model) if fa_path.is_file() else None
    return ntp, fa


def load_archived(model: str, question: Question) -> PreparedData | None:
    """One archived model prepared the way the sweep prepares its own runs.

    Cells are retained on every mode the archive holds for the model, so the Mixtral
    rows reproduce the reference rows of the fidelity tables to the last digit.
    """
    ntp, fa = archived_frames(model, question)
    if ntp is None and fa is None:
        return None
    return prepare_data(ntp, fa, question)


def archived_identity(model: str, question: Question, mode: str) -> dict[str, Any]:
    return {
        "question": question.var,
        "question_label": question.label,
        "model_key": None,
        "model_label": model,
        "arm": None,
        "series": archived_series(model),
        "source": f"archived/{model}",
        "mode": mode,
        REPLICATE: FIRST_REPLICATE,
    }


def build_archived(
    questions: list[Question] | None = None,
    models: tuple[str, ...] = ARCHIVED_MODELS,
) -> pd.DataFrame:
    """Every archived series scored per group and level, shaped like the groups table."""
    facets = cell_facets()
    rows: list[dict[str, Any]] = []
    for question in questions or resolve_questions(None):
        for model in models:
            prepared = load_archived(model, question)
            if prepared is None:
                continue
            names = prepared.names
            facet = facets.reindex(names)
            wvs = answer_array(prepared, prepared.wvs_props, names)
            for mode in prepared.modes():
                llm = answer_array(prepared, prepared.props(mode), names)
                identity = archived_identity(model, question, mode)
                rows += grouped_scores(identity, facet, group_fidelity, wvs, llm)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["binding_term"] = binding_term(table)
    return lead_with(table, [*IDENTITY, "group", "level"])
