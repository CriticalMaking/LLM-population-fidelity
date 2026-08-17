from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from machine_bias_reproduction.config import FIGURES_ROOT, OUTPUTS_ROOT, PROJECT_ROOT
from machine_bias_reproduction.io_utils import atomic_write_json, hash_paths
from machine_bias_reproduction.questions import resolve_questions

from . import backtranslation, bootstrap, discriminator, figures_appendix, figures_main, robustness
from . import tables as tables_module
from .load import load_question
from .series import COEFFICIENT_SERIES, MAIN_MODELS, SMOKE_SERIES, resolve_series

REPORTS_OUTPUTS = OUTPUTS_ROOT / "reports"
REPORTS_FIGURES = FIGURES_ROOT / "reports"
TEX_DIRECTORY = REPORTS_OUTPUTS / "tex"

EXCLUSIONS = {
    "Figure 1": "Schematic in the paper; no generator in the replication package.",
    "Figure 5": "Schematic in the paper; no generator in the replication package.",
    "Table 1": "Descriptive listing in the paper; not produced by 6-results.R.",
    "Figure S3": "No generator in 6-results.R; the MDS set jumps S2 to S4.",
    "Figure S11": "No generator in 6-results.R.",
    "Table S3": "No generator in 6-results.R; the table set jumps S2 to S4.",
}

NOTES = {
    "Figure S1": (
        "6-results.R:1238 prints this to a device and never saves it; written here so the "
        "appendix set is complete on disk."
    ),
    "Figure S12": (
        "6-results.R:1438 writes this plate as Figure-S10-prompting-strategy-MDS.png into the "
        "working directory. Emitted here under its correct S12 name."
    ),
    "Table S5": "Identical in construction to Table 3; both are emitted from one computation.",
    "Figures S5-S8": (
        "6-results.R:1287 redraws Figure 3 for the remaining series rather than generating "
        "separate plates. Emitted here under the Figure-3-social-<series> names."
    ),
    "Table S8": "sklearn RandomForestClassifier(oob_score=True) stands in for R ranger.",
    "Figure S16": "statsmodels MNLogit stands in for R nnet::multinom.",
}

SECTIONS = ("all", "main", "appendix", "robustness", "tables", "figures", "smoke")


def _write(frame: pd.DataFrame, name: str, caption: str, label: str) -> list[Path]:
    if frame is None or frame.empty:
        return []
    REPORTS_OUTPUTS.mkdir(parents=True, exist_ok=True)
    TEX_DIRECTORY.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_OUTPUTS / f"{name}.csv"
    tex_path = TEX_DIRECTORY / f"{name}.tex"
    frame.to_csv(csv_path, index=False)
    tex_path.write_text(tables_module.to_latex(frame, caption, label), encoding="utf-8")
    return [csv_path, tex_path]


def build_reports(
    section: str = "all",
    *,
    questions: Sequence[str] | None = None,
    series: Sequence[str] | None = None,
) -> dict[str, Any]:
    if section not in SECTIONS:
        raise ValueError(f"unknown section: {section} (known: {', '.join(SECTIONS)})")
    smoke = section == "smoke"
    selected_questions = resolve_questions(["d_happy"] if smoke else questions)
    selected_series = resolve_series(list(SMOKE_SERIES) if smoke else series)

    wants_tables = section in {"all", "main", "appendix", "robustness", "tables", "smoke"}
    wants_figures = section in {"all", "main", "appendix", "robustness", "figures", "smoke"}
    wants_main = section in {"all", "main", "tables", "figures", "smoke"}
    wants_appendix = section in {"all", "appendix", "tables", "figures"}
    wants_robustness = section in {"all", "robustness"}

    REPORTS_FIGURES.mkdir(parents=True, exist_ok=True)
    loaded = {
        question.var: load_question(question, selected_series) for question in selected_questions
    }
    written: list[Path] = []
    figure_paths: list[Path] = []
    produced: dict[str, list[str]] = {"tables": [], "figures": []}
    skipped: list[str] = []

    distances = tables_module.subpopulation_distances(loaded)
    fit, f_tests, coefficients = (
        tables_module.table5_and_6(loaded)
        if wants_main or wants_appendix
        else (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
    )

    if wants_tables and wants_main:
        for frame, name, caption, label in (
            (
                tables_module.table2(loaded),
                "table-2-average-nemd",
                "nEMD between average model predictions and average WVS answers.",
                "tab:average-nemd",
            ),
            (
                tables_module.table3(distances),
                "table-3-quality",
                "Share of subpopulations in each prediction-quality band.",
                "tab:quality",
            ),
            (
                tables_module.table4(loaded),
                "table-4-pairwise",
                "Median pairwise nEMD within each series.",
                "tab:pairwise",
            ),
            (fit, "table-5-regression-fit", "Regression goodness-of-fit.", "tab:regression-fit"),
            (f_tests, "table-6-f-tests", "F-tests of nested models.", "tab:f-tests"),
            (
                tables_module.response_distributions(loaded),
                "response-distributions",
                "Marginal answer distributions by question and series.",
                "tab:response-distributions",
            ),
            (
                distances,
                "subpopulation-distances",
                "Per-subpopulation nEMD for every method.",
                "tab:subpopulation-distances",
            ),
            (
                coefficients,
                "regression-coefficients",
                "Regression coefficients behind Figures 3 and 6.",
                "tab:coefficients",
            ),
        ):
            paths = _write(frame, name, caption, label)
            written.extend(paths)
            (produced["tables"] if paths else skipped).append(name)

    if wants_tables and wants_appendix:
        for frame, name, caption, label in (
            (
                tables_module.table_s1(),
                "table-S1-sample-size",
                "WVS sample size by country and survey year.",
                "tab:sample-size",
            ),
            (
                tables_module.table_s2(),
                "table-S2-predictors",
                "Descriptive statistics of predictor variables by country.",
                "tab:predictors",
            ),
            (
                tables_module.table_s4(loaded),
                "table-S4-ntp-compliance",
                "NTP compliance: mean probability mass on expected answers.",
                "tab:ntp-compliance",
            ),
        ):
            paths = _write(frame, name, caption, label)
            written.extend(paths)
            (produced["tables"] if paths else skipped).append(name)

    if wants_figures and wants_main and not distances.empty:
        figures = figures_main.figure2(distances, REPORTS_FIGURES)
        available = {entry.name for entry in selected_series}
        figures.extend(
            figures_main.figure4(
                loaded, [name for name in MAIN_MODELS if name in available], REPORTS_FIGURES
            )
        )
        for name in COEFFICIENT_SERIES:
            if name not in available or coefficients.empty:
                continue
            figures.extend(figures_main.figure3(coefficients, fit, name, REPORTS_FIGURES))
            figures.extend(figures_main.figure6(coefficients, fit, name, REPORTS_FIGURES))
        figure_paths.extend(figures)

    if wants_figures and wants_appendix:
        figures = figures_appendix.figure_s1(REPORTS_FIGURES)
        figures.extend(figures_appendix.figures_s2_s4(loaded, REPORTS_FIGURES))
        figures.extend(figures_appendix.figure_s9(fit, REPORTS_FIGURES))
        figures.extend(figures_appendix.figure_s10(loaded, REPORTS_FIGURES))
        figures.extend(figures_appendix.figure_s15(loaded, REPORTS_FIGURES))
        figure_paths.extend(figures)

    if wants_robustness:
        s9, orders = backtranslation.table_s9(loaded)
        temperature_figures, temperature_frame = robustness.figures_s13_s14(REPORTS_FIGURES)
        s16_figures, s16_frame = bootstrap.figure_s16(loaded, REPORTS_FIGURES)
        for frame, name, caption, label in (
            (
                robustness.table_s6(),
                "table-S6-prompting-strategies",
                "Median nEMD across GPT-4T prompting strategies.",
                "tab:prompting-strategies",
            ),
            (
                robustness.table_s7(),
                "table-S7-quantization",
                "nEMD between quantized and unquantized Mistral-7B.",
                "tab:quantization",
            ),
            (
                discriminator.table_s8(loaded),
                "table-S8-discriminator",
                "Out-of-bag accuracy of a random-forest discriminator.",
                "tab:discriminator",
            ),
            (
                s9,
                "table-S9-backtranslation",
                "Share of subpopulations reached by nearest-neighbour backtranslation.",
                "tab:backtranslation",
            ),
            (
                temperature_frame,
                "temperature-distances",
                "Per-subpopulation nEMD by full-answer sampling temperature.",
                "tab:temperature",
            ),
            (
                s16_frame,
                "table-S16-coefficient-comparison",
                "Ground-truth versus model answer-model coefficients.",
                "tab:coefficient-comparison",
            ),
        ):
            paths = _write(frame, name, caption, label)
            written.extend(paths)
            (produced["tables"] if paths else skipped).append(name)

        figures = robustness.figure_s12(REPORTS_FIGURES)
        figures.extend(temperature_figures)
        figures.extend(s16_figures)
        figures.extend(backtranslation.figures_s17_s18(loaded, orders, REPORTS_FIGURES))
        figure_paths.extend(figures)

    written.extend(figure_paths)
    produced["figures"] = sorted({path.stem for path in figure_paths})

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "section": section,
        "questions": [question.var for question in selected_questions],
        "series": [entry.name for entry in selected_series],
        "produced": produced,
        "empty_this_run": skipped,
        "not_reproduced": EXCLUSIONS,
        "notes": NOTES,
        "artifacts": hash_paths(written, PROJECT_ROOT),
    }
    REPORTS_OUTPUTS.mkdir(parents=True, exist_ok=True)
    atomic_write_json(REPORTS_OUTPUTS / "reports_manifest.json", manifest)
    _write_report_index(manifest)
    return {
        "section": section,
        "questions": manifest["questions"],
        "series": manifest["series"],
        "tables": len(produced["tables"]),
        "figures": len(produced["figures"]),
        "artifacts": len(written),
        "outputs": str(REPORTS_OUTPUTS),
        "figures_directory": str(REPORTS_FIGURES),
    }


def _write_report_index(manifest: dict[str, Any]) -> None:
    lines = [
        "# Paper report set",
        "",
        f"Section: `{manifest['section']}`  ",
        f"Questions: {', '.join(manifest['questions'])}  ",
        f"Series: {', '.join(manifest['series'])}",
        "",
        "## Tables",
        "",
        *(f"- `outputs/reports/{name}.csv`" for name in manifest["produced"]["tables"]),
        "",
        "## Figures",
        "",
        *(f"- `figures/reports/{name}.pdf`" for name in manifest["produced"]["figures"]),
        "",
        "## Not reproduced",
        "",
        *(f"- **{name}** — {EXCLUSIONS[name]}" for name in EXCLUSIONS),
        "",
        "## Notes",
        "",
        *(f"- **{name}** — {NOTES[name]}" for name in NOTES),
        "",
    ]
    (REPORTS_OUTPUTS / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
