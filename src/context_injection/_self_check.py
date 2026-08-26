from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from machine_bias_reproduction.prompts import build_prompt as baseline_prompt

from .audit import audit_mapping
from .base import ContextSpec
from .compare import stack_analysis, write_tables
from .conditions import build_first_pilot
from .export import build_rows, validate_columns, write_jsonl, write_manifest
from .metadata import variable_inventory
from .prompts import build_prompt
from .records import records_by_condition
from .run import condition_source, default_run_name, infer_question
from .theories.inglehart_welzel import InglehartWelzelMapping, TheoryItem


def main() -> None:
    row = pd.Series(
        {
            "id": 1,
            "profile": "United States 2017 40 Male High Working Single",
            "i_surveyyear": 2017,
            "i_country": "United States",
            "i_age": 40,
            "i_sex": "Male",
            "i_education": "High",
            "i_employment": "Working",
            "i_marstat": "Single",
            "d_trust": "Most people can be trusted",
        }
    )
    text, context = build_prompt(row, "d_trust", "ntp", ContextSpec())
    assert text == baseline_prompt(row, "d_trust", "ntp")
    assert context.context_text == ""

    spec = ContextSpec(strategy="raw", source_variables=("i_age", "d_trust"))
    text, context = build_prompt(row, "d_trust", "ntp", spec)
    assert "Age: 40" in text
    assert "d_trust" not in context.source_variables
    assert context.excluded_variables == ("d_trust",)

    pilot = build_first_pilot(
        row,
        "d_trust",
        raw_variables=("i_age", "i_sex", "d_trust"),
        selected_variables=("i_age", "d_trust"),
    )
    assert pilot["C1"].source_variables == pilot["C2"].source_variables
    assert pilot["C3"].source_variables == pilot["C4"].source_variables
    assert pilot["C1"].source_variables == ("i_age", "i_sex")
    assert pilot["C3"].source_variables == ("i_age",)

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "prompts.jsonl"
        manifest = Path(tmp) / "manifest.json"
        frame = pd.DataFrame([row])
        rows = build_rows(
            frame,
            question="d_trust",
            mode="ntp",
            raw_variables=("i_age", "d_trust"),
            selected_variables=("i_age", "d_trust"),
            conditions=("C0", "C1", "C2", "C3", "C4"),
        )
        assert write_jsonl(rows, path) == 5
        assert infer_question(path) == "d_trust"
        assert default_run_name(Path("reduced-d-trust.jsonl")) == "reduced_d_trust"
        assert condition_source("reduced_d_trust", "C0") == "culture/context_reduced_d_trust/C0"
        grouped = records_by_condition(path)
        assert sorted(grouped) == ["C0", "C1", "C2", "C3", "C4"]
        assert grouped["C1"][0].prompt_id == "United States 2017 40 Male High Working Single"
        qualified = records_by_condition(path, qualify_prompt_id=True)
        assert qualified["C1"][0].prompt_id.startswith("C1:")
        write_manifest(
            manifest,
            count=5,
            wvs_path=Path(tmp) / "wvs.csv",
            out_path=path,
            question="d_trust",
            mode="ntp",
            raw_variables=("i_age", "d_trust"),
            selected_variables=("i_age", "d_trust"),
            conditions=("C0", "C1", "C2", "C3", "C4"),
            theory_map=None,
        )
        assert '"record_count": 5' in manifest.read_text()
        try:
            validate_columns(
                frame.drop(columns=["i_age"]),
                mode="ntp",
                raw_variables=("i_age",),
                selected_variables=(),
            )
        except ValueError as error:
            assert "i_age" in str(error)
        else:
            raise AssertionError("missing source variable was not rejected")

    mapping = InglehartWelzelMapping(
        items=(
            TheoryItem(
                variable="i_education",
                dimension="traditional_secular",
                scores={"Low": -1.0, "High": 1.0},
            ),
        )
    )
    spec = ContextSpec(strategy="theory_structured", source_variables=("i_education",))
    text, context = build_prompt(row, "d_trust", "ntp", spec, mapping=mapping)
    assert "traditional-secular orientation: secular-rational" in text
    assert context.derived_constructs["traditional_secular"]["source_variables"] == [
        "i_education"
    ]
    report = audit_mapping(pd.DataFrame([row]), mapping, question="d_trust")
    assert report["ok"] is True
    bad = audit_mapping(pd.DataFrame([row]), mapping, question="d_happy")
    assert bad["missing_variables"] == []

    leaking = InglehartWelzelMapping(
        items=(
            TheoryItem(
                variable="d_trust",
                dimension="traditional_secular",
                scores={"Most people can be trusted": 1.0},
            ),
        )
    )
    report = audit_mapping(pd.DataFrame([row]), leaking, question="d_trust")
    assert report["ok"] is False
    assert report["target_leakage_variables"] == ["d_trust"]

    questions = pd.DataFrame(
        [
            {
                "var": "i_age",
                "type": "numerical",
                "full_q": "How old are you?",
            },
            {
                "var": "d_trust",
                "type": "categorical",
                "full_q": "Can most people be trusted?",
            },
        ]
    )
    levels = pd.DataFrame(
        [
            {
                "variable": "d_trust",
                "level": 1,
                "label": "Most people can be trusted",
            }
        ]
    )
    inventory = variable_inventory(pd.DataFrame([row]), questions, levels)
    assert {"d_trust", "i_age"}.issubset({item["variable"] for item in inventory})
    assert next(item for item in inventory if item["variable"] == "d_trust")["role"] == "target"

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        c0 = root / "C0"
        c1 = root / "C1"
        c0.mkdir()
        c1.mkdir()
        pd.DataFrame([{"metric": "overall_nEMD", "method": "NTP", "value": 0.4}]).to_csv(
            c0 / "summary_metrics.csv",
            index=False,
        )
        pd.DataFrame([{"metric": "overall_nEMD", "method": "NTP", "value": 0.3}]).to_csv(
            c1 / "summary_metrics.csv",
            index=False,
        )
        tables = stack_analysis({"C0": c0, "C1": c1})
        assert set(tables["summary_metrics"]["condition"]) == {"C0", "C1"}
        written = write_tables(tables, root / "stacked")
        assert all(path.exists() for path in written)


if __name__ == "__main__":
    main()
