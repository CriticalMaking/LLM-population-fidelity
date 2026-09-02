from __future__ import annotations

import pandas as pd
import pytest

from reports.report import EXCLUSIONS, NOTES, SECTIONS
from reports.robustness import _profile_key
from reports.series import COEFFICIENT_SERIES, MAIN_MODELS, SERIES_NAMES, resolve_series
from reports.tables import to_latex


def test_the_six_archived_series_are_registered() -> None:
    assert len(SERIES_NAMES) == 6
    assert set(MAIN_MODELS) == {
        "NTP-GPT-4T",
        "NTP-Llama-3-70B",
        "NTP-Mixtral-8x7B",
    }
    assert "FA-Llama-3-70B" in COEFFICIENT_SERIES


def test_resolve_series_defaults_expands_all_and_rejects_typos() -> None:
    assert len(resolve_series(None)) == 6
    assert len(resolve_series(["all"])) == 6
    assert [entry.name for entry in resolve_series(["FA-GPT-3"])] == ["FA-GPT-3"]
    with pytest.raises(ValueError, match="unknown series"):
        resolve_series(["NTP-Claude"])


def test_upstream_artifacts_without_a_generator_are_named() -> None:
    assert set(EXCLUSIONS) == {
        "Figure 1",
        "Figure 5",
        "Table 1",
        "Figure S3",
        "Figure S11",
        "Table S3",
    }
    assert all(reason for reason in EXCLUSIONS.values())


def test_substitutions_for_the_r_packages_are_recorded() -> None:
    assert "ranger" in NOTES["Table S8"]
    assert "multinom" in NOTES["Figure S16"]


def test_sections_cover_the_documented_driver_commands() -> None:
    assert set(SECTIONS) == {
        "all",
        "main",
        "appendix",
        "robustness",
        "tables",
        "figures",
        "smoke",
    }


def test_latex_escapes_the_characters_that_would_break_a_build() -> None:
    frame = pd.DataFrame([{"series": "NTP_Mixtral & co", "value": 0.5}])
    rendered = to_latex(frame, "A caption", "tab:example")
    assert r"NTP\_Mixtral \& co" in rendered
    assert "\\begin{tabular}{ll}" in rendered
    assert "\\toprule" in rendered and "\\bottomrule" in rendered
    assert "0.500" in rendered


def test_latex_of_an_empty_table_is_a_comment_not_a_broken_environment() -> None:
    assert to_latex(pd.DataFrame(), "x", "tab:empty").startswith("%")


def test_robustness_profile_key_strips_only_the_language_prefix() -> None:
    assert (
        _profile_key("en§United_States§1995§18§Female§High§Student§Single")
        == "United_States§1995§18§Female§High§Student§Single"
    )
    assert (
        _profile_key("Australia§1995§17§Female§Middle§Working§Single")
        == "Australia§1995§17§Female§Middle§Working§Single"
    )
