from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau, pearsonr, spearmanr

from culture import fidelity
from culture import fidelity_sensitivity as sensitivity
from culture.population import RunSource
from machine_bias_reproduction.config import OUTPUTS_ROOT
from machine_bias_reproduction.data import Coverage, PreparedData, respondent_counts
from machine_bias_reproduction.metrics import nemd, pairwise_nemd
from machine_bias_reproduction.questions import QUESTIONS, resolve_question

CELLS = (
    "Germany 2018 Female 45-54 Middle Working Married",
    "Germany 1997 Male 25-34 High Student Single",
    "Mexico 2005 Female 55-64 Low Retired Widowed",
    "Russia 2017 Male <25 High NA Divorced or separated",
    "United States 1995 Female 75+ Middle Homemaker Cohabiting",
)

SURVEY = np.array(
    [
        [0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4],
        [0.25, 0.25, 0.25, 0.25],
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.7],
    ]
)

MODEL = np.array(
    [
        [0.35, 0.35, 0.2, 0.1],
        [0.15, 0.25, 0.3, 0.3],
        [0.2, 0.3, 0.3, 0.2],
        [0.6, 0.2, 0.1, 0.1],
        [0.2, 0.1, 0.2, 0.5],
    ]
)

BASE = np.array(
    [
        [0.3, 0.3, 0.3, 0.1],
        [0.25, 0.25, 0.25, 0.25],
        [0.2, 0.3, 0.3, 0.2],
        [0.4, 0.3, 0.2, 0.1],
        [0.3, 0.2, 0.2, 0.3],
    ]
)

# The model's rows dealt to the wrong cells, so its pairwise distances rank against the
# survey's: rho is about -0.5, and the default clips it to a structure score of zero.
ANTI = MODEL[[3, 0, 4, 1, 2]]

# The survey's own pattern stretched away from uniform, so the model spreads further than
# the survey (A is about 1.33) while ranking its cells almost exactly like the survey.
AMPLIFIED = np.array(
    [
        [0.55, 0.3, 0.1, 0.05],
        [0.05, 0.1, 0.3, 0.55],
        [0.25, 0.25, 0.25, 0.25],
        [0.85, 0.05, 0.05, 0.05],
        [0.05, 0.05, 0.05, 0.85],
    ]
)

FLAT = np.tile(SURVEY[0], (len(SURVEY), 1))

ONES = np.ones(len(SURVEY))

WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 100.0])

BY_NAME = {variant.name: variant for variant in sensitivity.VARIANTS}

needs_shipped = pytest.mark.skipif(
    not fidelity.GROUPS_TABLE.is_file()
    or not sensitivity.SENSITIVITY_TABLE.is_file()
    or not (OUTPUTS_ROOT / "archived" / "d_happy").is_dir(),
    reason="the shipped fidelity tables or the archived Mixtral run are not staged",
)


def _approx(value: Any, expected: Any) -> None:
    assert value == pytest.approx(expected, abs=1e-15, nan_ok=True)


def test_the_default_variant_reproduces_group_fidelity_to_the_last_digit() -> None:
    for model in (MODEL, FLAT):
        reading = sensitivity.readings(SURVEY, model, ONES)
        row = sensitivity.variant_row(reading, sensitivity.DEFAULT)
        expected = fidelity.group_fidelity(SURVEY, model)

        assert reading["model_flat"] == expected["model_flat"]
        for ours, theirs in sensitivity.DEFAULT_COLUMNS.items():
            _approx(reading[ours] if ours in reading else row[ours], expected[theirs])


def test_the_pairwise_vectors_are_computed_once_per_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def counted(distributions: np.ndarray) -> Any:
        calls.append(len(distributions))
        return pairwise_nemd(distributions)

    monkeypatch.setattr(sensitivity, "pairwise_nemd", counted)
    reading = sensitivity.readings(SURVEY, MODEL, ONES)

    assert calls == [5, 5]
    assert set(sensitivity.READINGS) <= set(reading)
    assert reading["n_pairs"] == 10
    assert reading["n_respondents"] == 5.0


def test_mean_dispersion_reads_the_mean_of_the_same_pairs_and_the_ratio_follows() -> None:
    reading = sensitivity.readings(SURVEY, MODEL, ONES)
    survey, model = pairwise_nemd(SURVEY), pairwise_nemd(MODEL)

    assert reading["d_wvs_mean"] == pytest.approx(survey.mean())
    assert reading["d_llm_mean"] == pytest.approx(model.mean())
    assert reading["d_wvs_median"] == pytest.approx(np.median(survey))

    row = sensitivity.variant_row(reading, BY_NAME["mean_dispersion"])
    default = sensitivity.variant_row(reading, sensitivity.DEFAULT)

    assert row["adaptability_ratio"] == pytest.approx(model.mean() / survey.mean())
    assert row["score_dispersion"] == pytest.approx(
        fidelity.dispersion_score(row["adaptability_ratio"])
    )
    assert row["adaptability_ratio"] != default["adaptability_ratio"]
    assert row["score_structure"] == default["score_structure"]
    assert row["score_accuracy"] == default["score_accuracy"]


def test_one_sided_adaptability_forgives_amplification_only() -> None:
    assert sensitivity.one_sided_score(2.0) == 1.0
    assert fidelity.dispersion_score(2.0) == pytest.approx(0.5)
    assert sensitivity.one_sided_score(0.5) == pytest.approx(0.5)
    assert sensitivity.one_sided_score(0.0) == 0.0
    assert np.isnan(sensitivity.one_sided_score(float("nan")))

    compressed = sensitivity.readings(SURVEY, MODEL, ONES)
    amplified = sensitivity.readings(SURVEY, AMPLIFIED, ONES)
    one_sided = BY_NAME["one_sided_adaptability"]

    assert compressed["d_llm_median"] < compressed["d_wvs_median"]
    assert amplified["d_llm_median"] > amplified["d_wvs_median"]
    assert sensitivity.variant_row(compressed, one_sided) == sensitivity.variant_row(
        compressed, sensitivity.DEFAULT
    )
    assert sensitivity.variant_row(amplified, one_sided)["score_dispersion"] == 1.0
    assert sensitivity.variant_row(amplified, sensitivity.DEFAULT)["score_dispersion"] < 1.0


def test_pearson_and_kendall_match_scipy_on_the_condensed_vectors() -> None:
    reading = sensitivity.readings(SURVEY, MODEL, ONES)
    survey, model = pairwise_nemd(SURVEY), pairwise_nemd(MODEL)

    assert reading["rho_spearman"] == pytest.approx(spearmanr(survey, model).statistic)
    assert reading["r_pearson"] == pytest.approx(pearsonr(survey, model).statistic)
    assert reading["tau_kendall"] == pytest.approx(kendalltau(survey, model).statistic)

    pearson = sensitivity.variant_row(reading, BY_NAME["pearson_structure"])
    kendall = sensitivity.variant_row(reading, BY_NAME["kendall_structure"])

    assert pearson["structure_coefficient"] == reading["r_pearson"]
    assert pearson["score_structure"] == fidelity.structure_score(reading["r_pearson"])
    assert kendall["structure_coefficient"] == reading["tau_kendall"]

    two_cells = sensitivity.readings(SURVEY[:2], MODEL[:2], ONES[:2])
    flat = sensitivity.readings(SURVEY, FLAT, ONES)
    for column in sensitivity.COEFFICIENTS.values():
        assert np.isnan(two_cells[column])
        assert np.isnan(flat[column])
    assert two_cells["n_pairs"] == 1
    assert flat["model_flat"]


def test_unclipped_structure_keeps_the_sign_and_agrees_with_the_default_when_rho_is_positive() -> (
    None
):
    unclipped = BY_NAME["unclipped_structure"]
    anti = sensitivity.readings(SURVEY, ANTI, ONES)
    default = sensitivity.variant_row(anti, sensitivity.DEFAULT)
    row = sensitivity.variant_row(anti, unclipped)

    assert anti["rho_spearman"] < 0.0
    assert default["score_structure"] == 0.0
    assert default["pfs"] == 0.0
    assert row["score_structure"] == anti["rho_spearman"]
    assert row["pfs"] < 0.0
    assert row["pfs"] == pytest.approx(
        -(
            (row["score_accuracy"] * row["score_dispersion"] * abs(row["score_structure"]))
            ** (1 / 3)
        )
    )

    positive = sensitivity.readings(SURVEY, MODEL, ONES)
    for key, value in sensitivity.variant_row(positive, sensitivity.DEFAULT).items():
        _approx(sensitivity.variant_row(positive, unclipped)[key], value)

    assert sensitivity.signed_geometric_mean([0.9, 0.9, 0.0]) == 0.0
    assert sensitivity.signed_geometric_mean([0.0, 0.9, -0.5]) == 0.0
    assert sensitivity.signed_geometric_mean([0.5, 0.5, -0.5]) == pytest.approx(-0.5)
    assert sensitivity.signed_geometric_mean([0.5, 0.5, 0.5]) == pytest.approx(0.5)
    assert np.isnan(sensitivity.signed_geometric_mean([0.9, float("nan"), 0.5]))


def test_directions_count_changes_beyond_the_tolerance_and_signs_treat_the_rest_as_zero() -> None:
    assert sensitivity._directions(np.array([2e-12, -2e-12, 5e-13, np.nan])) == (1, 1, 1)
    assert sensitivity._directions(np.array([], dtype=np.float64)) == (0, 0, 0)
    assert sensitivity._sign(np.array([5e-13, -3.0, 2.0, -5e-13])).tolist() == [0.0, -1.0, 1.0, 0.0]


def _identity(question: str, arm: str | None, series: str, mode: str = "ntp") -> dict[str, Any]:
    key = None if arm is None else "gemma4_31b"
    return {
        "question": question,
        "question_label": question,
        "model_key": key,
        "model_label": series,
        "arm": arm,
        "series": series,
        "source": "archived" if key is None else f"culture/{key}/{arm}",
        "mode": mode,
        "replicate": 1,
    }


QUESTIONS_USED: tuple[str, ...] = ("d_happy", "d_trust")

LEVELS: tuple[tuple[str, str, slice], ...] = (
    (fidelity.POPULATION, fidelity.EVERY_CELL, slice(None)),
    ("country", "Germany", slice(0, 3)),
    ("country", "Mexico", slice(2, 5)),
)

# Each series answers differently under the two modes, so the NTP and FA rankings of one
# question disagree and the as-released rows split between compression and amplification.
SERIES: tuple[tuple[str | None, str, dict[str, np.ndarray]], ...] = (
    ("base", "Gemma (as released)", {"ntp": BASE, "fa": AMPLIFIED}),
    ("german", "Gemma (german)", {"ntp": MODEL, "fa": MODEL}),
    ("spanish-mx", "Gemma (spanish-mx)", {"ntp": ANTI, "fa": ANTI}),
    (None, "Mixtral archived", {"ntp": MODEL, "fa": BASE}),
)

# The German variant improves on happiness and answers exactly like its base on trust; the
# Mexican-Spanish variant is anti-correlated on both.
TUNED: dict[tuple[str, str], np.ndarray] = {
    ("german", "d_happy"): MODEL,
    ("german", "d_trust"): BASE,
    ("spanish-mx", "d_happy"): ANTI,
    ("spanish-mx", "d_trust"): ANTI,
}

BASE_SOURCE = "culture/gemma4_31b/base"


def readings_frame() -> pd.DataFrame:
    rows = []
    for question in QUESTIONS_USED:
        for arm, series, answers in SERIES:
            for mode in ("ntp", "fa"):
                for group, level, cells in LEVELS:
                    reading = sensitivity.readings(SURVEY[cells], answers[mode][cells], ONES[cells])
                    identity = _identity(question, arm, series, mode)
                    rows.append({**identity, "group": group, "level": level, **reading})
    return pd.DataFrame(rows)


def groups_frame() -> pd.DataFrame:
    rows = []
    for question in QUESTIONS_USED:
        for arm, series, answers in SERIES:
            for mode in ("ntp", "fa"):
                for group, level, cells in LEVELS:
                    scored = fidelity.group_fidelity(SURVEY[cells], answers[mode][cells])
                    identity = _identity(question, arm, series, mode)
                    rows.append({**identity, "group": group, "level": level, **scored})
    frame = pd.DataFrame(rows)
    frame["binding_term"] = fidelity.binding_term(frame)
    return frame


def paired_frame(scorer: Any) -> pd.DataFrame:
    rows = []
    for question in QUESTIONS_USED:
        for arm, series, _ in SERIES[1:3]:
            assert arm is not None
            tuned = TUNED[(arm, question)]
            for mode in ("ntp", "fa"):
                for group, level, cells in LEVELS:
                    if scorer is sensitivity.paired_readings:
                        scored = scorer(SURVEY[cells], tuned[cells], BASE[cells], ONES[cells])
                    else:
                        scored = scorer(SURVEY[cells], tuned[cells], BASE[cells])
                    identity = _identity(question, arm, series, mode)
                    rows.append(
                        {
                            **identity,
                            "base_source": BASE_SOURCE,
                            "group": group,
                            "level": level,
                            **scored,
                        }
                    )
    frame = pd.DataFrame(rows)
    if scorer is fidelity.paired_fidelity:
        frame["binding_term"] = fidelity.binding_term(frame)
        frame["base_binding_term"] = fidelity.binding_term(frame, "base_")
    return frame


def _pooled_block(scored: pd.DataFrame, name: str) -> pd.DataFrame:
    block = scored[scored["variant"].eq(name) & scored["group"].eq(fidelity.POPULATION)]
    return block.set_index(["question", "series", "mode"])


def test_the_arithmetic_mean_keeps_the_binding_term_and_lifts_a_zero() -> None:
    assert sensitivity.arithmetic_mean([0.9, 0.6, 0.0]) == pytest.approx(0.5)
    assert np.isnan(sensitivity.arithmetic_mean([0.9, float("nan")]))

    scored = sensitivity.score_variants(
        readings_frame(), (sensitivity.DEFAULT, BY_NAME["arithmetic_mean"])
    )
    default = scored[scored["variant"].eq("default")].reset_index(drop=True)
    arithmetic = scored[scored["variant"].eq("arithmetic_mean")].reset_index(drop=True)

    assert arithmetic["binding_term"].tolist() == default["binding_term"].tolist()
    zeros = default["pfs"].eq(0.0)
    assert zeros.any()
    assert arithmetic["pfs"][zeros].gt(0.0).all()
    assert arithmetic["variant_label"].iat[0] == "Arithmetic mean"
    assert sensitivity.score_variants(readings_frame().iloc[0:0]).empty


def test_respondent_weights_move_only_e_and_c_and_unit_weights_reproduce_the_default() -> None:
    unweighted = sensitivity.readings(SURVEY, MODEL, ONES)
    assert unweighted["e_mean_nemd_weighted"] == pytest.approx(unweighted["e_mean_nemd"])
    assert unweighted["c_center_nemd_weighted"] == pytest.approx(unweighted["c_center_nemd"])
    for key, value in sensitivity.variant_row(unweighted, sensitivity.DEFAULT).items():
        _approx(sensitivity.variant_row(unweighted, BY_NAME["weighted_cells"])[key], value)

    weighted = sensitivity.readings(SURVEY, MODEL, WEIGHTS)
    error = np.asarray(nemd(SURVEY, MODEL))

    assert weighted["n_respondents"] == 104.0
    assert weighted["e_mean_nemd_weighted"] == pytest.approx(np.average(error, weights=WEIGHTS))
    assert weighted["c_center_nemd_weighted"] == pytest.approx(
        nemd(
            np.average(SURVEY, axis=0, weights=WEIGHTS), np.average(MODEL, axis=0, weights=WEIGHTS)
        )
    )
    for key in sensitivity.READINGS:
        if not key.endswith("_weighted") and key != "n_respondents":
            _approx(weighted[key], unweighted[key])
    row = sensitivity.variant_row(weighted, BY_NAME["weighted_cells"])
    default = sensitivity.variant_row(weighted, sensitivity.DEFAULT)
    assert row["score_accuracy"] != default["score_accuracy"]
    assert row["score_center"] != default["score_center"]
    assert row["score_dispersion"] == default["score_dispersion"]
    assert row["score_structure"] == default["score_structure"]


def test_grouped_rows_mask_the_weights_with_their_cells() -> None:
    facets = fidelity.cell_facets(CELLS).reindex(list(CELLS))
    rows = fidelity.grouped_scores(
        _identity("d_happy", "german", "Gemma (german)"),
        facets,
        sensitivity.readings,
        SURVEY,
        MODEL,
        WEIGHTS,
    )
    germany = next(r for r in rows if r["group"] == "country" and r["level"] == "Germany")
    cohabiting = next(r for r in rows if r["level"] == "Cohabiting")
    expected = sensitivity.readings(SURVEY[:2], MODEL[:2], WEIGHTS[:2])

    assert germany["n_cells"] == 2
    assert germany["n_respondents"] == 2.0
    for key in sensitivity.READINGS:
        _approx(germany[key], expected[key])
    assert cohabiting["n_respondents"] == 100.0
    assert rows[0]["n_respondents"] == 104.0


def _prepared(names: tuple[str, ...], answers: list[Any], cells: list[str]) -> PreparedData:
    question = QUESTIONS["d_happy"]
    index = pd.Index(names, name="name")
    columns = list(question.answer_columns)
    props = pd.DataFrame(SURVEY[: len(names)], index=index, columns=columns)
    return PreparedData(
        question=question,
        wvs=pd.DataFrame({"id": range(len(answers)), question.var: answers}),
        subpops=pd.DataFrame({"subpop": cells}),
        ntp_raw=None,
        fa_raw=pd.DataFrame(),
        names=index,
        wvs_props=props,
        ntp_props=None,
        fa_props=props,
        social_predictors=pd.DataFrame(index=index),
        coverage=Coverage(0, 0, 0, 0, len(names), len(names)),
    )


def test_respondent_counts_count_valid_survey_answers_per_retained_cell() -> None:
    labels = QUESTIONS["d_happy"].wvs_labels
    answers = [labels[0], labels[1], None, labels[0], labels[0], labels[1]]
    cells = ["A", "A", "A", "B", "B", "B"]

    counts = respondent_counts(_prepared(("A", "B"), answers, cells))
    assert counts.tolist() == [2.0, 3.0]
    assert counts.index.tolist() == ["A", "B"]

    weights = sensitivity.run_weights(_prepared(("B", "A"), answers, cells), pd.Index(["B", "A"]))
    assert weights.tolist() == [3.0, 2.0]

    with pytest.raises(ValueError, match="valid survey answer"):
        respondent_counts(_prepared(("A", "C"), answers, cells))
    with pytest.raises(ValueError, match="positive respondent count"):
        sensitivity.run_weights(_prepared(("A", "B"), answers, cells), pd.Index(["A", "C"]))

    empty = PreparedData(
        question=QUESTIONS["d_happy"],
        wvs=pd.DataFrame(),
        subpops=pd.DataFrame(),
        ntp_raw=None,
        fa_raw=pd.DataFrame(),
        names=pd.Index(["A", "B"]),
        wvs_props=pd.DataFrame(),
        ntp_props=None,
        fa_props=pd.DataFrame(),
        social_predictors=pd.DataFrame(),
        coverage=Coverage(0, 0, 0, 0, 2, 2),
    )
    assert sensitivity.run_weights(empty, pd.Index(["A", "B"])).tolist() == [1.0, 1.0]


def test_a_paired_row_differences_each_variant_against_its_own_base() -> None:
    frame = paired_frame(sensitivity.paired_readings)
    scored = sensitivity.score_paired(frame)
    expected = fidelity.paired_fidelity(SURVEY, MODEL, BASE)

    assert len(scored) == len(frame) * len(sensitivity.VARIANTS)
    for metric in (*sensitivity.DIAGNOSTICS, *sensitivity.SCORES):
        delta = scored[metric] - scored[f"base_{metric}"]
        assert np.allclose(scored[fidelity.delta_column(metric)], delta, equal_nan=True)
    assert "base_binding_term" in scored.columns

    default = scored[
        scored["variant"].eq("default")
        & scored["series"].eq("Gemma (german)")
        & scored["question"].eq("d_happy")
        & scored["group"].eq(fidelity.POPULATION)
    ].iloc[0]
    for column in ("pfs", "base_pfs", "delta_pfs_vs_base", "score_center", "base_score_center"):
        _approx(default[column], expected[column])
    assert default["base_e_mean_nemd"] == pytest.approx(expected["base_e_mean_nemd"])
    assert sensitivity.score_paired(frame.iloc[0:0]).empty


def test_the_summary_levels_binding_and_rankings_are_read_off_the_scored_pooled_rows() -> None:
    frame = readings_frame()
    summary = sensitivity.sensitivity_summary(frame, pd.DataFrame()).set_index("variant")
    scored = sensitivity.score_variants(frame)
    default_pooled = _pooled_block(scored, "default")
    default = summary.loc["default"]

    assert summary.index.tolist() == [variant.name for variant in sensitivity.VARIANTS]
    assert summary["n_pooled"].eq(16).all()
    assert summary["groups_n"].eq(32).all()
    for column in sensitivity.DESIGN:
        assert column in summary.columns
    assert "german_n" not in summary.columns

    assert default["accuracy_mean"] == pytest.approx(default_pooled["score_accuracy"].mean())
    assert default["accuracy_median"] == pytest.approx(default_pooled["score_accuracy"].median())
    assert default["accuracy_mean"] != default["accuracy_median"]
    assert default["adaptability_mean"] == pytest.approx(default_pooled["score_dispersion"].mean())
    assert default["adaptability_median"] == pytest.approx(
        default_pooled["score_dispersion"].median()
    )
    assert default["structure_mean"] == pytest.approx(default_pooled["score_structure"].mean())
    assert default["structure_median"] == pytest.approx(default_pooled["score_structure"].median())
    assert default["coefficient_mean"] == pytest.approx(
        default_pooled["structure_coefficient"].mean()
    )
    assert default["coefficient_median"] == pytest.approx(
        default_pooled["structure_coefficient"].median()
    )
    assert default["pfs_mean"] == pytest.approx(default_pooled["pfs"].mean())
    assert default["pfs_median"] == pytest.approx(default_pooled["pfs"].median())
    assert default["pfs_min"] == 0.0
    assert default["pfs_nonpositive"] == 4

    counts = default_pooled["binding_term"].value_counts()
    for term in ("accuracy", "adaptability", "structure"):
        assert default[f"binding_{term}"] == counts.get(term, 0)
    assert default["binding_agreement_vs_default"] == 1.0
    one_sided = summary.loc["one_sided_adaptability"]
    assert one_sided["binding_accuracy"] > default["binding_accuracy"]
    assert one_sided["binding_agreement_vs_default"] < 1.0

    assert default["rank_vs_default"] == pytest.approx(1.0)
    for question in QUESTIONS_USED:
        assert default[f"rank_vs_default_{question}"] == pytest.approx(1.0)
    combined_pooled = _pooled_block(scored, "combined").reindex(default_pooled.index)
    expected_rank = spearmanr(combined_pooled["pfs"], default_pooled["pfs"]).statistic
    assert expected_rank < 1.0
    assert summary.loc["combined", "rank_vs_default"] == pytest.approx(expected_rank)
    assert default["rank_vs_structure"] == pytest.approx(
        spearmanr(default_pooled["pfs"], default_pooled["score_structure"]).statistic
    )

    agreements = []
    for question in QUESTIONS_USED:
        wide = default_pooled.xs(question, level="question")["pfs"].unstack("mode")
        agreements.append(spearmanr(wide["ntp"], wide["fa"]).statistic)
    assert min(agreements) < 1.0
    assert default["mode_agreement_min"] == pytest.approx(min(agreements))
    assert default["mode_agreement_max"] == pytest.approx(max(agreements))
    assert summary.loc["combined", "mode_agreement_min"] != default["mode_agreement_min"]

    released = default_pooled[default_pooled["arm"].eq("base")]
    assert default["released_n"] == 4
    assert default["released_compressed"] == int(released["adaptability_ratio"].lt(1.0).sum()) == 2
    assert default["released_amplified"] == int(released["adaptability_ratio"].gt(1.0).sum()) == 2

    levels = scored[scored["variant"].eq("default") & ~scored["group"].eq(fidelity.POPULATION)]
    assert default["groups_binding_structure"] == int(levels["binding_term"].eq("structure").sum())
    assert default["groups_negative_coefficient"] == int(
        levels["structure_coefficient"].lt(0.0).sum()
    )
    assert default["groups_negative_coefficient"] == 8
    assert default["groups_coefficient_below_minus_010"] == int(
        levels["structure_coefficient"].lt(-0.10).sum()
    )

    unclipped = summary.loc["unclipped_structure"]
    assert unclipped["pfs_min"] < 0.0
    assert unclipped["pfs_nonpositive"] == 4
    arithmetic = summary.loc["arithmetic_mean"]
    assert arithmetic["pfs_nonpositive"] == 0
    assert arithmetic["binding_agreement_vs_default"] == 1.0
    assert arithmetic["pfs_mean"] > default["pfs_mean"]


def test_the_summary_reads_each_finetuning_arm_pooled_and_inside_its_own_home_country() -> None:
    frame = readings_frame()
    paired = paired_frame(sensitivity.paired_readings)
    summary = sensitivity.sensitivity_summary(frame, paired).set_index("variant")
    default = summary.loc["default"]

    pooled = fidelity.paired_fidelity(SURVEY, MODEL, BASE)
    germany = fidelity.paired_fidelity(SURVEY[:3], MODEL[:3], BASE[:3])
    mexico = fidelity.paired_fidelity(SURVEY[2:], ANTI[2:], BASE[2:])
    assert pooled["delta_pfs_vs_base"] > 0.0 > mexico["delta_pfs_vs_base"]

    # German rows: two improvements on happiness and two exact ties on trust.
    assert default["german_n"] == 4
    assert default["german_delta_pfs_mean"] == pytest.approx(pooled["delta_pfs_vs_base"] / 2)
    assert (default["german_improve"], default["german_decline"], default["german_unchanged"]) == (
        2,
        0,
        2,
    )
    assert default["german_delta_center_mean"] == pytest.approx(
        pooled["delta_score_center_vs_base"] / 2
    )
    assert default["german_center_improve"] == 2
    assert default["german_home_n"] == 4
    assert default["german_home_delta_pfs_mean"] == pytest.approx(germany["delta_pfs_vs_base"] / 2)
    assert default["german_home_delta_pfs_mean"] != pytest.approx(default["german_delta_pfs_mean"])
    assert (default["german_home_improve"], default["german_home_decline"]) == (2, 0)
    assert default["german_home_center_improve"] == 0

    # Mexican-Spanish rows: anti-correlated everywhere, so every comparison declines.
    assert default["spanish_mx_n"] == 4
    assert (default["spanish_mx_improve"], default["spanish_mx_decline"]) == (0, 4)
    assert default["spanish_mx_delta_pfs_mean"] < 0.0
    assert default["spanish_mx_home_n"] == 4
    assert default["spanish_mx_home_delta_pfs_mean"] == pytest.approx(mexico["delta_pfs_vs_base"])
    assert default["spanish_mx_home_delta_center_mean"] == pytest.approx(
        mexico["delta_score_center_vs_base"]
    )
    assert default["spanish_mx_home_delta_pfs_mean"] != pytest.approx(
        default["german_home_delta_pfs_mean"]
    )
    assert default["german_sign_agreement_vs_default"] == 1.0
    assert default["spanish_mx_sign_agreement_vs_default"] == 1.0
    assert summary["german_delta_pfs_mean"].gt(0.0).all()
    assert summary["spanish_mx_decline"].eq(4).all()

    # The same comparison on each condition's own cells is reported beside the paired one.
    def own(cells: slice, tuned: np.ndarray) -> float:
        ntp = fidelity.group_fidelity(SURVEY[cells], tuned[cells])["pfs"]
        fa = fidelity.group_fidelity(SURVEY[cells], tuned[cells])["pfs"]
        base_ntp = fidelity.group_fidelity(SURVEY[cells], BASE[cells])["pfs"]
        base_fa = fidelity.group_fidelity(SURVEY[cells], AMPLIFIED[cells])["pfs"]
        return float(((ntp - base_ntp) + (fa - base_fa)) / 2)

    assert default["german_delta_pfs_mean_own_cells"] == pytest.approx(own(slice(None), MODEL))
    assert default["german_home_delta_pfs_mean_own_cells"] == pytest.approx(own(slice(0, 3), MODEL))
    assert default["german_delta_pfs_mean_own_cells"] != pytest.approx(
        default["german_delta_pfs_mean"]
    )
    assert default["spanish_mx_home_delta_pfs_mean_own_cells"] == pytest.approx(
        own(slice(2, 5), ANTI)
    )

    # Without country rows the home columns are empty rather than wrong.
    pooled_only = paired[paired["group"].eq(fidelity.POPULATION)]
    bare = sensitivity.sensitivity_summary(frame, pooled_only).set_index("variant").loc["default"]
    assert bare["german_home_n"] == 0
    assert np.isnan(bare["german_home_delta_pfs_mean"])
    assert bare["german_n"] == 4


def test_a_repeated_replicate_counts_once_in_the_summary() -> None:
    frame = readings_frame()
    paired = paired_frame(sensitivity.paired_readings)
    again = frame.copy()
    again["replicate"] = 2
    again["source"] = again["source"] + "/rep2"
    german = again["arm"].eq("german")
    again["e_mean_nemd"] = again["e_mean_nemd"] + np.where(german, 0.9, 0.01)
    paired_again = paired.copy()
    paired_again["replicate"] = 2
    paired_again["source"] = paired_again["source"] + "/rep2"
    paired_again["base_source"] = paired_again["base_source"] + "/rep2"
    doubled = pd.concat([frame, again], ignore_index=True)
    paired_doubled = pd.concat([paired, paired_again], ignore_index=True)

    once = sensitivity.sensitivity_summary(frame, paired).set_index("variant")
    twice = sensitivity.sensitivity_summary(doubled, paired_doubled).set_index("variant")

    assert twice["n_pooled"].eq(16).all()
    assert twice["groups_n"].eq(32).all()
    assert twice["german_n"].eq(4).all()
    assert twice["spanish_mx_n"].eq(4).all()
    assert twice["german_home_n"].eq(4).all()
    binding_once = once["binding_accuracy"].to_dict()["default"]
    binding_twice = twice["binding_accuracy"].to_dict()["default"]
    assert binding_twice > binding_once
    assert twice.loc["default", "binding_agreement_vs_default"] == 1.0
    assert twice.loc["default", "german_delta_pfs_mean"] == pytest.approx(
        once.loc["default", "german_delta_pfs_mean"]
    )

    # Replicates that were not averaged are refused rather than double counted.
    with pytest.raises(ValueError, match="replicates not averaged"):
        sensitivity._finetuning(
            sensitivity.score_paired(paired_doubled),
            sensitivity.score_paired(paired_doubled),
            sensitivity.score_variants(frame),
        )


def test_check_sensitivity_refuses_a_default_row_that_drifted_or_a_survey_that_moved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sensitivity, "STRUCTURE_TABLE", tmp_path / "absent.csv")
    frame = readings_frame()
    paired = paired_frame(sensitivity.paired_readings)
    groups = groups_frame()
    reference = paired_frame(fidelity.paired_fidelity)
    summary = sensitivity.sensitivity_summary(frame, paired)

    sensitivity.check_sensitivity(frame, paired, groups, reference, summary)

    drifted = groups.copy()
    drifted["pfs"] = drifted["pfs"] + np.where(drifted.index == 0, 0.01, 0.0)
    with pytest.raises(AssertionError, match="default"):
        sensitivity.check_sensitivity(frame, paired, drifted, reference, summary)

    moved = frame.copy()
    moved["d_wvs_mean"] = moved["d_wvs_mean"] + np.where(moved.index == 0, 0.01, 0.0)
    with pytest.raises(AssertionError, match="moved"):
        sensitivity.check_sensitivity(moved, paired, groups, reference, summary)

    unpeopled = frame.copy()
    unpeopled["n_respondents"] = np.where(unpeopled.index == 0, 0.5, unpeopled["n_respondents"])
    with pytest.raises(AssertionError, match="no respondents"):
        sensitivity.check_sensitivity(unpeopled, paired, groups, reference, summary)

    scored = sensitivity.score_variants(frame)
    out_of_bounds = scored.copy()
    out_of_bounds["pfs"] = np.where(out_of_bounds.index == 0, 1.5, out_of_bounds["pfs"])
    with pytest.raises(AssertionError, match="out of bounds"):
        sensitivity._check_bounds(out_of_bounds, sensitivity.VARIANTS)
    negative = scored.copy()
    negative["pfs"] = np.where(negative.index == 0, -0.1, negative["pfs"])
    with pytest.raises(AssertionError, match="default: pfs out of bounds"):
        sensitivity._check_bounds(negative, sensitivity.VARIANTS)
    signed = scored[scored["variant"].eq("unclipped_structure")].copy()
    signed["pfs"] = np.where(signed.index == signed.index[0], -0.1, signed["pfs"])
    sensitivity._check_bounds(signed, (BY_NAME["unclipped_structure"],))

    pooled = frame[frame["group"].eq(fidelity.POPULATION)]
    stored = pooled[["question", "source", "mode", "r_pearson"]].rename(
        columns={"r_pearson": "pearson_structure"}
    )
    stored.loc[stored.index[0], "pearson_structure"] = np.nan
    stored.to_csv(tmp_path / "structure.csv", index=False)
    monkeypatch.setattr(sensitivity, "STRUCTURE_TABLE", tmp_path / "structure.csv")
    sensitivity.check_sensitivity(frame, paired, groups, reference, summary)

    stored["pearson_structure"] = 0.123
    stored.to_csv(tmp_path / "structure.csv", index=False)
    with pytest.raises(AssertionError, match=r"Pearson disagrees .* d_happy/"):
        sensitivity.check_sensitivity(frame, paired, groups, reference, summary)


def test_the_variant_set_names_the_default_first_and_uniquely() -> None:
    names = [variant.name for variant in sensitivity.VARIANTS]
    labels = [variant.label for variant in sensitivity.VARIANTS]

    assert sensitivity.VARIANTS[0] is sensitivity.DEFAULT
    assert len(set(names)) == len(names)
    assert len(set(labels)) == len(labels)
    assert not sensitivity.DEFAULT.signed
    assert BY_NAME["combined"].signed
    assert sensitivity.HOME == {"german": "Germany", "spanish-mx": "Mexico"}
    for field in sensitivity.DESIGN:
        assert getattr(sensitivity.DEFAULT, field) == getattr(sensitivity.Variant("x", "x"), field)


@needs_shipped
def test_the_default_rows_reproduce_the_shipped_tables() -> None:
    question = resolve_question("d_happy")
    run = RunSource("Mixtral archived", "archived", None, "Mixtral archived", None, 1)

    frame, paired = sensitivity.build_sensitivity([question], runs=[run])
    scored = sensitivity.score_variants(frame, (sensitivity.DEFAULT,))
    shipped = pd.read_csv(fidelity.GROUPS_TABLE)
    reference = shipped[
        shipped["series"].eq("Mixtral archived") & shipped["question"].eq("d_happy")
    ].set_index(["mode", "group", "level"])
    ours = scored.set_index(["mode", "group", "level"]).reindex(reference.index)

    assert paired.empty
    assert len(ours) == len(reference) > 0
    for column in ("pfs", "score_center", "score_structure", "adaptability_ratio", "n_cells"):
        assert np.allclose(ours[column], reference[column], equal_nan=True)
    assert np.allclose(ours["structure_coefficient"], reference["rho_structure"], equal_nan=True)
    assert np.allclose(ours["d_wvs_median"], reference["d_wvs"])

    summary = pd.read_csv(sensitivity.SENSITIVITY_TABLE).set_index("variant")
    default = summary.loc["default"]
    assert default["n_pooled"] == 164
    assert default["binding_structure"] == 159
    assert default["german_delta_pfs_mean"] == pytest.approx(-0.0243, abs=5e-4)
    assert default["spanish_mx_home_delta_center_mean"] == pytest.approx(0.0383, abs=5e-4)
