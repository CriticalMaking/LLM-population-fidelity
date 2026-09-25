from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, spearmanr

from culture import archived_reference, fidelity, fidelity_plates
from culture.population import RunSource
from machine_bias_reproduction.data import Coverage, PreparedData
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


def test_a_cell_is_grouped_by_its_label_not_by_its_respondents() -> None:
    facets = fidelity.cell_facets(CELLS)

    assert list(facets.columns) == list(fidelity.FAMILIES)
    assert facets.loc[CELLS[0], "country"] == "Germany"
    assert facets.loc[CELLS[0], "wave"] == "2017-2018"
    assert facets.loc[CELLS[1], "wave"] == "1995-1997"
    assert facets.loc[CELLS[4], "marital_status"] == "Cohabiting"


def test_an_na_token_sits_out_its_own_family_and_counts_in_every_other() -> None:
    facets = fidelity.cell_facets(CELLS)

    assert facets["employment"].isna().loc[CELLS[3]]
    assert facets.loc[CELLS[3], "country"] == "Russia"
    assert fidelity.levels("employment", facets["employment"]) == [
        "Working",
        "Student",
        "Homemaker",
        "Retired",
    ]


def test_an_unparsable_label_is_refused_rather_than_grouped_wrongly() -> None:
    with pytest.raises(ValueError, match="unparsable"):
        fidelity.cell_facets(["Germany 2018 Female Middle Working Married"])


def test_dispersion_penalises_compression_and_over_spread_alike() -> None:
    assert fidelity.dispersion_score(0.5) == pytest.approx(0.5)
    assert fidelity.dispersion_score(2.0) == pytest.approx(0.5)
    assert fidelity.dispersion_score(1.0) == pytest.approx(1.0)
    assert fidelity.dispersion_score(0.0) == 0.0
    assert np.isnan(fidelity.dispersion_score(float("nan")))


def test_negative_structure_earns_no_credit_and_stops_the_whole_score() -> None:
    assert fidelity.structure_score(-0.4) == 0.0
    assert fidelity.structure_score(0.3) == pytest.approx(0.3)
    assert fidelity.geometric_mean([0.9, 0.9, 0.0]) == 0.0
    assert np.isnan(fidelity.geometric_mean([0.9, float("nan")]))
    assert fidelity.geometric_mean([0.25, 0.25]) == pytest.approx(0.25)


def test_a_group_is_scored_on_its_own_cells_and_its_own_pairs() -> None:
    values = fidelity.group_fidelity(SURVEY, MODEL)

    assert values["n_cells"] == 5
    assert values["n_pairs"] == 10
    assert values["e_q10_nemd"] <= values["e_median_nemd"] <= values["e_q90_nemd"]
    assert values["adaptability_ratio"] == pytest.approx(values["d_llm"] / values["d_wvs"])
    assert values["score_accuracy"] == pytest.approx(1.0 - values["e_mean_nemd"])
    assert values["score_center"] == pytest.approx(1.0 - values["c_center_nemd"])
    assert not values["model_flat"]
    assert values["pfs"] == pytest.approx(
        fidelity.geometric_mean(
            [values["score_accuracy"], values["score_dispersion"], values["score_structure"]]
        )
    )
    assert values["pfs_with_center"] != values["pfs"]


def test_a_model_with_one_answer_for_every_cell_scores_zero_not_undefined() -> None:
    flat = np.tile(SURVEY[0], (len(SURVEY), 1))
    values = fidelity.group_fidelity(SURVEY, flat)

    assert values["model_flat"]
    assert np.isnan(values["rho_structure"])
    assert values["score_structure"] == 0.0
    assert values["adaptability_ratio"] == 0.0
    assert values["pfs"] == 0.0


def test_a_paired_score_reads_both_conditions_off_the_same_cells() -> None:
    values = fidelity.paired_fidelity(SURVEY, MODEL, BASE)
    tuned = fidelity.group_fidelity(SURVEY, MODEL)
    reference = fidelity.group_fidelity(SURVEY, BASE)

    assert values["n_cells"] == 5
    for metric in fidelity.PAIRED_METRICS:
        assert values[metric] == pytest.approx(tuned[metric])
        assert values[f"base_{metric}"] == pytest.approx(reference[metric])
        assert values[fidelity.delta_column(metric)] == pytest.approx(
            tuned[metric] - reference[metric]
        )
    assert values["base_model_flat"] is False


def _prepared(names: tuple[str, ...], model: np.ndarray, modes: tuple[str, ...]) -> PreparedData:
    index = pd.Index(names, name="name")
    columns = list(QUESTIONS["d_happy"].answer_columns)
    survey = pd.DataFrame(SURVEY[: len(names)], index=index, columns=columns)
    answers = pd.DataFrame(model[: len(names)], index=index, columns=columns)
    return PreparedData(
        question=QUESTIONS["d_happy"],
        wvs=pd.DataFrame(),
        subpops=pd.DataFrame(),
        ntp_raw=pd.DataFrame() if "ntp" in modes else None,
        fa_raw=pd.DataFrame() if "fa" in modes else None,
        names=index,
        wvs_props=survey,
        ntp_props=answers if "ntp" in modes else None,
        fa_props=answers if "fa" in modes else None,
        social_predictors=pd.DataFrame(index=index),
        coverage=Coverage(0, 0, 0, 0, len(names), len(names)),
    )


def _run(arm: str | None, replicate: int = 1) -> RunSource:
    if arm is None:
        return RunSource("Mixtral archived", "archived", None, "Mixtral archived", None, 1)
    source = f"culture/gemma4_31b/{arm}" + ("" if replicate == 1 else f"/rep{replicate}")
    return RunSource(f"Gemma ({arm})", source, "gemma4_31b", "Gemma", arm, replicate)


def test_common_cells_is_the_intersection_over_every_run_named() -> None:
    runs = [
        _prepared(("e", "d", "c", "b", "a"), MODEL, ("fa",)),
        _prepared(("d", "c", "b", "a"), MODEL, ("fa",)),
        _prepared(("c", "b", "e"), MODEL, ("fa",)),
    ]

    assert list(fidelity.common_cells(runs)) == ["b", "c"]
    with pytest.raises(ValueError, match="no runs"):
        fidelity.common_cells([])


def test_each_replicate_is_paired_with_its_own_base_on_the_cells_every_replicate_kept() -> None:
    loaded = [
        (_run("base"), _prepared(("a", "b", "c", "d", "e"), BASE, ("ntp", "fa"))),
        (_run("base", 2), _prepared(("a", "b", "c", "d"), BASE, ("fa",))),
        (_run("german"), _prepared(("b", "c", "d", "e"), MODEL, ("ntp", "fa"))),
        (_run("german", 2), _prepared(("a", "b", "c", "d", "e"), MODEL, ("fa",))),
        (_run("german", 3), _prepared(("a", "b", "c", "d", "e"), MODEL, ("fa",))),
        (_run("spanish-mx"), _prepared(("a", "b", "c", "d", "e"), MODEL, ("fa",))),
        (_run(None), _prepared(("a", "b", "c", "d", "e"), MODEL, ("ntp", "fa"))),
    ]

    pairs = list(fidelity.paired_runs(loaded, "fa"))
    described = [
        (run.arm, run.replicate, base.replicate, list(common))
        for (run, _), (base, _), common in pairs
    ]
    assert described == [
        ("german", 1, 1, ["b", "c", "d"]),
        ("german", 2, 2, ["b", "c", "d"]),
        ("spanish-mx", 1, 1, ["a", "b", "c", "d", "e"]),
    ]

    ntp_pairs = list(fidelity.paired_runs(loaded, "ntp"))
    assert [(run.arm, list(common)) for (run, _), _, common in ntp_pairs] == [
        ("german", ["b", "c", "d", "e"])
    ]


def _identity(question: str, arm: str | None, series: str) -> dict[str, Any]:
    key = None if arm is None else "gemma4_31b"
    return {
        "question": question,
        "question_label": question,
        "model_key": key,
        "model_label": series,
        "arm": arm,
        "series": series,
        "source": "archived" if key is None else f"culture/{key}/{arm}",
        "mode": "ntp",
        "replicate": 1,
    }


def scored() -> pd.DataFrame:
    rows = []
    for question in ("d_happy", "d_trust"):
        for arm, series, answers in (
            ("base", "Gemma (as released)", BASE),
            ("german", "Gemma (german)", MODEL),
            (None, "Mixtral archived", MODEL),
        ):
            for group, level in ((fidelity.POPULATION, fidelity.EVERY_CELL), ("sex", "Female")):
                rows.append(
                    {
                        **_identity(question, arm, series),
                        "group": group,
                        "level": level,
                        **fidelity.group_fidelity(SURVEY, answers),
                    }
                )
    frame = pd.DataFrame(rows)
    frame["binding_term"] = fidelity.binding_term(frame)
    return frame


def paired() -> pd.DataFrame:
    rows = []
    for question in ("d_happy", "d_trust"):
        for group, level in ((fidelity.POPULATION, fidelity.EVERY_CELL), ("sex", "Female")):
            rows.append(
                {
                    **_identity(question, "german", "Gemma (german)"),
                    "base_source": "culture/gemma4_31b/base",
                    "group": group,
                    "level": level,
                    **fidelity.paired_fidelity(SURVEY, MODEL, BASE),
                }
            )
    frame = pd.DataFrame(rows)
    frame["binding_term"] = fidelity.binding_term(frame)
    frame["base_binding_term"] = fidelity.binding_term(frame, "base_")
    return frame


def test_the_binding_term_names_the_smallest_of_the_three() -> None:
    frame = scored()
    smallest = frame[["score_accuracy", "score_dispersion", "score_structure"]].idxmin(axis=1)
    expected = smallest.map(dict(fidelity.TERMS))

    assert frame["binding_term"].tolist() == expected.tolist()
    assert set(frame["binding_term"]) <= {"accuracy", "adaptability", "structure"}


def test_the_all_view_is_the_geometric_mean_across_the_topics() -> None:
    overall = fidelity.overall_fidelity(scored())
    pooled = overall[
        overall["series"].eq("Gemma (as released)") & overall["group"].eq(fidelity.POPULATION)
    ]
    expected = fidelity.group_fidelity(SURVEY, BASE)["pfs"]

    assert len(pooled) == 1
    assert pooled["n_questions"].iat[0] == 2
    assert pooled["pfs"].iat[0] == pytest.approx(expected)
    assert "delta_pfs_vs_base" not in overall.columns


def test_the_paired_all_view_differences_the_two_geometric_means() -> None:
    frame = paired()
    frame.loc[frame["question"].eq("d_trust"), "pfs"] = 0.9
    frame.loc[frame["question"].eq("d_happy"), "pfs"] = 0.4
    frame["base_pfs"] = 0.5
    frame["delta_pfs_vs_base"] = frame["pfs"] - frame["base_pfs"]

    overall = fidelity.overall_paired(frame)
    pooled = overall[overall["group"].eq(fidelity.POPULATION)]

    assert len(pooled) == 1
    assert pooled["pfs"].iat[0] == pytest.approx(0.6)
    assert pooled["base_pfs"].iat[0] == pytest.approx(0.5)
    assert pooled["delta_pfs_vs_base"].iat[0] == pytest.approx(0.1)
    assert fidelity.overall_paired(frame.iloc[0:0]).empty


CONDITIONS: tuple[tuple[str | None, str], ...] = (
    ("base", "Gemma (as released)"),
    ("german", "Gemma (german)"),
    ("spanish-mx", "Gemma (spanish-mx)"),
    ("french", "Gemma (french)"),
    (None, "Mixtral archived"),
)

PFS = (0.10, 0.20, 0.30, 0.40, 0.50)

CENTER = (0.60, 0.70, 0.75, 0.90, 0.65)


def centered() -> pd.DataFrame:
    rows = [
        {
            **_identity("d_happy", arm, series),
            "group": group,
            "level": level,
            "pfs": pfs,
            "score_center": center,
        }
        for (arm, series), pfs, center in zip(CONDITIONS, PFS, CENTER, strict=True)
        for group, level in ((fidelity.POPULATION, fidelity.EVERY_CELL), ("sex", "Female"))
    ]
    return pd.DataFrame(rows)


RESAMPLES = 400


def test_the_correlations_match_scipy_along_the_last_axis() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(5, 12))
    y = rng.normal(size=(5, 12))
    x[:, :4] = np.round(x[:, :4])
    x[0, 5] = np.nan
    y[1, 2] = np.nan

    found = fidelity.correlations(x, y)

    for row in range(len(x)):
        kept = np.isfinite(x[row]) & np.isfinite(y[row])
        expected_rho = spearmanr(x[row, kept], y[row, kept]).statistic
        expected_r = pearsonr(x[row, kept], y[row, kept]).statistic
        assert found["spearman"][row] == pytest.approx(expected_rho)
        assert found["pearson"][row] == pytest.approx(expected_r)


def test_pfs_meets_center_over_the_served_conditions_and_over_every_series() -> None:
    correlation = fidelity.center_correlation(centered(), RESAMPLES)
    pooled = correlation[correlation["group"].eq(fidelity.POPULATION)].set_index("scope")
    served = pooled.loc[fidelity.SERVED]
    every = pooled.loc[fidelity.EVERY_SERIES]

    assert len(correlation) == 4
    assert correlation.columns.tolist() == [
        "scope",
        *fidelity.CORRELATION_KEYS,
        "n_conditions",
        "spearman",
        "spearman_ci_low",
        "spearman_ci_high",
        "pearson",
        "pearson_ci_low",
        "pearson_ci_high",
    ]
    assert served["n_conditions"] == 4
    assert served["spearman"] == pytest.approx(1.0)
    assert (served["spearman_ci_low"], served["spearman_ci_high"]) == (1.0, 1.0)
    assert served["pearson"] == pytest.approx(np.corrcoef(PFS[:4], CENTER[:4])[0, 1])
    assert served["pearson_ci_low"] <= served["pearson"] <= served["pearson_ci_high"]
    assert every["n_conditions"] == 5
    assert every["spearman"] == pytest.approx(0.4)
    assert every["spearman_ci_low"] < every["spearman"] < every["spearman_ci_high"]
    assert every["pearson"] == pytest.approx(np.corrcoef(PFS, CENTER)[0, 1])
    pd.testing.assert_frame_equal(correlation, fidelity.center_correlation(centered(), RESAMPLES))


def test_a_repeated_condition_counts_once_and_a_flat_level_has_no_correlation() -> None:
    frame = centered()
    repeated = pd.concat(
        [frame, frame[frame["series"].eq("Gemma (as released)")].assign(replicate=2)],
        ignore_index=True,
    )
    repeated.loc[repeated["level"].eq("Female"), "score_center"] = 0.8
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        correlation = fidelity.center_correlation(repeated, RESAMPLES)
        few = fidelity.center_correlation(frame[frame["arm"].isin(["base", "german"])], RESAMPLES)
    indexed = correlation.set_index(["scope", "group"])
    statistics = [column for column in correlation.columns if column.startswith(("spear", "pear"))]

    assert indexed.loc[(fidelity.SERVED, fidelity.POPULATION), "n_conditions"] == 4
    assert indexed.loc[(fidelity.SERVED, fidelity.POPULATION), "spearman"] == pytest.approx(1)
    assert indexed.loc[(fidelity.SERVED, "sex"), "n_conditions"] == 4
    assert indexed.loc[(fidelity.SERVED, "sex"), statistics].isna().to_numpy().all()
    assert few["n_conditions"].eq(2).all()
    assert few[statistics].isna().to_numpy().all()


def test_the_summary_sets_the_pooled_correlation_beside_the_mean_across_levels() -> None:
    frame = centered()
    female = frame[frame["level"].eq("Female")]
    groups = pd.concat(
        [frame, female.assign(level="Male", pfs=[0.3, 0.1, 0.4, 0.2, 0.5])], ignore_index=True
    )

    correlation = fidelity.center_correlation(groups, RESAMPLES)
    summary = fidelity.center_correlation_summary(groups, RESAMPLES)
    single = fidelity.center_correlation_summary(frame, RESAMPLES).iloc[0]
    alone = fidelity.center_correlation(frame, RESAMPLES)

    served = correlation[correlation["scope"].eq(fidelity.SERVED)]
    pooled = served[served["group"].eq(fidelity.POPULATION)].iloc[0]
    levels = served[~served["group"].eq(fidelity.POPULATION)]
    row = summary[summary["scope"].eq(fidelity.SERVED)].iloc[0]
    assert len(summary) == 2
    assert (row["n_conditions"], row["n_levels"]) == (4, 2)
    for part in fidelity.INTERVAL_PARTS:
        assert row[f"population_spearman{part}"] == pytest.approx(pooled[f"spearman{part}"])
    assert row["spearman_mean"] == pytest.approx(levels["spearman"].mean())
    assert row["spearman_min"] == pytest.approx(levels["spearman"].min())
    assert row["pearson_max"] == pytest.approx(levels["pearson"].max())
    assert "spearman_sd" not in summary.columns
    one_level = alone[alone["scope"].eq(fidelity.SERVED) & alone["level"].eq("Female")].iloc[0]
    assert single["spearman_mean_ci_low"] == pytest.approx(one_level["spearman_ci_low"])
    assert single["pearson_mean_ci_high"] == pytest.approx(one_level["pearson_ci_high"])


def test_the_checks_accept_paired_rows_that_never_exceed_their_runs() -> None:
    groups = scored()

    fidelity._check_paired(groups, paired())
    fidelity._check_paired(groups, paired().iloc[0:0])

    oversized = paired()
    oversized["n_cells"] = 6
    with pytest.raises(AssertionError, match="exceed"):
        fidelity._check_paired(groups, oversized)

    uneven = pd.concat([paired(), paired().assign(replicate=2, n_cells=4)], ignore_index=True)
    with pytest.raises(AssertionError, match="different cells"):
        fidelity._check_paired(groups, uneven)


def test_the_root_holds_the_pooled_report_and_a_subfolder_holds_each_family(
    tmp_path: Path,
) -> None:
    groups = scored()
    shifts = paired()
    written = fidelity_plates.fidelity_plates(
        groups,
        fidelity.overall_fidelity(groups),
        shifts,
        fidelity.overall_paired(shifts),
        tmp_path,
    )

    assert sorted({path.suffix for path in written}) == [".pdf", ".png"]
    assert sorted(path.name for path in tmp_path.glob("*.png")) == [
        "fig_fidelity_cells_all_ntp.png",
        "fig_fidelity_cells_happiness_ntp.png",
        "fig_fidelity_cells_trust_ntp.png",
        "fig_fidelity_center_all_ntp.png",
        "fig_fidelity_center_happiness_ntp.png",
        "fig_fidelity_center_trust_ntp.png",
        "fig_fidelity_components_all_ntp.png",
        "fig_fidelity_components_happiness_ntp.png",
        "fig_fidelity_components_trust_ntp.png",
        "fig_fidelity_ranking_all_ntp.png",
        "fig_fidelity_ranking_happiness_ntp.png",
        "fig_fidelity_ranking_trust_ntp.png",
        "fig_fidelity_shift_all_ntp.png",
        "fig_fidelity_shift_happiness_ntp.png",
        "fig_fidelity_shift_trust_ntp.png",
        "fig_fidelity_subgroups_all.png",
        "fig_fidelity_subgroups_happiness.png",
        "fig_fidelity_subgroups_models_all_ntp.png",
        "fig_fidelity_subgroups_models_happiness_ntp.png",
        "fig_fidelity_subgroups_models_questions_ntp.png",
        "fig_fidelity_subgroups_models_trust_ntp.png",
        "fig_fidelity_subgroups_models_wide_all_ntp.png",
        "fig_fidelity_subgroups_models_wide_happiness_ntp.png",
        "fig_fidelity_subgroups_models_wide_trust_ntp.png",
        "fig_fidelity_subgroups_questions_ntp.png",
        "fig_fidelity_subgroups_trust.png",
        "fig_fidelity_subgroups_wide_all_ntp.png",
        "fig_fidelity_subgroups_wide_happiness_ntp.png",
        "fig_fidelity_subgroups_wide_trust_ntp.png",
    ]
    assert sorted(path.name for path in (tmp_path / "sex").glob("*.png")) == [
        "fig_fidelity_sex_all_ntp.png",
        "fig_fidelity_sex_components_all_ntp.png",
        "fig_fidelity_sex_components_happiness_ntp.png",
        "fig_fidelity_sex_components_trust_ntp.png",
        "fig_fidelity_sex_happiness_ntp.png",
        "fig_fidelity_sex_shift_all_ntp.png",
        "fig_fidelity_sex_shift_happiness_ntp.png",
        "fig_fidelity_sex_shift_trust_ntp.png",
        "fig_fidelity_sex_trust_ntp.png",
    ]
    assert not (tmp_path / "country").exists()


def test_the_subgroup_plate_lists_the_pooled_cells_first_and_each_family_after_a_spacer(
    tmp_path: Path,
) -> None:
    groups = scored()
    overall = fidelity.overall_fidelity(groups)

    rows = fidelity_plates.level_rows(groups)
    assert [(family, level) for family, level, _ in rows] == [
        (fidelity.POPULATION, fidelity.EVERY_CELL),
        (None, None),
        ("sex", "Female"),
    ]
    assert rows[0][2] == r"$\bf{All\ subpop.}$"
    assert rows[2][2] == r"$\bf{Sex}$: Female"
    assert fidelity_plates.family_blocks(rows) == [("sex", 2.0, 2.0)]

    shuffled = pd.DataFrame({"group": ["age", "country", "age"], "level": ["75+", "Mexico", "<25"]})
    assert [level for _, level, _ in fidelity_plates.level_rows(shuffled)] == [
        "Mexico",
        None,
        "<25",
        "75+",
    ]

    frame = fidelity_plates.subgroup_frame(groups, overall, "happiness", "ntp")
    assert set(frame["series"]) == {"Gemma (as released)", "Gemma (german)"}
    assert set(frame["group"]) == {fidelity.POPULATION, "sex"}
    assert fidelity_plates.subgroup_frame(groups, overall, "happiness", "fa").empty

    written = fidelity_plates.subgroups_plate(groups, overall, "happiness", tmp_path)
    assert sorted(path.name for path in written) == [
        "fig_fidelity_subgroups_happiness.pdf",
        "fig_fidelity_subgroups_happiness.png",
    ]
    assert fidelity_plates.subgroups_plate(groups, overall, "politics", tmp_path) == []

    offsets = fidelity_plates.variant_offsets(frame)
    assert list(offsets) == ["base", "german"]
    assert offsets["base"] < 0.0 < offsets["german"]
    named = fidelity_plates.subgroup_models_plate(groups, overall, "trust", "ntp", tmp_path)
    assert sorted(path.name for path in named) == [
        "fig_fidelity_subgroups_models_trust_ntp.pdf",
        "fig_fidelity_subgroups_models_trust_ntp.png",
    ]
    assert fidelity_plates.subgroup_models_plate(groups, overall, "trust", "fa", tmp_path) == []

    blocks = fidelity_plates.family_blocks(
        fidelity_plates.level_rows(
            pd.DataFrame({"group": ["age", "age", "country"], "level": ["<25", "75+", "Mexico"]})
        )
    )
    assert blocks == [("country", 0.0, 0.0), ("age", 2.0, 3.0)]
    wide = fidelity_plates.subgroups_wide_plate(groups, overall, "trust", "ntp", tmp_path)
    assert sorted(path.name for path in wide) == [
        "fig_fidelity_subgroups_wide_trust_ntp.pdf",
        "fig_fidelity_subgroups_wide_trust_ntp.png",
    ]
    assert fidelity_plates.subgroups_wide_plate(groups, overall, "trust", "fa", tmp_path) == []
    models = fidelity_plates.subgroup_models_wide_plate(groups, overall, "trust", "ntp", tmp_path)
    assert sorted(path.name for path in models) == [
        "fig_fidelity_subgroups_models_wide_trust_ntp.pdf",
        "fig_fidelity_subgroups_models_wide_trust_ntp.png",
    ]
    stacked = fidelity_plates.subgroups_questions_plate(groups, overall, "ntp", tmp_path)
    assert sorted(path.name for path in stacked) == [
        "fig_fidelity_subgroups_questions_ntp.pdf",
        "fig_fidelity_subgroups_questions_ntp.png",
    ]
    assert fidelity_plates.subgroups_questions_plate(groups, overall, "fa", tmp_path) == []
    assert fidelity_plates.subgroup_models_questions_plate(groups, overall, "fa", tmp_path) == []


def published() -> pd.DataFrame:
    rows = []
    for question in ("d_happy", "d_trust"):
        for model, mode, answers in (("GPT-4T", "ntp", MODEL), ("GPT-3", "fa", BASE)):
            for group, level in ((fidelity.POPULATION, fidelity.EVERY_CELL), ("sex", "Female")):
                rows.append(
                    {
                        **archived_reference.archived_identity(
                            model, resolve_question(question), mode
                        ),
                        "group": group,
                        "level": level,
                        **fidelity.group_fidelity(SURVEY, answers),
                    }
                )
    frame = pd.DataFrame(rows)
    frame["binding_term"] = fidelity.binding_term(frame)
    return frame


def test_the_proprietary_plate_reads_each_archived_series_in_its_own_mode(tmp_path: Path) -> None:
    groups = scored()
    overall = fidelity.overall_fidelity(groups)
    archived = published()
    off_mode = archived[archived["model_label"].eq("GPT-3")].assign(mode="ntp")
    archived = pd.concat([archived, off_mode], ignore_index=True)
    archived_overall = fidelity.overall_fidelity(archived)

    frame = fidelity_plates.proprietary_frame(groups, overall, archived, archived_overall, "trust")
    assert dict(zip(frame["series"], frame["mode"], strict=True)) == {"GPT-3 archived": "fa"}
    pooled = fidelity_plates.proprietary_frame(groups, overall, archived, archived_overall, "all")
    assert "question" not in pooled.columns
    assert len(pooled) == 2
    labels = {fidelity_plates._proprietary_label(row) for row in frame.itertuples()}
    assert labels == {"GPT-3 archived, FA"}

    written = fidelity_plates.proprietary_plate(
        groups, overall, archived, archived_overall, "trust", tmp_path
    )
    assert sorted(path.name for path in written) == [
        "fig_fidelity_subgroups_proprietary_trust.pdf",
        "fig_fidelity_subgroups_proprietary_trust.png",
    ]
    wide = fidelity_plates.proprietary_wide_plate(
        groups, overall, archived, archived_overall, "trust", tmp_path
    )
    assert sorted(path.name for path in wide) == [
        "fig_fidelity_subgroups_proprietary_wide_trust.pdf",
        "fig_fidelity_subgroups_proprietary_wide_trust.png",
    ]
    stacked = fidelity_plates.proprietary_questions_plate(
        groups, overall, archived, archived_overall, tmp_path
    )
    assert sorted(path.name for path in stacked) == [
        "fig_fidelity_subgroups_proprietary_questions.pdf",
        "fig_fidelity_subgroups_proprietary_questions.png",
    ]

    shifts = paired()
    fidelity_plates.fidelity_plates(
        groups,
        overall,
        shifts,
        fidelity.overall_paired(shifts),
        tmp_path / "with_archive",
        archived=archived,
    )
    assert sorted(path.name for path in (tmp_path / "with_archive").glob("*proprietary*.png")) == [
        "fig_fidelity_subgroups_proprietary_all.png",
        "fig_fidelity_subgroups_proprietary_happiness.png",
        "fig_fidelity_subgroups_proprietary_questions.png",
        "fig_fidelity_subgroups_proprietary_trust.png",
        "fig_fidelity_subgroups_proprietary_wide_all.png",
        "fig_fidelity_subgroups_proprietary_wide_happiness.png",
        "fig_fidelity_subgroups_proprietary_wide_trust.png",
    ]


def test_a_shift_plate_spans_each_axis_on_its_own_reach() -> None:
    frame = pd.DataFrame(
        {
            fidelity_plates.DELTA_PFS: [-0.4, 0.1],
            fidelity_plates.DELTA_CENTER: [0.05, -0.1],
            f"{fidelity_plates.DELTA_CENTER}_ci": [0.02, np.nan],
        }
    )
    assert fidelity_plates._span(frame, fidelity_plates.DELTA_PFS) == (-0.5, 0.5)
    assert fidelity_plates._span(frame, fidelity_plates.DELTA_CENTER) == pytest.approx(
        (-0.125, 0.125)
    )
    assert fidelity_plates._span(frame.iloc[0:0], fidelity_plates.DELTA_PFS) == (-0.125, 0.125)
    assert fidelity_plates._span(frame, fidelity_plates.DELTA_CENTER, 0.7) == pytest.approx(
        (-0.17, 0.17)
    )


def test_every_shift_corner_names_both_scores_in_the_direction_of_its_quadrant() -> None:
    corners = fidelity_plates.SHIFT_CORNERS
    assert len({(x > 0.5, y > 0.5) for x, y, *_ in corners}) == 4
    for x, y, _, _, text in corners:
        center = "improved" if y > 0.5 else "worsened"
        pfs = "improved" if x > 0.5 else "worsened"
        assert text == f"Center {center},\nfidelity {pfs}"


def test_the_component_legend_names_each_score_in_words_and_by_its_symbol() -> None:
    from matplotlib.mathtext import MathTextParser

    parser = MathTextParser("path")
    labels = [label for _, _, label in fidelity_plates.COMPONENTS]
    assert labels == [
        r"Accuracy ($S_{\mathrm{acc}}$)",
        r"Adaptability ($S_{\mathrm{adapt}}$)",
        r"Structure ($S_{\mathrm{struct}}$)",
    ]
    for label in labels:
        parser.parse(label)


def test_the_center_legend_names_the_score_in_words_and_by_its_symbol() -> None:
    from matplotlib.mathtext import MathTextParser

    assert fidelity_plates.CENTER_LEGEND == r"Center alignment ($S_{\mathrm{center}}$)"
    MathTextParser("path").parse(fidelity_plates.CENTER_LEGEND)
    labels = [handle.get_label() for handle in fidelity_plates._box_handles("FA 1")]
    assert labels[:2] == [fidelity_plates.PFS_LABEL, fidelity_plates.CENTER_LEGEND]


def test_the_wide_box_plate_stands_its_level_names_upright_under_their_columns() -> None:
    from matplotlib.figure import Figure

    rows = fidelity_plates.level_rows(scored())
    leaning = Figure().subplots(1, 1)
    fidelity_plates._wide_ticks(leaning, rows)
    labels = leaning.get_xticklabels()
    assert [label.get_text() for label in labels] == ["All subpop.", "", "Female"]
    assert fidelity_plates._tick_names([("marital", "Divorced or separated", "")]) == [
        "Divor. / separ."
    ]
    marital = fidelity_plates.level_rows(
        pd.DataFrame({"group": ["marital_status"], "level": ["Divorced or separated"]})
    )
    assert marital[0][2] == r"$\bf{Marital\ status}$: Divor. / separ."
    assert {label.get_rotation() for label in labels} == {90.0}
    assert {label.get_horizontalalignment() for label in labels} == {"center"}

    stacked = Figure().subplots(1, 1)
    fidelity_plates._wide_tick_labels(stacked, rows)
    assert stacked.get_xticklabels()[0].get_text() == "All subpop."
    assert {label.get_rotation() for label in stacked.get_xticklabels()} == {60.0}


def test_replicates_are_drawn_as_one_marker_with_a_spread(tmp_path: Path) -> None:
    groups = scored()
    second = groups[groups["arm"].eq("german")].copy()
    second["replicate"] = 2
    second["pfs"] = second["pfs"] + 0.05
    shifts = paired()
    again = shifts.copy()
    again["replicate"] = 2
    again["delta_pfs_vs_base"] = again["delta_pfs_vs_base"] + 0.1
    stacked = pd.concat([groups, second], ignore_index=True)
    stacked_shifts = pd.concat([shifts, again], ignore_index=True)

    written = fidelity_plates.fidelity_plates(
        stacked,
        fidelity.overall_fidelity(stacked),
        stacked_shifts,
        fidelity.overall_paired(stacked_shifts),
        tmp_path,
    )

    assert (tmp_path / "fig_fidelity_shift_happiness_ntp.png") in written


HOME_CONDITIONS: tuple[tuple[str | None, str | None, str], ...] = (
    ("gemma4_31b", "base", "Gemma (as released)"),
    ("gemma4_31b", "german", "Gemma (german)"),
    ("gemma4_31b", "spanish-mx", "Gemma (spanish-mx)"),
    ("llama3_2_3b", "base", "Llama (as released)"),
    ("llama3_2_3b", "german", "Llama (german)"),
    ("llama3_2_3b", "spanish-mx", "Llama (spanish-mx)"),
    ("terra", "base", "GPT-5.6-Terra (as released)"),
    (None, None, "Mixtral archived"),
)

HOME_LEVELS: tuple[tuple[str, str], ...] = (
    (fidelity.POPULATION, fidelity.EVERY_CELL),
    ("country", "Germany"),
    ("country", "Mexico"),
)

HOME_PFS: dict[tuple[str, str], float] = {
    ("Gemma (as released)", fidelity.EVERY_CELL): 0.30,
    ("Llama (as released)", fidelity.EVERY_CELL): 0.50,
    ("GPT-5.6-Terra (as released)", fidelity.EVERY_CELL): 0.40,
    ("Mixtral archived", fidelity.EVERY_CELL): 0.45,
    ("Gemma (as released)", "Germany"): 0.50,
    ("Gemma (german)", "Germany"): 0.55,
    ("Llama (as released)", "Germany"): 0.60,
    ("Llama (german)", "Germany"): 0.40,
    ("GPT-5.6-Terra (as released)", "Germany"): 0.70,
    ("Mixtral archived", "Germany"): 0.65,
    ("Gemma (as released)", "Mexico"): 0.60,
    ("Gemma (spanish-mx)", "Mexico"): 0.30,
    ("Llama (as released)", "Mexico"): 0.50,
    ("Llama (spanish-mx)", "Mexico"): 0.45,
    ("GPT-5.6-Terra (as released)", "Mexico"): 0.70,
    ("Mixtral archived", "Mexico"): 0.65,
}

TUNED_ELSEWHERE_PFS = 0.95


def _home_identity(question: str, key: str | None, arm: str | None, series: str) -> dict[str, Any]:
    return {
        "question": question,
        "question_label": question,
        "model_key": key,
        "model_label": series.split(" (")[0],
        "arm": arm,
        "series": series,
        "source": "archived" if key is None else f"culture/{key}/{arm}",
        "mode": "ntp",
        "replicate": 1,
    }


def homed() -> pd.DataFrame:
    rows = []
    for question in ("d_happy", "d_trust"):
        for key, arm, series in HOME_CONDITIONS:
            for group, level in HOME_LEVELS:
                rows.append(
                    {
                        **_home_identity(question, key, arm, series),
                        "group": group,
                        "level": level,
                        **fidelity.group_fidelity(SURVEY, BASE if arm == "base" else MODEL),
                        "pfs": HOME_PFS.get((series, level), TUNED_ELSEWHERE_PFS),
                    }
                )
    frame = pd.DataFrame(rows)
    frame["binding_term"] = fidelity.binding_term(frame)
    return frame


def homed_pairs() -> pd.DataFrame:
    rows = []
    for question in ("d_happy", "d_trust"):
        for key, arm, series in HOME_CONDITIONS:
            if arm in (None, "base"):
                continue
            for group, level in (*HOME_LEVELS, ("country", "Russia")):
                rows.append(
                    {
                        **_home_identity(question, key, arm, series),
                        "base_source": f"culture/{key}/base",
                        "group": group,
                        "level": level,
                        **fidelity.paired_fidelity(SURVEY, MODEL, BASE),
                    }
                )
    frame = pd.DataFrame(rows)
    frame["binding_term"] = fidelity.binding_term(frame)
    frame["base_binding_term"] = fidelity.binding_term(frame, "base_")
    return frame


def test_the_home_plate_pairs_each_finetuned_condition_with_its_released_model_on_its_own_cells(
    tmp_path: Path,
) -> None:
    levels = homed().assign(n_replicates=1)
    pooled = fidelity_plates.view_frame(levels, levels, "happiness", "ntp", fidelity.POPULATION)
    countries = fidelity_plates.view_frame(levels, levels, "happiness", "ntp", "country")

    blocks = fidelity_plates.home_blocks(pooled, countries)

    assert [header for header, _ in blocks] == [
        "All retained cells",
        "German cells only",
        "Mexican cells only",
    ]
    released, germany, mexico = (rows for _, rows in blocks)
    assert released["series"].tolist() == [
        "Llama (as released)",
        "Mixtral archived",
        "GPT-5.6-Terra (as released)",
        "Gemma (as released)",
    ]
    assert set(released["level"]) == {fidelity.EVERY_CELL}
    assert germany["series"].tolist() == [
        "Llama (as released)",
        "Llama (german)",
        "Gemma (as released)",
        "Gemma (german)",
    ]
    assert set(germany["level"]) == {"Germany"}
    assert mexico["series"].tolist() == [
        "Gemma (as released)",
        "Gemma (spanish-mx)",
        "Llama (as released)",
        "Llama (spanish-mx)",
    ]
    assert set(mexico["level"]) == {"Mexico"}

    files = fidelity_plates.home_components_plate(levels, levels, "happiness", "ntp", tmp_path)

    assert {path.name for path in files} == {
        "fig_fidelity_components_home_happiness_ntp.pdf",
        "fig_fidelity_components_home_happiness_ntp.png",
    }
    flat = scored().assign(n_replicates=1)
    assert fidelity_plates.home_components_plate(flat, flat, "happiness", "ntp", tmp_path) == []


def test_the_home_shift_reads_each_finetuned_condition_on_its_target_country(
    tmp_path: Path,
) -> None:
    shifts = homed_pairs()

    homed_rows = fidelity_plates.home_shifts(
        shifts, fidelity.overall_paired(shifts), "happiness", "ntp"
    )

    assert sorted(zip(homed_rows["series"], homed_rows["level"], strict=True)) == [
        ("Gemma (german)", "Germany"),
        ("Gemma (spanish-mx)", "Mexico"),
        ("Llama (german)", "Germany"),
        ("Llama (spanish-mx)", "Mexico"),
    ]
    written = fidelity_plates.shift_plate(
        homed_rows.assign(n_replicates=1), "ntp", "happiness", tmp_path, fidelity_plates.HOME_SCOPE
    )
    assert {path.name for path in written} == {
        "fig_fidelity_shift_home_happiness_ntp.pdf",
        "fig_fidelity_shift_home_happiness_ntp.png",
    }
    pooled = fidelity_plates.view_frame(shifts, shifts, "happiness", "ntp", fidelity.POPULATION)
    default = fidelity_plates.shift_plate(
        pooled.assign(n_replicates=1), "ntp", "happiness", tmp_path
    )
    assert {path.name for path in default} == {
        "fig_fidelity_shift_happiness_ntp.pdf",
        "fig_fidelity_shift_happiness_ntp.png",
    }
    assert fidelity_plates.SHIFT_XLABELS[fidelity_plates.HOME_SCOPE].endswith(
        "(target-country cells)"
    )
    assert fidelity_plates.home_shifts(
        paired(), fidelity.overall_paired(paired()), "happiness", "ntp"
    ).empty


def test_the_sweep_draws_the_home_plates_when_the_country_cells_are_scored(tmp_path: Path) -> None:
    groups, shifts = homed(), homed_pairs()

    written = fidelity_plates.fidelity_plates(
        groups,
        fidelity.overall_fidelity(groups),
        shifts,
        fidelity.overall_paired(shifts),
        tmp_path,
    )

    assert {
        "fig_fidelity_components_home_happiness_ntp.png",
        "fig_fidelity_components_home_all_ntp.png",
        "fig_fidelity_shift_home_happiness_ntp.png",
        "fig_fidelity_shift_home_all_ntp.png",
    } <= {path.name for path in written}
