from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from culture import fidelity, fidelity_orientation
from culture.population import RunSource, run_identity
from machine_bias_reproduction.config import GLOBAL_SEED
from machine_bias_reproduction.questions import resolve_question

MODEL_KEY = "gemma4_31b"

CELLS = (
    "Germany 2018 Female 45-54 Middle Working Married",
    "Germany 1997 Male 25-34 High Student Single",
    "Mexico 2005 Female 55-64 Low Retired Widowed",
    "Mexico 2017 Male <25 High Working Single",
    "United States 1995 Female 75+ Middle Homemaker Cohabiting",
    "United States 2017 Male 35-44 High Working Single",
)

SURVEY = np.array(
    [
        [0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4],
        [0.25, 0.25, 0.25, 0.25],
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.7],
        [0.2, 0.2, 0.3, 0.3],
    ]
)

RELEASED = np.array(
    [
        [0.35, 0.35, 0.2, 0.1],
        [0.15, 0.25, 0.3, 0.3],
        [0.2, 0.3, 0.3, 0.2],
        [0.6, 0.2, 0.1, 0.1],
        [0.2, 0.1, 0.2, 0.5],
        [0.25, 0.25, 0.25, 0.25],
    ]
)

REVERSED = SURVEY[:, ::-1]

COUNTS = np.array([40.0, 35.0, 50.0, 30.0, 45.0, 60.0])

ANSWERS: dict[str, np.ndarray] = {
    "base": RELEASED,
    "german": REVERSED,
    "spanish-mx": 0.5 * (RELEASED + SURVEY),
}

TRUSTING = np.array([[0.8, 0.2], [0.5, 0.5], [0.1, 0.9]])

EXAMPLE_SURVEY = np.array([[0.80, 0.20], [0.50, 0.50], [0.20, 0.80]])

EXAMPLE_MODEL = np.array([[0.21, 0.79], [0.51, 0.49], [0.78, 0.22]])

EVEN = np.full(3, 50.0)

ALTERNATIVES = fidelity_orientation.ALTERNATIVES


def _binary(shares: tuple[float, ...]) -> np.ndarray:
    yes = np.array(shares)
    return np.column_stack([1.0 - yes, yes])


def _run(arm: str, replicate: int = 1) -> RunSource:
    source = f"culture/{MODEL_KEY}/{arm}"
    if replicate > 1:
        source = f"{source}/rep{replicate}"
    return RunSource(f"Gemma ({arm})", source, MODEL_KEY, "Gemma", arm, replicate)


def _facet() -> pd.DataFrame:
    return fidelity.cell_facets(CELLS).reindex(list(CELLS))


def _groups(replicate: int = 1, published: bool = False) -> pd.DataFrame:
    question = resolve_question("d_happy")
    rows: list[dict[str, Any]] = []
    for arm, answers in ANSWERS.items():
        identity = run_identity(_run(arm, replicate), question, "ntp")
        if published:
            rows += fidelity.grouped_scores(
                identity, _facet(), fidelity.group_fidelity, SURVEY, answers
            )
        else:
            rows += fidelity.grouped_scores(
                identity,
                _facet(),
                fidelity_orientation.orientation_fidelity,
                SURVEY,
                answers,
                COUNTS,
            )
    frame = pd.DataFrame(rows)
    return frame.assign(binding_term=fidelity.binding_term(frame))


def _paired(replicate: int = 1, published: bool = False) -> pd.DataFrame:
    question = resolve_question("d_happy")
    rows: list[dict[str, Any]] = []
    for arm, answers in ANSWERS.items():
        if arm == "base":
            continue
        identity = {
            **run_identity(_run(arm, replicate), question, "ntp"),
            "base_source": _run("base", replicate).source,
        }
        if published:
            rows += fidelity.grouped_scores(
                identity, _facet(), fidelity.paired_fidelity, SURVEY, answers, RELEASED
            )
        else:
            rows += fidelity.grouped_scores(
                identity,
                _facet(),
                fidelity_orientation.paired_orientation,
                SURVEY,
                answers,
                RELEASED,
                COUNTS,
            )
    frame = pd.DataFrame(rows)
    frame["binding_term"] = fidelity.binding_term(frame)
    frame["base_binding_term"] = fidelity.binding_term(frame, "base_")
    return frame


def _pooled(frame: pd.DataFrame, arm: str) -> pd.Series:
    rows = frame[frame["group"].eq(fidelity.POPULATION) & frame["arm"].eq(arm)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_the_anchor_is_the_normalized_mean_response() -> None:
    scale = np.linspace(0.0, 1.0, SURVEY.shape[1])

    assert np.allclose(fidelity_orientation.anchor_scores(SURVEY), SURVEY @ scale)
    assert np.allclose(fidelity_orientation.anchor_scores(TRUSTING), TRUSTING[:, 1])
    assert fidelity_orientation.anchor_scores(SURVEY[:1]).shape == (1,)


def test_a_reversed_ordering_keeps_its_structure_and_loses_it_under_every_variant() -> None:
    row = fidelity_orientation.orientation_fidelity(TRUSTING, TRUSTING[:, ::-1], EVEN)

    assert row["rho_structure"] == pytest.approx(1.0)
    assert row["pfs"] > 0.0
    assert row["rho_orientation"] == pytest.approx(-1.0)
    assert row["rho_signed"] == pytest.approx(-1.0)
    assert row["orientation_reversed"] is True
    for variant in ALTERNATIVES:
        assert row[fidelity_orientation.structure_column(variant)] == 0.0
        assert row[fidelity_orientation.pfs_column(variant)] == 0.0


def test_the_worked_example_reverses_three_groups_that_structure_scores_high() -> None:
    row = fidelity_orientation.orientation_fidelity(EXAMPLE_SURVEY, EXAMPLE_MODEL, EVEN)

    assert row["rho_structure"] > 0.86
    assert row["rho_orientation"] == pytest.approx(-1.0)
    assert row["rho_signed"] < -0.86
    assert row["score_structure_capped"] == 0.0
    assert row["score_structure_signed"] == 0.0
    assert row["pfs"] > 0.0
    assert row["pfs_capped"] == 0.0
    assert row["pfs_signed"] == 0.0


def test_a_mirror_image_of_the_survey_scores_zero_under_every_variant() -> None:
    survey = 0.5 * SURVEY + 0.125
    mirror = 2.0 * survey.mean(axis=0) - survey
    assert (mirror >= 0.0).all()

    row = fidelity_orientation.orientation_fidelity(survey, mirror, COUNTS)

    assert row["rho_structure"] > 0.99
    assert row["adaptability_ratio"] == pytest.approx(1.0)
    assert row["score_center"] == pytest.approx(1.0)
    assert row["pfs"] > 0.8
    assert row["rho_orientation"] == pytest.approx(-1.0)
    assert row["rho_signed"] < -0.99
    for variant in ALTERNATIVES:
        assert row[fidelity_orientation.pfs_column(variant)] == 0.0


def test_a_model_that_keeps_the_ordering_scores_what_the_original_scores() -> None:
    survey = _binary((0.10, 0.30, 0.40, 0.80))
    model = _binary((0.20, 0.25, 0.50, 0.60))

    row = fidelity_orientation.orientation_fidelity(survey, model, np.full(4, 50.0))

    assert 0.0 < row["rho_structure"] < 1.0
    assert row["rho_orientation"] == pytest.approx(1.0)
    assert row["orientation_reversed"] is False
    assert row["rho_signed"] == pytest.approx(row["rho_structure"])
    assert row["n_pairs_signed"] == row["n_pairs"] == 6
    for variant in ("gated", "capped", "signed"):
        assert row[f"score_structure_{variant}"] == pytest.approx(row["score_structure"])
        assert row[f"pfs_{variant}"] == pytest.approx(row["pfs"])


def test_the_cap_lowers_a_structure_that_outruns_its_orientation() -> None:
    survey = _binary((0.10, 0.22, 0.80, 0.93))
    model = _binary((0.21, 0.10, 0.81, 0.93))

    row = fidelity_orientation.orientation_fidelity(survey, model, np.full(4, 50.0))

    assert row["rho_orientation"] == pytest.approx(0.8)
    assert 0.0 < row["rho_orientation"] < row["score_structure"]
    assert row["score_structure_gated"] == pytest.approx(row["score_structure"])
    assert row["score_structure_capped"] == pytest.approx(row["rho_orientation"])
    assert row["pfs_capped"] < row["pfs_gated"] == pytest.approx(row["pfs"])


def test_the_readings_ignore_the_order_of_the_cells_and_the_direction_of_the_scale() -> None:
    draws = np.random.default_rng(GLOBAL_SEED)
    survey = draws.dirichlet(np.ones(5), size=8)
    model = draws.dirichlet(np.ones(5), size=8)
    counts = draws.integers(20, 80, size=8).astype(np.float64)
    shuffled = draws.permutation(8)
    readings = (
        *fidelity_orientation.ORIENTATION_READINGS,
        *fidelity_orientation.PAIR_COUNTS,
        *fidelity_orientation.VARIANT_SCORES,
    )

    row = fidelity_orientation.orientation_fidelity(survey, model, counts)
    moved = fidelity_orientation.orientation_fidelity(
        survey[shuffled], model[shuffled], counts[shuffled]
    )
    flipped = fidelity_orientation.orientation_fidelity(survey[:, ::-1], model[:, ::-1], counts)

    for reading in readings:
        assert moved[reading] == pytest.approx(row[reading], nan_ok=True)
        assert flipped[reading] == pytest.approx(row[reading], nan_ok=True)


def test_a_near_tie_in_the_survey_means_is_dropped_at_its_tolerance() -> None:
    survey = _binary((0.100, 0.115, 0.400, 0.800))
    model = _binary((0.115, 0.100, 0.400, 0.800))

    sure = fidelity_orientation.orientation_fidelity(survey, model, np.full(4, 1e6))
    unsure = fidelity_orientation.orientation_fidelity(survey, model, np.full(4, 10.0))
    tied = fidelity_orientation.orientation_fidelity(
        _binary((0.100, 0.100, 0.400, 0.800)), model, np.full(4, 1e6)
    )

    assert sure["n_pairs_signed"] == sure["n_pairs_signed_tie_01"] == 6
    assert sure["n_pairs_signed_tie_02"] == sure["n_pairs_signed_tie_05"] == 5
    assert sure["n_pairs_signed_tie_se"] == 6
    assert unsure["n_pairs_signed_tie_se"] < 6
    assert np.isfinite(sure["rho_signed"]) and np.isfinite(sure["rho_signed_tie_02"])
    assert sure["rho_signed"] != pytest.approx(sure["rho_signed_tie_02"])
    assert tied["n_pairs_signed"] == 5


def test_a_model_without_an_ordering_earns_no_structure() -> None:
    survey = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4]])
    unordered = np.array([[0.25, 0.5, 0.25], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.1, 0.8, 0.1]])

    row = fidelity_orientation.orientation_fidelity(survey, unordered, np.full(4, 50.0))

    assert row["model_flat"] is False
    assert np.isnan(row["rho_orientation"])
    assert np.isnan(row["rho_signed"])
    assert row["orientation_reversed"] is False
    for variant in ALTERNATIVES:
        assert row[fidelity_orientation.structure_column(variant)] == 0.0
        assert row[fidelity_orientation.pfs_column(variant)] == 0.0


def test_a_level_too_small_for_structure_stays_blank_or_flat_as_the_original_does() -> None:
    alone = fidelity_orientation.orientation_fidelity(SURVEY[:1], RELEASED[:1], COUNTS[:1])
    two = fidelity_orientation.orientation_fidelity(SURVEY[:2], RELEASED[:2], COUNTS[:2])

    assert np.isnan(alone["pfs"])
    assert np.isnan(two["rho_orientation"])
    assert two["score_structure"] == 0.0
    for variant in ALTERNATIVES:
        assert np.isnan(alone[fidelity_orientation.pfs_column(variant)])
        assert two[fidelity_orientation.structure_column(variant)] == 0.0


def test_the_original_columns_are_the_scores_the_original_scorer_gives() -> None:
    row = fidelity_orientation.orientation_fidelity(SURVEY, RELEASED, COUNTS)
    original = fidelity.group_fidelity(SURVEY, RELEASED)

    assert list(row)[: len(original)] == list(original)
    for column, value in original.items():
        assert row[column] == pytest.approx(value, nan_ok=True)


def test_a_paired_row_changes_by_tuned_minus_base_under_every_variant() -> None:
    row = fidelity_orientation.paired_orientation(SURVEY, REVERSED, RELEASED, COUNTS)
    original = fidelity.paired_fidelity(SURVEY, REVERSED, RELEASED)
    base = fidelity_orientation.orientation_fidelity(SURVEY, RELEASED, COUNTS)

    for column, value in original.items():
        assert row[column] == pytest.approx(value, nan_ok=True)
    for metric in fidelity_orientation.ORIENTATION_METRICS:
        assert row[f"base_{metric}"] == pytest.approx(base[metric], nan_ok=True)
        assert row[fidelity.delta_column(metric)] == pytest.approx(
            row[metric] - base[metric], nan_ok=True
        )
    assert row["base_orientation_reversed"] is False
    assert row["orientation_reversed"] is True
    assert row[fidelity.delta_column("pfs_capped")] < row[fidelity.delta_column("pfs")]


def test_the_view_of_a_variant_reads_as_the_published_groups_table() -> None:
    frame, published = _groups(), _groups(published=True)

    original = fidelity_orientation.variant_view(frame, "original")
    capped = fidelity_orientation.variant_view(frame, "capped")

    assert list(original.columns) == list(published.columns)
    pd.testing.assert_frame_equal(original, published)
    assert list(capped.columns) == list(published.columns)
    assert np.allclose(capped["pfs"], frame["pfs_capped"], equal_nan=True)
    assert np.allclose(capped["score_structure"], frame["score_structure_capped"], equal_nan=True)
    assert capped["binding_term"].equals(fidelity.binding_term(capped))
    reversed_row = _pooled(capped, "german")
    assert _pooled(published, "german")["pfs"] > 0.0
    assert reversed_row["pfs"] == 0.0
    assert reversed_row["pfs_with_center"] == 0.0
    assert reversed_row["binding_term"] == "structure"
    kept = _pooled(capped, "spanish-mx")
    assert kept["pfs_with_center"] == pytest.approx(
        fidelity.geometric_mean(
            [
                kept["score_accuracy"],
                kept["score_dispersion"],
                kept["score_structure"],
                kept["score_center"],
            ]
        )
    )
    with pytest.raises(ValueError, match="unknown variant"):
        fidelity_orientation.variant_view(frame, "mirrored")


def test_the_view_of_a_paired_variant_differences_its_own_scores() -> None:
    paired, published = _paired(), _paired(published=True)

    original = fidelity_orientation.variant_view(paired, "original")
    signed = fidelity_orientation.variant_view(paired, "signed")

    assert set(original.columns) == set(published.columns)
    pd.testing.assert_frame_equal(original[list(published.columns)], published)
    assert set(signed.columns) == set(published.columns)
    assert np.allclose(
        signed["delta_pfs_vs_base"], paired["delta_pfs_signed_vs_base"], equal_nan=True
    )
    assert np.allclose(signed["base_pfs"], paired["base_pfs_signed"], equal_nan=True)
    assert np.allclose(
        signed["delta_score_structure_vs_base"],
        signed["score_structure"] - signed["base_score_structure"],
        equal_nan=True,
    )
    assert signed["base_binding_term"].equals(fidelity.binding_term(signed, "base_"))
    assert fidelity_orientation.variant_view(paired.iloc[:0], "signed").empty


def test_the_summary_counts_what_each_variant_changes_on_the_first_run() -> None:
    frame = pd.concat([_groups(), _groups(replicate=2)], ignore_index=True)
    paired = pd.concat([_paired(), _paired(replicate=2)], ignore_index=True)

    summary = fidelity_orientation.orientation_summary(frame, paired)
    pooled = summary[summary["scope"].eq("population")].set_index("variant")

    assert list(pooled.index) == [variant.name for variant in fidelity_orientation.VARIANTS]
    assert set(summary["scope"]) == {"population", "subgroups", "home", "levels"}
    assert pooled["n_scores"].eq(len(ANSWERS)).all()
    every_level = frame[frame["replicate"].eq(1) & frame["group"].ne(fidelity.POPULATION)]
    assert summary[summary["scope"].eq("levels")]["n_scores"].eq(len(every_level)).all()
    assert pooled.loc["original", "structure_mean"] == pytest.approx(
        frame[frame["replicate"].eq(1) & frame["group"].eq(fidelity.POPULATION)][
            "score_structure"
        ].mean()
    )
    assert pooled.loc["original", "n_changed"] == 0
    assert pooled.loc["original", "rank_vs_original"] == pytest.approx(1.0)
    assert pooled["n_reversed"].eq(1).all()
    lowered, zeros, means = (
        pooled[column].to_dict() for column in ("n_lowered", "n_zero_pfs", "pfs_mean")
    )
    for name in ("gated", "capped", "signed"):
        assert int(lowered[name]) >= 1
        assert int(zeros[name]) == int(zeros["original"]) + 1
    assert pooled.loc["gated", "n_raised"] == pooled.loc["capped", "n_raised"] == 0
    assert float(means["capped"]) <= float(means["gated"])
    first = paired[paired["replicate"].eq(1) & paired["group"].eq(fidelity.POPULATION)]
    german = first[first["arm"].eq("german")]
    assert pooled.loc["capped", "german_n_paired"] == len(german) == 1
    assert pooled.loc["capped", "german_delta_pfs_mean"] == pytest.approx(
        german["delta_pfs_capped_vs_base"].mean()
    )
    assert pooled.loc["original", "german_delta_pfs_mean"] == pytest.approx(
        german["delta_pfs_vs_base"].mean()
    )
    assert pooled.loc["original", "german_sign_agreement"] == pytest.approx(1.0)
    assert pooled.loc["capped", "german_n_pfs_down"] == 1


def test_the_check_reconciles_the_published_tables_and_refuses_a_drifted_score() -> None:
    frame, paired = _groups(), _paired()
    published, published_paired = _groups(published=True), _paired(published=True)

    matched = fidelity_orientation.check_orientation(frame, paired, published, published_paired)

    assert matched == len(published) + len(published_paired)
    assert fidelity_orientation.check_orientation(
        frame, paired.iloc[:0], published, published_paired
    ) == len(published)
    drifted = published.assign(pfs=published["pfs"] + 0.01)
    with pytest.raises(AssertionError, match="pfs drifted"):
        fidelity_orientation.check_orientation(frame, paired, drifted, published_paired)
    with pytest.raises(AssertionError, match="other rows"):
        fidelity_orientation.check_orientation(
            frame[frame["level"].ne("Germany")], paired, published, published_paired
        )
    above = frame.assign(pfs_capped=frame["pfs_gated"] + 0.01)
    with pytest.raises(AssertionError, match="above the gate"):
        fidelity_orientation.check_orientation(above, paired, published, published_paired)


def test_each_plotted_variant_draws_the_paper_plates_in_its_own_folder(tmp_path: Path) -> None:
    frame = pd.concat([_groups(), _groups(replicate=2)], ignore_index=True)
    paired = pd.concat([_paired(), _paired(replicate=2)], ignore_index=True)

    written = fidelity_orientation.orientation_plates(frame, paired, tmp_path)

    assert {path.parent.name for path in written} == set(fidelity_orientation.PLATE_VARIANTS)
    for name in fidelity_orientation.PLATE_VARIANTS:
        assert {
            "fig_fidelity_components_happiness_ntp.png",
            "fig_fidelity_components_all_ntp.png",
            "fig_fidelity_components_home_happiness_ntp.png",
            "fig_fidelity_shift_happiness_ntp.png",
            "fig_fidelity_shift_home_happiness_ntp.png",
        } <= {path.name for path in written if path.parent.name == name}
    assert all(path.is_file() for path in written)


needs_shipped = pytest.mark.skipif(
    not (
        fidelity.GROUPS_TABLE.is_file() and fidelity_orientation.ORIENTATION_GROUPS_TABLE.is_file()
    ),
    reason="the shipped fidelity tables are not on disk",
)


@needs_shipped
def test_the_shipped_table_keeps_the_original_scores_beside_its_variants() -> None:
    shipped = pd.read_csv(fidelity_orientation.ORIENTATION_GROUPS_TABLE)
    groups = pd.read_csv(fidelity.GROUPS_TABLE)

    matched = fidelity_orientation._reconcile(
        shipped, groups, [*fidelity.IDENTITY, "group", "level"]
    )

    assert matched == len(groups) == len(shipped)
    first = shipped[shipped["replicate"].eq(1) & shipped["group"].eq(fidelity.POPULATION)]
    assert int(first["orientation_reversed"].sum()) == 19
    assert int(first["pfs"].eq(0.0).sum()) == 30
    assert int(first["pfs_gated"].eq(0.0).sum()) == 49
    assert int(first["pfs_capped"].eq(0.0).sum()) == 49
