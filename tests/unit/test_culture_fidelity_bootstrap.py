from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from culture import fidelity, fidelity_bootstrap
from culture.fidelity_bootstrap import Bootstrap, Cells
from culture.population import RunSource, run_identity
from machine_bias_reproduction.questions import resolve_question, resolve_questions

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

MODEL = np.array(
    [
        [0.35, 0.35, 0.2, 0.1],
        [0.15, 0.25, 0.3, 0.3],
        [0.2, 0.3, 0.3, 0.2],
        [0.6, 0.2, 0.1, 0.1],
        [0.2, 0.1, 0.2, 0.5],
        [0.25, 0.25, 0.25, 0.25],
    ]
)

ANSWERS = np.array([40.0, 20.0, 60.0, 20.0, 40.0, 80.0])

EXPECTED_ACCURACY = {"A": 0.946, "B": 0.942, "C": 0.982}

EXPECTED_RATIO = {"A": 0.24 / 0.33, "B": 0.345 / 0.33, "C": 0.315 / 0.33}

EXPECTED_STRUCTURE = {"A": 1.0 - 12.0 / 990.0, "B": 1.0 - 180.0 / 990.0, "C": 1.0 - 12.0 / 990.0}


def _reading(result: Bootstrap, label: str, name: str) -> float:
    return result.score(label, name)


SPREAD = np.array(
    [
        [0.9, 0.05, 0.03, 0.02],
        [0.02, 0.03, 0.05, 0.9],
        [0.25, 0.25, 0.25, 0.25],
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.7],
        [0.05, 0.05, 0.45, 0.45],
    ]
)

MANY = np.full(6, 1_000_000.0)


def test_shares_that_do_not_sum_to_one_are_refused() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        Cells(SURVEY * 0.6)


def test_a_redrawn_cell_needs_an_answer() -> None:
    with pytest.raises(ValueError, match="at least one answer"):
        Cells(SURVEY, np.array([40.0, 0.0, 60.0, 20.0, 40.0, 80.0]))
    with pytest.raises(ValueError, match="per cell"):
        Cells(SURVEY, np.array([40.0, 20.0]))


def test_a_case_keeps_its_seed_whatever_else_is_asked() -> None:
    first = fidelity_bootstrap.case_seed(7, "d_trust", "fa")
    again = fidelity_bootstrap.case_seed(7, "d_trust", "fa")
    other = fidelity_bootstrap.case_seed(7, "d_trust", "ntp")
    assert first.entropy == again.entropy
    assert first.entropy != other.entropy
    assert first.entropy != fidelity_bootstrap.case_seed(8, "d_trust", "fa").entropy


def test_a_child_seed_leaves_its_parent_untouched() -> None:
    parent = np.random.SeedSequence(11)
    first = fidelity_bootstrap.child_seed(parent, 3)
    again = fidelity_bootstrap.child_seed(parent, 3)
    assert parent.n_children_spawned == 0
    assert first.spawn_key == again.spawn_key == (3,)
    assert np.array_equal(first.generate_state(4), again.generate_state(4))
    assert not np.array_equal(
        first.generate_state(4), fidelity_bootstrap.child_seed(parent, 4).generate_state(4)
    )
    spawned = np.random.SeedSequence(11).spawn(4)[3]
    assert np.array_equal(first.generate_state(4), spawned.generate_state(4))


def test_the_survey_is_scored_as_the_survey() -> None:
    result = fidelity_bootstrap.paired_bootstrap(
        Cells(SURVEY, MANY), {"wide": Cells(SPREAD)}, replicates=5, seed=3
    )
    ratio = fidelity.group_fidelity(SURVEY, SPREAD)["adaptability_ratio"]
    assert ratio > 1.5
    assert result.reading("wide", "adaptability_ratio") == pytest.approx(ratio, rel=0.02)


def test_a_sampled_model_is_redrawn_and_probabilities_are_not() -> None:
    survey = Cells(SURVEY)
    fixed = fidelity_bootstrap.paired_bootstrap(survey, {"model": Cells(MODEL)}, 15, seed=4)
    drawn = fidelity_bootstrap.paired_bootstrap(
        survey, {"model": Cells(MODEL, ANSWERS)}, 15, seed=4
    )
    assert np.ptp(fixed.reading("model", "score_accuracy")) == 0.0
    assert np.ptp(drawn.reading("model", "score_accuracy")) > 0.0


def test_probabilities_are_never_redrawn() -> None:
    cells = Cells(MODEL)
    assert fidelity_bootstrap.redrawn(np.random.default_rng(3), cells) is MODEL


def test_a_redrawn_cell_keeps_the_number_of_its_answers() -> None:
    drawn = fidelity_bootstrap.redrawn(np.random.default_rng(3), Cells(SURVEY, ANSWERS))
    assert drawn.shape == SURVEY.shape
    assert np.allclose(drawn.sum(axis=1), 1.0)
    whole = drawn * ANSWERS[:, None]
    assert np.allclose(whole, np.round(whole))
    assert not np.allclose(drawn, SURVEY)


def test_a_seed_fixes_every_replicate() -> None:
    survey, models = Cells(SURVEY, ANSWERS), {"sampled": Cells(MODEL, ANSWERS)}
    first = fidelity_bootstrap.paired_bootstrap(survey, models, replicates=12, seed=5)
    again = fidelity_bootstrap.paired_bootstrap(survey, models, replicates=12, seed=5)
    other = fidelity_bootstrap.paired_bootstrap(survey, models, replicates=12, seed=6)
    assert np.array_equal(first.replicates["sampled"], again.replicates["sampled"], equal_nan=True)
    assert not np.array_equal(
        first.replicates["sampled"], other.replicates["sampled"], equal_nan=True
    )
    assert first.n_replicates == 12
    assert first.n_cells == 6


def test_the_observed_scores_are_those_of_group_fidelity() -> None:
    result = fidelity_bootstrap.paired_bootstrap(
        Cells(SURVEY, ANSWERS), {"model": Cells(MODEL)}, replicates=2, seed=1
    )
    reference = fidelity.group_fidelity(SURVEY, MODEL)
    for name in fidelity_bootstrap.READINGS:
        assert _reading(result, "model", name) == pytest.approx(reference[name], nan_ok=True)


def test_every_model_meets_the_same_redrawn_survey() -> None:
    models = {"one": Cells(MODEL), "other": Cells(MODEL)}
    result = fidelity_bootstrap.paired_bootstrap(
        Cells(SURVEY, ANSWERS), models, replicates=20, seed=2
    )
    assert np.array_equal(result.replicates["one"], result.replicates["other"], equal_nan=True)
    assert np.ptp(result.reading("one", "score_accuracy")) > 0.0
    rows = fidelity_bootstrap.difference_rows({"study": "test"}, result, (("one", "other"),))
    accuracy = next(row for row in rows if row["reading"] == "score_accuracy")
    assert accuracy["observed"] == 0.0
    assert accuracy["share_tied"] == 1.0
    assert accuracy["low"] == accuracy["high"] == 0.0


def test_the_interval_reaches_from_the_replicates_to_the_observed_value() -> None:
    values = np.arange(101, dtype=np.float64) / 100.0
    below = fidelity_bootstrap.spread(values, observed=0.8)
    assert below["replicate_mean"] == pytest.approx(0.5)
    assert below["shift"] == pytest.approx(-0.3)
    assert below["percentile_low"] == pytest.approx(0.025)
    assert below["percentile_high"] == pytest.approx(0.975)
    assert below["centered_low"] == pytest.approx(0.325)
    assert below["centered_high"] == pytest.approx(1.275)
    assert below["low"] == pytest.approx(0.025)
    assert below["high"] == pytest.approx(1.275)
    assert below["replicate_sd"] == pytest.approx(values.std(ddof=1))
    above = fidelity_bootstrap.spread(values, observed=0.4)
    assert above["shift"] == pytest.approx(0.1)
    assert above["low"] == pytest.approx(-0.075)
    assert above["high"] == pytest.approx(0.975)


def test_an_unshifted_interval_is_the_percentile_interval() -> None:
    values = np.arange(101, dtype=np.float64) / 100.0
    row = fidelity_bootstrap.spread(values, observed=0.5)
    assert row["low"] == pytest.approx(row["percentile_low"])
    assert row["high"] == pytest.approx(row["percentile_high"])
    assert row["centered_low"] == pytest.approx(row["percentile_low"])


def test_an_undefined_replicate_stays_out_of_the_spread() -> None:
    values = np.array([0.1, 0.2, np.nan, 0.3])
    row = fidelity_bootstrap.spread(values, observed=0.2)
    assert row["replicate_mean"] == pytest.approx(0.2)
    blank = fidelity_bootstrap.spread(np.array([np.nan, np.nan]), observed=0.2)
    assert np.isnan(blank["low"]) and np.isnan(blank["percentile_high"])
    assert blank["observed"] == 0.2


def test_the_differences_count_the_replicates_on_each_side_of_zero() -> None:
    columns = len(fidelity_bootstrap.READINGS)
    ahead = np.tile(np.array([0.3, 0.3, 0.3, 0.1]), (columns, 1)).T
    behind = np.tile(np.array([0.1, 0.3, 0.2, 0.2]), (columns, 1)).T
    result = Bootstrap(
        {"ahead": np.full(columns, 0.3), "behind": np.full(columns, 0.2)},
        {"ahead": ahead, "behind": behind},
        n_cells=5,
    )
    rows = fidelity_bootstrap.difference_rows({"study": "test"}, result, (("ahead", "behind"),))
    row = next(entry for entry in rows if entry["reading"] == "pfs")
    assert row["observed"] == pytest.approx(0.1)
    assert row["share_above"] == pytest.approx(0.5)
    assert row["share_tied"] == pytest.approx(0.25)
    assert row["share_below"] == pytest.approx(0.25)
    assert row["model"] == "ahead" and row["reference"] == "behind"
    assert len(rows) == columns


def _tied(share: float, observed: float = 0.3) -> dict[str, Any]:
    columns = len(fidelity_bootstrap.READINGS)
    count = 1000
    ties = round(share * count)
    ahead = np.concatenate([np.zeros(ties), np.linspace(0.25, 0.35, count - ties)])
    result = Bootstrap(
        {"ahead": np.full(columns, observed), "behind": np.zeros(columns)},
        {
            "ahead": np.tile(ahead, (columns, 1)).T,
            "behind": np.zeros((count, columns)),
        },
        n_cells=5,
    )
    rows = fidelity_bootstrap.difference_rows({}, result, (("ahead", "behind"),))
    return next(entry for entry in rows if entry["reading"] == "pfs")


def test_ties_in_one_replicate_in_forty_keep_the_models_together() -> None:
    row = _tied(0.10)
    assert row["share_tied"] == pytest.approx(0.10)
    assert row["percentile_low"] == 0.0
    assert row["centered_low"] > 0.0
    assert row["low"] == 0.0
    assert not row["separated"]


def test_a_few_ties_leave_an_interval_above_zero_separated() -> None:
    row = _tied(0.01)
    assert row["percentile_low"] > 0.0
    assert row["low"] > 0.0
    assert row["separated"]


def test_an_interval_across_zero_is_not_separated() -> None:
    row = _tied(0.0, observed=0.01)
    assert row["centered_low"] < 0.0 < row["centered_high"]
    assert row["percentile_low"] > 0.0
    assert row["low"] < 0.0 < row["high"]
    assert not row["separated"]


def test_a_printed_zero_carries_no_sign() -> None:
    assert fidelity_bootstrap.signed(0.0816) == "+0.082"
    assert fidelity_bootstrap.signed(-0.1281) == "-0.128"
    assert fidelity_bootstrap.signed(0.0) == "0.000"
    assert fidelity_bootstrap.signed(-0.0004) == "0.000"
    assert fidelity_bootstrap.signed(0.0004) == "0.000"


def test_the_synthetic_example_scores_as_worked_by_hand() -> None:
    result = fidelity_bootstrap.synthetic_bootstrap(replicates=2, seed=1)
    for label in fidelity_bootstrap.SYNTHETIC_MODELS:
        assert _reading(result, label, "score_accuracy") == pytest.approx(EXPECTED_ACCURACY[label])
        assert _reading(result, label, "adaptability_ratio") == pytest.approx(EXPECTED_RATIO[label])
        assert _reading(result, label, "score_structure") == pytest.approx(
            EXPECTED_STRUCTURE[label]
        )
        assert _reading(result, label, fidelity.ORIENTATION) == pytest.approx(1.0)
    assert _reading(result, "B", "pfs") > _reading(result, "A", "pfs")
    assert _reading(result, "C", "pfs") > _reading(result, "B", "pfs")


def test_no_two_distances_of_the_synthetic_example_tie() -> None:
    shares = {"survey": fidelity_bootstrap.SYNTHETIC_SURVEY, **fidelity_bootstrap.SYNTHETIC_MODELS}
    for values in shares.values():
        hundredths = np.round(np.asarray(values) * 100).astype(int)
        first, second = np.triu_indices(len(hundredths), k=1)
        gaps = np.abs(hundredths[first] - hundredths[second])
        assert len(set(gaps.tolist())) == len(gaps)


def test_the_synthetic_inputs_list_one_row_per_group() -> None:
    inputs = fidelity_bootstrap.synthetic_inputs()
    assert list(inputs.columns) == ["group", "survey", "A", "B", "C"]
    assert len(inputs) == len(fidelity_bootstrap.SYNTHETIC_GROUPS)


def test_more_responses_narrow_the_synthetic_interval() -> None:
    def width(responses: int) -> float:
        result = fidelity_bootstrap.synthetic_bootstrap(replicates=300, seed=4, responses=responses)
        rows = fidelity_bootstrap.difference_rows({}, result, (("C", "A"),))
        row = next(entry for entry in rows if entry["reading"] == "pfs")
        return float(row["high"] - row["low"])

    assert width(20_000) < width(500)


def test_the_calibration_draws_datasets_from_the_given_tables() -> None:
    drawn = fidelity_bootstrap.synthetic_calibration(datasets=5, replicates=20, seed=3, workers=1)
    again = fidelity_bootstrap.synthetic_calibration(datasets=5, replicates=20, seed=3, workers=1)
    pd.testing.assert_frame_equal(drawn, again)
    pairs = len(fidelity_bootstrap.SYNTHETIC_PAIRS)
    assert len(drawn) == 5 * pairs * len(fidelity_bootstrap.CALIBRATED)
    assert set(drawn["dataset"]) == set(range(5))
    result = fidelity_bootstrap.synthetic_bootstrap(replicates=2, seed=1)
    ahead = drawn[drawn["model"].eq("C") & drawn["reading"].eq("pfs")]
    assert ahead["population"].nunique() == 1
    assert ahead["population"].iloc[0] == pytest.approx(result.difference("C", "A", "pfs"))
    assert ahead["observed"].nunique() == 5
    assert ahead["n_replicates"].eq(20).all()


def test_scoring_one_list_of_datasets_twice_gives_the_same_rows() -> None:
    models = {
        label: fidelity_bootstrap.sampled_cells(shares, 500)
        for label, shares in fidelity_bootstrap.SYNTHETIC_MODELS.items()
    }
    tasks, populations = fidelity_bootstrap.calibration_tasks(
        fidelity_bootstrap.SYNTHETIC_IDENTITY,
        fidelity_bootstrap.sampled_cells(fidelity_bootstrap.SYNTHETIC_SURVEY, 500),
        models,
        fidelity_bootstrap.SYNTHETIC_PAIRS,
        3,
        10,
        np.random.SeedSequence(5),
    )
    first = fidelity_bootstrap.calibrated(tasks, populations, 1)
    again = fidelity_bootstrap.calibrated(tasks, populations, 1)
    pd.testing.assert_frame_equal(first, again)


def test_the_calibration_summary_counts_the_intervals_that_hold_the_expected_value() -> None:
    drawn = pd.DataFrame(
        {
            "study": "synthetic",
            "question": "synthetic",
            "question_label": "Synthetic binary question",
            "mode": "fa",
            "model": "C",
            "reference": "A",
            "reading": "pfs",
            "n_replicates": 20,
            "observed": [0.08, 0.10, 0.12, 0.10],
            "population": 0.09,
            "replicate_sd": 0.01,
            "shift": [-0.01, -0.01, -0.02, 0.0],
            "low": [0.06, 0.095, 0.11, 0.05],
            "high": [0.11, 0.12, 0.13, 0.15],
            "percentile_low": [0.01, 0.02, 0.03, -0.01],
            "percentile_high": [0.05, 0.095, 0.07, 0.12],
            "centered_low": [0.07, 0.08, 0.09, 0.10],
            "centered_high": [0.12, 0.13, 0.14, 0.15],
        }
    )
    row = fidelity_bootstrap.calibration_summary(drawn).iloc[0]
    assert row["n_datasets"] == 4
    assert row["expected"] == pytest.approx(0.10)
    assert row["interval_covers_expected"] == pytest.approx(0.75)
    assert row["interval_covers_population"] == pytest.approx(0.5)
    assert row["percentile_covers_expected"] == pytest.approx(0.25)
    assert row["percentile_covers_population"] == pytest.approx(0.5)
    assert row["interval_excludes_zero"] == pytest.approx(1.0)
    assert row["percentile_excludes_zero"] == pytest.approx(0.75)
    assert row["centered_covers_expected"] == pytest.approx(1.0)
    assert row["centered_covers_population"] == pytest.approx(0.75)
    assert row["shift_mean"] == pytest.approx(-0.01)
    assert row["interval_width_mean"] == pytest.approx(0.04875)


def test_the_response_files_name_both_tables_of_every_question() -> None:
    files = fidelity_bootstrap.response_files()
    questions = resolve_questions(None)
    assert len(files) == 2 * len(fidelity_bootstrap.RELEASED_MODELS) * len(questions)
    assert "data/culture/qwen3_vl_8b/base/d_happy/NTP-qwen3_vl_8b-base-d_happy.csv" in files
    assert "data/culture/gemma4_31b/base/d_trust/FA-gemma4_31b-base-d_trust.csv" in files
    assert all("*" not in name and "rep" not in name for name in files)


def _groups() -> pd.DataFrame:
    question = resolve_question("d_happy")
    rows: list[dict[str, Any]] = []
    for key, pfs in zip(fidelity_bootstrap.RELEASED_MODELS, (0.40, 0.10), strict=True):
        label = fidelity_bootstrap.model_label(key)
        for mode, replicate, shift in (("fa", 1, 0.0), ("fa", 2, 0.02), ("ntp", 1, 0.3)):
            source = f"culture/{key}/base" + ("" if replicate == 1 else f"/rep{replicate}")
            run = RunSource(f"{label} (as released)", source, key, label, "base", replicate)
            rows.append(
                {
                    **run_identity(run, question, mode),
                    "group": fidelity.POPULATION,
                    "level": fidelity.EVERY_CELL,
                    "n_cells": 6,
                    "pfs": pfs + shift,
                    "score_accuracy": 0.8 + pfs / 10,
                    "score_dispersion": 0.5,
                    "score_structure": pfs / 4,
                }
            )
    return pd.DataFrame(rows)


def test_the_run_differences_set_the_two_models_side_by_side_in_every_sampled_run() -> None:
    table = fidelity_bootstrap.run_differences(_groups())
    assert set(table["mode"]) == {"fa"}
    assert len(table) == 2 * len(fidelity_bootstrap.CALIBRATED)
    pfs = table[table["reading"].eq("pfs")]
    assert list(pfs["replicate"]) == [1, 2]
    assert pfs["difference"].tolist() == pytest.approx([0.30, 0.30])
    assert pfs["score_model"].tolist() == pytest.approx([0.40, 0.42])
    accuracy = table[table["reading"].eq("score_accuracy")]
    assert accuracy["difference"].tolist() == pytest.approx([0.03, 0.03])
    assert set(table["model"]) == {"Qwen3-VL-8B-Thinking"}
    assert set(table["reference"]) == {"Gemma-4-31B-it"}


def _levels(shift: float = 0.0) -> pd.DataFrame:
    groups = _groups()
    first = groups[groups["replicate"].eq(1)]
    rows = [
        {
            "study": fidelity_bootstrap.SURVEYED,
            "question": row["question"],
            "mode": row["mode"],
            "model": row["model_label"],
            "reading": name,
            "n_cells": row["n_cells"],
            "observed": row.get(name, np.nan) + (shift if name == "pfs" else 0.0),
            "low": 0.0,
            "high": 1.0,
        }
        for _, row in first.iterrows()
        for name in fidelity_bootstrap.READINGS
    ]
    return pd.DataFrame(rows)


def test_the_check_reconciles_the_observed_scores_with_the_groups_table() -> None:
    groups = _groups().assign(
        score_center=0.9, adaptability_ratio=0.5, rho_structure=0.1, rho_orientation=0.2
    )
    levels = _levels()
    levels["observed"] = [
        groups[
            groups["model_label"].eq(row["model"])
            & groups["mode"].eq(row["mode"])
            & groups["replicate"].eq(1)
        ][row["reading"]].iloc[0]
        for _, row in levels.iterrows()
    ]
    assert fidelity_bootstrap.check_bootstrap(levels, groups) == len(levels)
    levels.loc[levels["reading"].eq("pfs"), "observed"] += 0.01
    with pytest.raises(AssertionError, match="pfs drifted"):
        fidelity_bootstrap.check_bootstrap(levels, groups)


@pytest.mark.integration
def test_a_released_run_reads_the_tables_of_one_model(tmp_path: Path) -> None:
    from machine_bias_reproduction.data import archived_fa, archived_ntp, prepare_data

    question = resolve_question("d_trust")
    key = fidelity_bootstrap.RELEASED_MODELS[0]
    folder = tmp_path / "culture" / key / "base" / question.var
    folder.mkdir(parents=True)
    archived_ntp(question).to_csv(folder / f"NTP-{key}-base-{question.var}.csv", index=False)
    archived_fa(question).to_csv(folder / f"FA-{key}-base-{question.var}.csv", index=False)
    released = fidelity_bootstrap.released_run(tmp_path, key, question)
    reference = prepare_data(archived_ntp(question), archived_fa(question), question)
    assert released.names.equals(reference.names)
    sampled = fidelity_bootstrap.model_cells(released, "fa", released.names)
    assert sampled.answers is not None and sampled.answers.min() >= 20
    assert reference.fa_props is not None
    assert np.allclose(sampled.shares, reference.fa_props.to_numpy(dtype=np.float64))
    assert fidelity_bootstrap.model_cells(released, "ntp", released.names).answers is None
    surveyed = fidelity_bootstrap.surveyed_cells(released, released.names)
    assert surveyed.answers is not None
    whole = surveyed.shares * surveyed.answers[:, None]
    assert np.allclose(whole, np.round(whole), atol=1e-6)
    shares, counts = fidelity_bootstrap.survey_cells(question)
    assert np.allclose(shares.loc[released.names].to_numpy(dtype=np.float64), surveyed.shares)
    assert np.array_equal(counts.loc[released.names].to_numpy(), surveyed.answers)
