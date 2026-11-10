from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from culture import fidelity, fidelity_baseline, fidelity_objectives
from culture.population import RunSource, run_identity
from machine_bias_reproduction.questions import resolve_question

MODEL_KEY, OTHER_KEY = fidelity_objectives.OBJECTIVE_MODELS[:2]

EVERY = fidelity_baseline.EVERY

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

FLATTENED = np.tile(np.array([0.3, 0.3, 0.2, 0.2]), (len(CELLS), 1))

ANSWERS: dict[str, np.ndarray] = {
    "base": RELEASED,
    "german": FLATTENED,
    "spanish-mx": 0.5 * (RELEASED + FLATTENED),
    "global": 0.5 * (RELEASED + SURVEY),
    "subpop": 0.25 * RELEASED + 0.75 * SURVEY,
}


def _run(key: str | None, arm: str | None, replicate: int = 1) -> RunSource:
    label = key or "Mixtral archived"
    source = f"culture/{key}/{arm}" if key else "archived"
    if replicate > 1:
        source = f"{source}/rep{replicate}"
    return RunSource(f"{label} ({arm})", source, key, label, arm, replicate)


def _facet() -> pd.DataFrame:
    return fidelity.cell_facets(CELLS).reindex(list(CELLS))


def _groups(
    mode: str = "ntp", replicate: int = 1, nudge: float = 0.0, key: str = MODEL_KEY
) -> pd.DataFrame:
    question = resolve_question("d_happy")
    rows: list[dict[str, Any]] = []
    for arm, answers in ANSWERS.items():
        identity = run_identity(_run(key, arm, replicate), question, mode)
        moved = (1.0 - nudge) * answers + nudge * SURVEY
        rows += fidelity.grouped_scores(identity, _facet(), fidelity.group_fidelity, SURVEY, moved)
    frame = pd.DataFrame(rows)
    return frame.assign(binding_term=fidelity.binding_term(frame))


def _paired(key: str = MODEL_KEY) -> pd.DataFrame:
    question = resolve_question("d_happy")
    rows: list[dict[str, Any]] = []
    for arm, answers in ANSWERS.items():
        if arm == "base":
            continue
        identity = run_identity(_run(key, arm), question, "ntp")
        rows += fidelity.grouped_scores(
            identity, _facet(), fidelity.paired_fidelity, SURVEY, answers, RELEASED
        )
    return pd.DataFrame(rows)


def _baseline(key: str = MODEL_KEY) -> pd.DataFrame:
    question = resolve_question("d_happy")
    rows: list[dict[str, Any]] = []
    for arm, answers in ANSWERS.items():
        identity = run_identity(_run(key, arm), question, "ntp")
        rows += fidelity.grouped_scores(
            identity, _facet(), fidelity_baseline.baseline_readings, SURVEY, answers
        )
    return pd.DataFrame(rows)


def _scope(summary: pd.DataFrame, scope: str, key: str = MODEL_KEY) -> pd.DataFrame:
    return summary[summary["scope"].eq(scope) & summary["model_key"].eq(key)].set_index("arm")


def _both(build: Callable[..., pd.DataFrame], **nudged: Any) -> pd.DataFrame:
    return pd.concat([build(key=MODEL_KEY), build(key=OTHER_KEY, **nudged)], ignore_index=True)


def test_the_comparison_keeps_every_replicate_of_its_models_and_variants() -> None:
    found = [
        _run(MODEL_KEY, "base"),
        _run(MODEL_KEY, "german"),
        _run(MODEL_KEY, "spanish"),
        _run(MODEL_KEY, "global"),
        _run(MODEL_KEY, "global", replicate=3),
        _run("terra", "base"),
        _run(None, None),
    ]

    kept = fidelity_objectives.objective_runs(found)

    assert [(run.arm, run.replicate) for run in kept] == [
        ("base", 1),
        ("german", 1),
        ("global", 1),
        ("global", 3),
    ]
    assert [run.arm for run in fidelity_objectives.first_runs(kept)] == ["base", "german", "global"]


def test_the_summary_reads_each_variant_beside_its_change_from_the_released_model() -> None:
    summary = fidelity_objectives.objective_summary(_groups(), _paired(), _baseline())
    pooled = _scope(summary, fidelity.POPULATION)

    assert list(pooled.index) == list(fidelity_objectives.OBJECTIVE_ARMS)
    assert pooled["n_scores"].eq(1).all()
    assert pd.isna(pooled["n_paired"]["base"])
    assert pooled["adaptability_ratio_median"]["german"] == 0.0
    assert pooled["n_pfs_down"]["german"] == 1
    assert pooled["n_pfs_up"]["global"] == 1
    assert pooled["n_center_up"]["global"] == 1
    accuracy = pooled["score_accuracy_median"]
    gain = float(pooled[fidelity_objectives.change_column("score_accuracy")]["global"])
    assert gain == pytest.approx(float(accuracy["global"]) - float(accuracy["base"]))


def test_the_summary_closes_each_scope_with_the_models_pooled() -> None:
    groups = _both(_groups, nudge=0.5)
    summary = fidelity_objectives.objective_summary(groups, _both(_paired), _both(_baseline))
    population = summary[summary["scope"].eq(fidelity.POPULATION)]
    first, second, pooled = (
        _scope(summary, fidelity.POPULATION, key) for key in (MODEL_KEY, OTHER_KEY, EVERY)
    )

    assert list(dict.fromkeys(population["model_key"])) == [MODEL_KEY, OTHER_KEY, EVERY]
    assert list(pooled.index) == list(fidelity_objectives.OBJECTIVE_ARMS)
    assert pooled["model_label"].eq(EVERY).all()
    assert pooled["series"]["global"] == f"{EVERY} (distribution-matched)"
    assert pooled["n_scores"].eq(2).all()
    assert pooled["n_paired"]["subpop"] == 2
    assert pooled["n_pfs_up"]["global"] == 2
    assert pd.isna(pooled["n_paired"]["base"])
    levels = pd.concat([first["pfs_median"], second["pfs_median"]], axis=1)
    assert first["pfs_median"]["subpop"] != second["pfs_median"]["subpop"]
    assert pooled["pfs_min"].to_numpy() == pytest.approx(levels.min(axis=1).to_numpy())
    assert pooled["pfs_max"].to_numpy() == pytest.approx(levels.max(axis=1).to_numpy())
    assert pooled["pfs_median"].to_numpy() == pytest.approx(levels.mean(axis=1).to_numpy())


def test_the_summary_counts_the_subgroup_scores_apart_from_the_pooled_ones() -> None:
    groups = _groups()
    summary = fidelity_objectives.objective_summary(groups, _paired(), _baseline())
    subgroups = _scope(summary, fidelity_objectives.SUBGROUPS)

    levels = groups[~groups["group"].eq(fidelity.POPULATION) & groups["arm"].eq("base")]
    assert subgroups["n_scores"]["base"] == len(levels)
    assert subgroups["n_paired"]["global"] == len(levels)
    assert subgroups["pfs_median"]["base"] == pytest.approx(levels["pfs"].median())
    assert subgroups["score_center_min"]["base"] == pytest.approx(levels["score_center"].min())
    assert subgroups["score_center_max"]["base"] == pytest.approx(levels["score_center"].max())


def test_the_summary_reads_levels_by_their_median_and_changes_by_their_mean() -> None:
    groups = _groups()
    paired = _paired()
    summary = fidelity_objectives.objective_summary(groups, paired, _baseline())
    subgroups = _scope(summary, fidelity_objectives.SUBGROUPS)

    shifts = paired[~paired["group"].eq(fidelity.POPULATION) & paired["arm"].eq("global")]
    change = fidelity.delta_column("score_center")
    assert subgroups[fidelity_objectives.change_column("score_center")]["global"] == pytest.approx(
        shifts[change].mean()
    )
    assert "score_center" not in summary.columns
    assert change not in summary.columns


def test_the_summary_scores_each_target_country_on_its_own_cells() -> None:
    groups = _groups()
    summary = fidelity_objectives.objective_summary(groups, _paired(), _baseline())

    assert list(dict.fromkeys(summary["scope"])) == list(fidelity_objectives.SCOPES)
    assert fidelity_objectives.HOME_COUNTRIES == ("Germany", "Mexico", "United States")
    germany = _scope(summary, "Germany")
    assert list(germany.index) == list(fidelity_objectives.OBJECTIVE_ARMS)
    assert germany["n_scores"].eq(1).all()
    inside = fidelity_objectives.within_country(groups, "Germany")
    assert set(inside["level"]) == {"Germany"}
    expected = float(inside[inside["arm"].eq("base")]["score_center"].iloc[0])
    assert germany["score_center_median"]["base"] == pytest.approx(expected)
    states = _scope(summary, "United States")
    assert list(states.index) == list(fidelity_objectives.OBJECTIVE_ARMS)
    assert set(fidelity_objectives.within_country(groups, "United States")["level"]) == {
        "United States"
    }


def test_each_block_scores_the_variants_that_belong_on_its_cells(tmp_path: Any) -> None:
    groups = _groups()
    levels = groups.assign(n_replicates=1)

    assert fidelity_objectives.BLOCK_SCOPES == (
        fidelity.POPULATION,
        "Germany",
        "Mexico",
        "United States",
    )
    assert fidelity_objectives.block_arms(fidelity.POPULATION) == fidelity_objectives.OBJECTIVE_ARMS
    assert fidelity_objectives.block_arms("Germany") == ("base", "german", "global")
    assert fidelity_objectives.block_arms("Mexico") == ("base", "spanish-mx", "global")
    assert fidelity_objectives.block_arms("United States") == ("base", "global", "subpop")
    germany = fidelity_objectives.block_rows(levels, "Germany")
    assert germany["arm"].tolist() == ["base", "german", "global"]
    assert set(germany["level"]) == {"Germany"}
    states = fidelity_objectives.block_rows(levels, "United States")
    assert states["arm"].tolist() == ["base", "global", "subpop"]
    assert set(states["level"]) == {"United States"}
    pooled = fidelity_objectives.block_rows(levels, fidelity.POPULATION)
    assert pooled["arm"].tolist() == list(fidelity_objectives.OBJECTIVE_ARMS)

    files = fidelity_objectives.scoped_components_plate(levels, "ntp", "happiness", tmp_path)

    assert {path.name for path in files} == {
        "fig_fidelity_components_scoped_happiness_ntp.pdf",
        "fig_fidelity_components_scoped_happiness_ntp.png",
    }
    assert (
        fidelity_objectives.scoped_components_plate(levels.iloc[:0], "ntp", "happiness", tmp_path)
        == []
    )


def test_each_model_is_drawn_into_its_own_folder(tmp_path: Any) -> None:
    manuscript = fidelity_objectives.MANUSCRIPT_MODEL
    other = next(key for key in fidelity_objectives.OBJECTIVE_MODELS if key != manuscript)
    groups = pd.concat([_groups(key=key) for key in (manuscript, other)], ignore_index=True)
    paired = pd.concat([_paired(key=key) for key in (manuscript, other)], ignore_index=True)

    assert fidelity_objectives.plate_folder(manuscript, tmp_path) == tmp_path
    assert fidelity_objectives.plate_folder(other, tmp_path) == tmp_path / other

    files = fidelity_objectives.objective_plates(groups, paired, tmp_path)

    stems = (
        "fig_fidelity_components_scoped_happiness_ntp",
        "fig_fidelity_components_happiness_ntp",
        "fig_fidelity_shift_happiness_ntp",
    )
    assert set(files) == {
        folder / f"{stem}.{suffix}"
        for folder in (tmp_path, tmp_path / other)
        for stem in stems
        for suffix in ("pdf", "png")
    }


def test_the_check_reconciles_the_published_variants_and_refuses_a_drifted_score() -> None:
    groups, paired = _groups(), _paired()
    published_groups = groups[groups["arm"].isin(fidelity_baseline.MANUSCRIPT_ARMS)]
    published_paired = paired[paired["arm"].isin(fidelity_baseline.MANUSCRIPT_ARMS)]

    matched = fidelity_objectives.check_objectives(
        groups, paired, published_groups, published_paired
    )

    assert matched == len(published_groups) + len(published_paired)
    with pytest.raises(AssertionError, match="missing"):
        fidelity_objectives.check_objectives(
            groups[groups["arm"].ne("subpop")],
            paired[paired["arm"].ne("subpop")],
            published_groups,
            published_paired,
        )
    drifted = published_groups.assign(pfs=published_groups["pfs"] + 0.01)
    with pytest.raises(AssertionError, match="drifted"):
        fidelity_objectives.check_objectives(groups, paired, drifted, published_paired)
    with pytest.raises(AssertionError, match="missing"):
        fidelity_objectives.check_objectives(
            published_groups, paired, published_groups, published_paired
        )


def test_the_check_refuses_a_model_short_of_a_variant_and_an_unpaired_run() -> None:
    groups, paired = _both(_groups), _both(_paired)
    short = groups["model_key"].eq(OTHER_KEY) & groups["arm"].eq("global")

    fidelity_objectives.check_variants(groups, paired)
    with pytest.raises(AssertionError, match=f"missing for {OTHER_KEY}"):
        fidelity_objectives.check_variants(groups[~short], paired)
    unpaired = pd.concat([groups, _groups(mode="fa", replicate=3)], ignore_index=True)
    with pytest.raises(AssertionError, match="no as-released run to pair with"):
        fidelity_objectives.check_variants(unpaired, paired)
