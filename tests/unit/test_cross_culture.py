from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

from culture import cross_culture
from culture.cross_culture import (
    DEFAULT_PAIR,
    MIN_CELLS_FOR_TEST,
    SURVEY,
    complete_matrix,
    culture_cross,
    head_to_head_cells,
    head_to_head_summary,
    league_table,
    pairwise_matrix,
    pairwise_rows,
    pooled_matrix,
    shared_countries,
)
from culture.cross_culture_panels import (
    culture_map,
    head_to_head_figure,
    league_figure,
    pairwise_heatmap,
)
from culture.registry import CULTURE_MODELS, CultureModel
from machine_bias_reproduction.data import Coverage, PreparedData
from machine_bias_reproduction.io_utils import write_csv
from machine_bias_reproduction.metrics import nemd
from machine_bias_reproduction.questions import Question, resolve_question

matplotlib.use("Agg")

QUESTION = resolve_question("d_happy")
COLUMNS = list(QUESTION.answer_columns)

SUBPOPULATIONS = (
    "Germany 1997 Female 25-34 High Student Cohabiting",
    "Germany 2006 Male 35-44 Low Working Married",
    "Mexico 2005 Male 45-54 Low Working Married",
    "Mexico 2018 Female 25-34 High Working Single",
    "United States 1995 Male 35-44 Low Working Single",
)

LEAGUE_COLUMNS = [
    "model_key",
    "model_label",
    "question",
    "culture",
    "countries",
    "mode",
    "subpopulations",
    "mean_nEMD",
    "median_nEMD",
    "mean_EMD",
    "mean_KL",
    "mean_JS",
    "mean_MMD",
    "base_mean_nEMD",
    "delta_vs_base",
    "mixtral_mean_nEMD",
    "delta_vs_mixtral",
    "rank",
]

CELL_COLUMNS = [
    "model_key",
    "model_label",
    "question",
    "culture_a",
    "culture_b",
    "countries",
    "mode",
    "subpopulation",
    "nEMD_a",
    "nEMD_b",
    "delta_nEMD",
]


def _props(names: list[str], seed: int) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    values = generator.random((len(names), len(COLUMNS))) + 0.1
    values /= values.sum(axis=1, keepdims=True)
    return pd.DataFrame(values, index=pd.Index(names, name="subpop"), columns=COLUMNS)


def _prepared(names: list[str], seed: int) -> PreparedData:
    empty = pd.DataFrame()
    return PreparedData(
        question=QUESTION,
        wvs=empty,
        subpops=empty,
        ntp_raw=empty,
        fa_raw=empty,
        names=pd.Index(names),
        wvs_props=_props(names, 0),
        ntp_props=_props(names, seed),
        fa_props=_props(names, seed + 100),
        social_predictors=empty,
        coverage=Coverage(0, 0, 0, 0, len(names), len(names)),
    )


def _distances(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["method", "subpopulation", "nEMD"])


def _matched_row(culture: str, countries: str, mode: str, mean: float) -> dict[str, object]:
    return {
        "model_key": "gemma4_31b",
        "model_label": "Gemma-4-31B-it",
        "culture": culture,
        "countries": countries,
        "mode": mode,
        "subpopulations": 2,
        "mean_nEMD": mean,
        "median_nEMD": mean,
        "mean_EMD": 0.5,
        "mean_KL": 1.0,
        "mean_JS": 0.3,
        "mean_MMD": 0.4,
    }


def _seed_tree(root: Path, *, base: bool = True) -> None:
    model_root = root / "culture" / "gemma4_31b"
    if base:
        write_csv(
            _distances(
                [
                    ("NTP", SUBPOPULATIONS[0], 0.30),
                    ("NTP", SUBPOPULATIONS[1], 0.10),
                    ("NTP", SUBPOPULATIONS[2], 0.50),
                    ("NTP", "Germany 2010 Female 35-44 High Working Married", 0.90),
                    ("FA", SUBPOPULATIONS[0], 0.40),
                    ("FA", SUBPOPULATIONS[1], 0.20),
                    ("FA", SUBPOPULATIONS[2], 0.60),
                    ("FA", "Germany 2010 Female 35-44 High Working Married", 0.90),
                ]
            ),
            model_root / "base" / "d_happy" / "subpopulation_distances.csv",
        )
    write_csv(
        _distances(
            [
                ("NTP", SUBPOPULATIONS[0], 0.20),
                ("NTP", SUBPOPULATIONS[1], 0.16),
                ("FA", SUBPOPULATIONS[0], 0.30),
                ("FA", SUBPOPULATIONS[1], 0.26),
            ]
        ),
        model_root / "german" / "d_happy" / "subpopulation_distances.csv",
    )
    write_csv(
        _distances(
            [
                ("NTP", SUBPOPULATIONS[2], 0.30),
                ("NTP", SUBPOPULATIONS[3], 0.20),
                ("NTP", "Mexico 1995 Male 35-44 Low Working Single", 0.10),
                ("NTP", SUBPOPULATIONS[0], 0.50),
                ("FA", SUBPOPULATIONS[2], 0.35),
                ("FA", SUBPOPULATIONS[3], 0.25),
                ("FA", "Mexico 1995 Male 35-44 Low Working Single", 0.15),
                ("FA", SUBPOPULATIONS[0], 0.55),
                ("Linear", SUBPOPULATIONS[2], 0.99),
                ("Random_01", SUBPOPULATIONS[3], 0.98),
            ]
        ),
        model_root / "spanish" / "d_happy" / "subpopulation_distances.csv",
    )
    write_csv(
        _distances(
            [
                ("NTP", SUBPOPULATIONS[2], 0.25),
                ("NTP", SUBPOPULATIONS[3], 0.22),
                ("NTP", "Mexico 2010 Female 55-64 High Retired Widowed", 0.40),
                ("NTP", SUBPOPULATIONS[0], 0.45),
                ("FA", SUBPOPULATIONS[2], 0.30),
                ("FA", SUBPOPULATIONS[3], 0.20),
                ("FA", "Mexico 2010 Female 55-64 High Retired Widowed", 0.45),
                ("FA", SUBPOPULATIONS[0], 0.50),
                ("Linear", SUBPOPULATIONS[2], 0.97),
                ("Random_01", SUBPOPULATIONS[3], 0.96),
            ]
        ),
        model_root / "spanish-mx" / "d_happy" / "subpopulation_distances.csv",
    )
    write_csv(
        pd.DataFrame(
            [
                _matched_row("german", "Germany", "NTP", 0.18),
                _matched_row("german", "Germany", "FA", 0.28),
                _matched_row("spanish-mx", "Mexico", "NTP", 0.22),
                _matched_row("spanish-mx", "Mexico", "FA", 0.35),
            ]
        ),
        model_root / "d_happy" / "culture_matched_comparison.csv",
    )
    write_csv(
        _distances(
            [
                ("NTP", SUBPOPULATIONS[0], 0.26),
                ("NTP", SUBPOPULATIONS[1], 0.22),
                ("NTP", SUBPOPULATIONS[2], 0.44),
                ("NTP", "Germany 2010 Female 35-44 High Working Married", 0.90),
                ("FA", SUBPOPULATIONS[0], 0.36),
                ("FA", SUBPOPULATIONS[1], 0.28),
                ("FA", SUBPOPULATIONS[2], 0.55),
                ("FA", "Germany 2010 Female 35-44 High Working Married", 0.90),
            ]
        ),
        root / "archived" / "d_happy" / "subpopulation_distances.csv",
    )


def _square(seed: int, entities: list[str]) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    values = generator.random((len(entities), len(entities))) * 0.3 + 0.05
    values = (values + values.T) / 2
    np.fill_diagonal(values, 0.0)
    return pd.DataFrame(values, index=entities, columns=entities)


def test_pairwise_matrix_is_symmetric_with_zero_diagonal() -> None:
    names = list(SUBPOPULATIONS)
    means, counts = pairwise_matrix({"base": _props(names, 5), "german": _props(names, 5)})

    assert means.loc["base", "german"] == pytest.approx(0.0)
    assert means.equals(means.T)
    assert means.loc["base", "base"] == 0.0
    assert means.loc["german", "german"] == 0.0
    assert counts.loc["base", "base"] == len(names)
    assert counts.loc["base", "german"] == len(names)


def test_pairwise_matrix_measures_a_known_shift() -> None:
    one_hot_a = pd.DataFrame([[1.0, 0.0, 0.0, 0.0]], index=pd.Index(["cell"]), columns=COLUMNS)
    one_hot_b = pd.DataFrame([[0.0, 1.0, 0.0, 0.0]], index=pd.Index(["cell"]), columns=COLUMNS)

    means, counts = pairwise_matrix({"base": one_hot_a, "german": one_hot_b})

    assert means.loc["base", "german"] == pytest.approx(1 / 3)
    assert counts.loc["base", "german"] == 1


def test_pairwise_matrix_aligns_on_shared_names_only() -> None:
    shared = list(SUBPOPULATIONS[:3])
    wide = _props(list(SUBPOPULATIONS), 7)
    narrow = _props(shared, 8)

    means, counts = pairwise_matrix({"base": wide, "german": narrow})

    expected = float(np.mean(nemd(wide.loc[shared].to_numpy(), narrow.to_numpy())))
    assert means.loc["base", "german"] == pytest.approx(expected)
    assert counts.loc["base", "german"] == len(shared)


def test_pairwise_matrix_marks_disjoint_arms_missing() -> None:
    means, counts = pairwise_matrix(
        {
            "base": _props(list(SUBPOPULATIONS[:2]), 1),
            "german": _props(list(SUBPOPULATIONS[2:]), 2),
        }
    )

    assert np.isnan(means.loc["base", "german"])
    assert counts.loc["base", "german"] == 0


def test_pairwise_rows_keeps_survey_anchor_and_drops_missing_pairs() -> None:
    survey = _props(list(SUBPOPULATIONS), 0)
    base = _props(list(SUBPOPULATIONS[:3]), 1)
    korean = _props(["Russia 2011 Female 55-64 High Retired Widowed"], 2)
    means, counts = pairwise_matrix({SURVEY: survey, "base": base, "korean": korean})

    rows = pairwise_rows("gemma4_31b", "Gemma-4-31B-it", QUESTION, "NTP", means, counts)

    assert list(rows.columns) == [
        "model_key",
        "model_label",
        "question",
        "mode",
        "culture_a",
        "culture_b",
        "subpopulations",
        "mean_nEMD",
    ]
    assert set(rows["question"]) == {"d_happy"}
    survey_rows = rows[rows["culture_a"] == SURVEY]
    assert list(survey_rows["culture_b"]) == ["base"]
    expected = float(np.mean(nemd(survey.loc[base.index].to_numpy(), base.to_numpy())))
    assert survey_rows["mean_nEMD"].iloc[0] == pytest.approx(expected)
    assert not ((rows["culture_a"] == "korean") | (rows["culture_b"] == "korean")).any()


def test_pooled_matrix_averages_models_with_different_arms() -> None:
    first = pd.DataFrame(
        [[0.0, 0.2, 0.3], [0.2, 0.0, 0.4], [0.3, 0.4, 0.0]],
        index=[SURVEY, "base", "german"],
        columns=[SURVEY, "base", "german"],
    )
    second = pd.DataFrame(
        [[0.0, 0.4, 0.5], [0.4, 0.0, 0.6], [0.5, 0.6, 0.0]],
        index=[SURVEY, "base", "spanish-mx"],
        columns=[SURVEY, "base", "spanish-mx"],
    )

    pooled = pooled_matrix([first, second])

    assert list(pooled.index) == [SURVEY, "base", "german", "spanish-mx"]
    assert pooled.loc[SURVEY, "base"] == pytest.approx(0.3)
    assert pooled.loc["base", "german"] == pytest.approx(0.4)
    assert np.isnan(pooled.loc["german", "spanish-mx"])
    assert pooled.loc["german", "german"] == 0.0


def test_complete_matrix_drops_the_sparsest_entity() -> None:
    entities = [SURVEY, "base", "german", "korean"]
    values = np.array(
        [
            [0.0, 0.2, 0.3, np.nan],
            [0.2, 0.0, 0.4, np.nan],
            [0.3, 0.4, 0.0, np.nan],
            [np.nan, np.nan, np.nan, 0.0],
        ]
    )

    completed = complete_matrix(pd.DataFrame(values, index=entities, columns=entities))

    assert list(completed.index) == [SURVEY, "base", "german"]
    assert np.isfinite(completed.to_numpy()).all()


def test_league_table_adds_reference_deltas_and_rank(tmp_path: Path) -> None:
    _seed_tree(tmp_path)

    league = league_table([CULTURE_MODELS["gemma4_31b"]], QUESTION, outputs_root=tmp_path)

    assert list(league.columns) == LEAGUE_COLUMNS
    assert league["mean_nEMD"].tolist() == [0.28, 0.35, 0.18, 0.22]
    ntp_german = league[(league["mode"] == "NTP") & (league["culture"] == "german")].iloc[0]
    assert ntp_german["base_mean_nEMD"] == pytest.approx(0.20)
    assert ntp_german["delta_vs_base"] == pytest.approx(0.18 - 0.20)
    assert ntp_german["mixtral_mean_nEMD"] == pytest.approx(0.24)
    assert ntp_german["delta_vs_mixtral"] == pytest.approx(0.18 - 0.24)
    assert ntp_german["rank"] == 1
    ntp_mx = league[(league["mode"] == "NTP") & (league["culture"] == "spanish-mx")].iloc[0]
    assert ntp_mx["base_mean_nEMD"] == pytest.approx(0.50)
    assert ntp_mx["mixtral_mean_nEMD"] == pytest.approx(0.44)
    assert ntp_mx["rank"] == 2


def test_league_table_survives_a_missing_base_run(tmp_path: Path) -> None:
    _seed_tree(tmp_path, base=False)

    league = league_table([CULTURE_MODELS["gemma4_31b"]], QUESTION, outputs_root=tmp_path)

    assert not league.empty
    assert league["base_mean_nEMD"].isna().all()
    assert league["delta_vs_base"].isna().all()
    assert np.isfinite(league["mixtral_mean_nEMD"]).all()


def test_head_to_head_pairs_shared_cells_only(tmp_path: Path) -> None:
    _seed_tree(tmp_path)

    cells = head_to_head_cells(
        CULTURE_MODELS["gemma4_31b"], DEFAULT_PAIR, QUESTION, outputs_root=tmp_path
    )

    assert list(cells.columns) == CELL_COLUMNS
    assert len(cells) == 4
    assert set(cells["mode"]) == {"NTP", "FA"}
    for mode in ("NTP", "FA"):
        assert len(cells[cells["mode"] == mode]) == 2
    assert (cells["delta_nEMD"] == cells["nEMD_b"] - cells["nEMD_a"]).all()
    assert set(cells["countries"]) == {"Mexico"}

    summary = head_to_head_summary(cells)
    assert MIN_CELLS_FOR_TEST > 2
    ntp = summary[summary["mode"] == "NTP"].iloc[0]
    assert ntp["subpopulations"] == 2
    assert ntp["share_b_closer"] == pytest.approx(0.5)
    fa = summary[summary["mode"] == "FA"].iloc[0]
    assert fa["share_b_closer"] == pytest.approx(1.0)
    assert summary["wilcoxon_p"].isna().all()


def test_head_to_head_no_ops_or_raises_cleanly(tmp_path: Path) -> None:
    empty = head_to_head_cells(
        CULTURE_MODELS["gemma4_31b"], DEFAULT_PAIR, QUESTION, outputs_root=tmp_path
    )
    assert empty.empty
    assert head_to_head_figure(empty, tmp_path) == []
    assert shared_countries(("arabic", "german")) == ()
    unmatched = head_to_head_cells(
        CULTURE_MODELS["gemma4_31b"], ("arabic", "german"), QUESTION, outputs_root=tmp_path
    )
    assert unmatched.empty
    with pytest.raises(ValueError, match="unknown cultures"):
        shared_countries(("klingon", "german"))


def test_report_no_ops_on_an_empty_tree(tmp_path: Path) -> None:
    payload = culture_cross(
        [CULTURE_MODELS["gemma4_31b"]],
        ["base", "german"],
        "d_happy",
        outputs_root=tmp_path,
        tables_root=tmp_path,
        figures_root=tmp_path / "figs",
    )

    assert payload["figures"] == []
    assert payload["tables"] == []
    assert payload["pair"] == {"cultures": ["spanish", "spanish-mx"], "countries": ["Mexico"]}
    assert not (tmp_path / "figs").exists()


def test_report_validates_the_pair_before_any_work(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown cultures"):
        culture_cross(
            [CULTURE_MODELS["gemma4_31b"]],
            ["base", "german"],
            "d_happy",
            pair=("klingon", "spanish-mx"),
            outputs_root=tmp_path,
            tables_root=tmp_path,
            figures_root=tmp_path / "figs",
        )
    assert not (tmp_path / "figs").exists()
    assert not (tmp_path / "culture").exists()

    unmatched = culture_cross(
        [CULTURE_MODELS["gemma4_31b"]],
        ["base", "german"],
        "d_happy",
        pair=("arabic", "german"),
        outputs_root=tmp_path,
        tables_root=tmp_path,
        figures_root=tmp_path / "figs",
    )
    assert unmatched["pair"] == {"cultures": ["arabic", "german"], "countries": []}


def test_report_draws_nothing_for_a_single_loaded_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_load_arm(
        model: CultureModel, arm: str, question: Question, outputs_root: Path | None = None
    ) -> PreparedData | None:
        return _prepared(list(SUBPOPULATIONS), 3) if arm == "base" else None

    monkeypatch.setattr(cross_culture, "load_arm", fake_load_arm)

    payload = culture_cross(
        [CULTURE_MODELS["gemma4_31b"]],
        ["base", "german"],
        "d_happy",
        outputs_root=tmp_path,
        tables_root=tmp_path,
        figures_root=tmp_path / "figs",
    )

    assert payload["models"] == {"gemma4_31b": ["base"]}
    assert payload["figures"] == []
    assert not (tmp_path / "figs").exists()


def test_panel_stems_split_by_mode(tmp_path: Path) -> None:
    entities = [SURVEY, "base", "german", "spanish-mx"]
    matrices = {"NTP": _square(1, entities), "FA": _square(2, entities)}

    heat = pairwise_heatmap(matrices, "Model", tmp_path, "fig_culture_pairwise_nemd_test")
    assert sorted({path.stem for path in heat}) == [
        "fig_culture_pairwise_nemd_test_fa",
        "fig_culture_pairwise_nemd_test_ntp",
    ]
    assert sorted({path.suffix for path in heat}) == [".pdf", ".png"]

    maps = culture_map(matrices, "Model", tmp_path, "fig_culture_map_test")
    assert sorted({path.stem for path in maps}) == [
        "fig_culture_map_test_fa",
        "fig_culture_map_test_ntp",
    ]

    two = matrices["NTP"].iloc[:2, :2]
    small = culture_map({"NTP": two, "FA": two}, "Model", tmp_path, "fig_culture_map_small")
    assert sorted({path.stem for path in small}) == [
        "fig_culture_map_small_fa",
        "fig_culture_map_small_ntp",
    ]

    _seed_tree(tmp_path)
    league = league_table([CULTURE_MODELS["gemma4_31b"]], QUESTION, outputs_root=tmp_path)
    league["mixtral_mean_nEMD"] = np.nan
    assert sorted({path.stem for path in league_figure(league, tmp_path)}) == [
        "fig_model_culture_matched_comparison_fa",
        "fig_model_culture_matched_comparison_ntp",
    ]

    cells = head_to_head_cells(
        CULTURE_MODELS["gemma4_31b"], DEFAULT_PAIR, QUESTION, outputs_root=tmp_path
    )
    assert sorted({path.stem for path in head_to_head_figure(cells, tmp_path)}) == [
        "fig_culture_head_to_head_fa",
        "fig_culture_head_to_head_ntp",
    ]


def test_driver_writes_into_explicit_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outputs_root = tmp_path / "outputs"
    tables_root = tmp_path / "tables"
    figures_root = tmp_path / "figures"
    _seed_tree(outputs_root)
    seeds = {"base": 1, "german": 2, "spanish-mx": 3}

    def fake_load_arm(
        model: CultureModel, arm: str, question: Question, outputs_root: Path | None = None
    ) -> PreparedData | None:
        seed = seeds.get(arm)
        if seed is None:
            return None
        return _prepared(list(SUBPOPULATIONS), seed + (0 if model.key == "gemma4_31b" else 10))

    monkeypatch.setattr(cross_culture, "load_arm", fake_load_arm)

    payload = culture_cross(
        [CULTURE_MODELS["gemma4_31b"], CULTURE_MODELS["gemma4_e4b"]],
        ["base", "german", "spanish", "spanish-mx"],
        "d_happy",
        outputs_root=outputs_root,
        tables_root=tables_root,
        figures_root=figures_root,
    )

    tables_dir = tables_root / "culture" / "d_happy"
    assert sorted(path.name for path in tables_dir.iterdir()) == [
        "culture_head_to_head.csv",
        "culture_head_to_head_cells.csv",
        "culture_league.csv",
        "culture_pairwise_nemd.csv",
    ]
    stems = {path.stem for path in (figures_root / "culture" / "d_happy").iterdir()}
    for stem in (
        "fig_culture_pairwise_nemd_gemma4_31b_ntp",
        "fig_culture_pairwise_nemd_gemma4_e4b_fa",
        "fig_culture_pairwise_nemd_pooled_ntp",
        "fig_culture_map_gemma4_31b_fa",
        "fig_culture_map_pooled_ntp",
        "fig_model_culture_matched_comparison_ntp",
        "fig_culture_head_to_head_fa",
    ):
        assert stem in stems
    assert payload["models"] == {
        "gemma4_31b": ["base", "german", "spanish-mx"],
        "gemma4_e4b": ["base", "german", "spanish-mx"],
    }
    assert payload["pair"] == {"cultures": ["spanish", "spanish-mx"], "countries": ["Mexico"]}
    assert all(str(tables_root) in path for path in payload["tables"])
    assert all(str(figures_root) in path for path in payload["figures"])
    assert payload["tables"] and payload["figures"]
    assert not (outputs_root / "culture" / "d_happy").exists()
