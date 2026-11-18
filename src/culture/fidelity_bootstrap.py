from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from machine_bias_reproduction.config import (
    FIGURES_ROOT,
    FIRST_REPLICATE,
    GLOBAL_SEED,
    OUTPUTS_ROOT,
)
from machine_bias_reproduction.data import (
    PreparedData,
    canonical_run_paths,
    group_responses,
    load_subpops,
    load_wvs,
    prepare_data,
    respondent_counts,
)
from machine_bias_reproduction.figures import save_plate
from machine_bias_reproduction.io_utils import stable_seed
from machine_bias_reproduction.plates import GRID, INK, MUTED_INK, SURFACE, tinted
from machine_bias_reproduction.questions import (
    Question,
    one_hot,
    resolve_question,
    resolve_questions,
)

from .fidelity import ORIENTATION, POPULATION, FloatArray, answer_array, group_fidelity
from .palette import MODE_TONES
from .population import REPLICATE
from .population_plates import MARKER_EDGE, MARKER_SIZE, NOTE_SIZE, PLATE_TEXT, legend_below
from .registry import BASE_ARM, CULTURE_MODELS, csv_stems, run_slug

READINGS: tuple[str, ...] = (
    "pfs",
    "score_accuracy",
    "score_dispersion",
    "score_structure",
    "score_center",
    "adaptability_ratio",
    "rho_structure",
    ORIENTATION,
)

CONFIDENCE = 0.95

TAILS: tuple[float, float] = ((1.0 - CONFIDENCE) / 2.0, (1.0 + CONFIDENCE) / 2.0)

REPLICATES = 10_000

SYNTHETIC = "synthetic"

SURVEYED = "wvs"

SAMPLED_MODE = "fa"

SYNTHETIC_SURVEY: tuple[float, ...] = (0.15, 0.30, 0.46, 0.65, 0.85)

SYNTHETIC_MODELS: dict[str, tuple[float, ...]] = {
    "A": (0.22, 0.35, 0.45, 0.60, 0.76),
    "B": (0.11, 0.43, 0.48, 0.60, 0.90),
    "C": (0.17, 0.33, 0.48, 0.65, 0.83),
}

SYNTHETIC_GROUPS: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5")

SYNTHETIC_RESPONSES = 2_000

SYNTHETIC_PAIRS: tuple[tuple[str, str], ...] = (("B", "A"), ("C", "A"))

SYNTHETIC_IDENTITY: dict[str, Any] = {
    "study": SYNTHETIC,
    "question": SYNTHETIC,
    "question_label": "Synthetic binary question",
    "mode": SAMPLED_MODE,
}

RELEASED_MODELS: tuple[str, str] = ("qwen3_vl_8b", "gemma4_31b")

RELEASED_REVISION = "3c908b5fd1dacd30c1ed910483f5d71a3c1d138e"

RELEASED_MODES: tuple[str, ...] = (SAMPLED_MODE, "ntp")

CALIBRATION_DATASETS = 100

CALIBRATION_REPLICATES = 200

SYNTHETIC_DATASETS = 1_000

SYNTHETIC_CALIBRATION_REPLICATES = 500

CALIBRATED: tuple[str, ...] = ("pfs", "score_accuracy", "score_dispersion", "score_structure")

SHARE_TOLERANCE = 1e-9

CALIBRATION_KEYS: tuple[str, ...] = (
    "study",
    "question",
    "question_label",
    "mode",
    "model",
    "reference",
    "reading",
)

CALIBRATED_INTERVALS: dict[str, tuple[str, str]] = {
    "interval": ("low", "high"),
    "percentile": ("percentile_low", "percentile_high"),
    "centered": ("centered_low", "centered_high"),
}

WORKERS = min(8, os.cpu_count() or 1)

BOOTSTRAP_FIGURES = FIGURES_ROOT / "fidelity" / "bootstrap"

BOOTSTRAP_INPUTS_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_bootstrap_synthetic.csv"

BOOTSTRAP_LEVELS_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_bootstrap_levels.csv"

BOOTSTRAP_DIFFERENCES_TABLE = (
    OUTPUTS_ROOT / "culture" / "population_fidelity_bootstrap_differences.csv"
)

BOOTSTRAP_RUNS_TABLE = OUTPUTS_ROOT / "culture" / "population_fidelity_bootstrap_runs.csv"

BOOTSTRAP_CALIBRATION_TABLE = (
    OUTPUTS_ROOT / "culture" / "population_fidelity_bootstrap_calibration.csv"
)

STRIP_WIDTH = 11.0

STRIP_ROW = 0.78

STRIP_MARGIN = 1.9

STRIP_HEIGHT = 0.62

STRIP_BAR = 0.2

STRIP_BINS = 90

STRIP_TINT = 0.55

STRIP_NOTE = "${observed}$ [${low}$, ${high}$]"

DIFFERENCE_LABEL = "Difference in PFS"

SEPARATED_LABEL = "separated from zero"

UNSEPARATED_LABEL = "not separated"

SPREAD_LABEL = "95% interval"

REPLICATE_LABEL = "replicate differences"

ZERO_LABEL = "no difference"


@dataclass(frozen=True, slots=True)
class Cells:
    shares: FloatArray
    answers: FloatArray | None = None

    def __post_init__(self) -> None:
        if not np.allclose(self.shares.sum(axis=1), 1.0, atol=SHARE_TOLERANCE):
            raise ValueError("the shares of a cell do not sum to one")
        if self.answers is None:
            return
        if self.answers.shape != (self.shares.shape[0],):
            raise ValueError("one count of answers is needed per cell")
        if not bool(np.all(self.answers >= 1.0)):
            raise ValueError("a redrawn cell needs at least one answer")


@dataclass(frozen=True, slots=True)
class Bootstrap:
    observed: dict[str, FloatArray]
    replicates: dict[str, FloatArray]
    n_cells: int

    @property
    def n_replicates(self) -> int:
        return int(next(iter(self.replicates.values())).shape[0])

    def reading(self, label: str, name: str) -> FloatArray:
        return np.asarray(self.replicates[label][:, READINGS.index(name)], dtype=np.float64)

    def score(self, label: str, name: str) -> float:
        return float(self.observed[label][READINGS.index(name)])

    def gap(self, model: str, reference: str, name: str) -> FloatArray:
        return self.reading(model, name) - self.reading(reference, name)

    def difference(self, model: str, reference: str, name: str) -> float:
        return self.score(model, name) - self.score(reference, name)


@dataclass(frozen=True, slots=True)
class Strip:
    label: str
    gap: FloatArray
    observed: float
    ink: str = INK


def case_seed(seed: int, *names: str) -> np.random.SeedSequence:
    return np.random.SeedSequence([seed, *(stable_seed(name, seed) for name in names)])


def child_seed(seed: np.random.SeedSequence, index: int) -> np.random.SeedSequence:
    return np.random.SeedSequence(seed.entropy, spawn_key=(*seed.spawn_key, index))


def redrawn(rng: np.random.Generator, cells: Cells) -> FloatArray:
    if cells.answers is None:
        return cells.shares
    sizes = cells.answers.astype(np.int64)
    return np.asarray(rng.multinomial(sizes, cells.shares) / sizes[:, None], dtype=np.float64)


def readings(survey: FloatArray, model: FloatArray) -> FloatArray:
    row = group_fidelity(survey, model)
    return np.array([row[name] for name in READINGS], dtype=np.float64)


def paired_bootstrap(
    survey: Cells,
    models: Mapping[str, Cells],
    replicates: int = REPLICATES,
    seed: int | np.random.SeedSequence = GLOBAL_SEED,
) -> Bootstrap:
    rng = np.random.default_rng(seed)
    observed = {label: readings(survey.shares, cells.shares) for label, cells in models.items()}
    drawn = {label: np.empty((replicates, len(READINGS)), dtype=np.float64) for label in models}
    for index in range(replicates):
        reference = redrawn(rng, survey)
        for label, cells in models.items():
            drawn[label][index] = readings(reference, redrawn(rng, cells))
    return Bootstrap(observed, drawn, int(survey.shares.shape[0]))


def spread(values: FloatArray, observed: float) -> dict[str, float]:
    defined = values[np.isfinite(values)]
    row = {"observed": observed, "n_defined": float(defined.size)}
    if not defined.size:
        blank = float("nan")
        return {
            **row,
            **dict.fromkeys(
                (
                    "replicate_mean",
                    "shift",
                    "replicate_sd",
                    "low",
                    "high",
                    "percentile_low",
                    "percentile_high",
                    "centered_low",
                    "centered_high",
                ),
                blank,
            ),
        }
    low, high = (float(value) for value in np.quantile(defined, TAILS))
    mean = float(defined.mean())
    shift = mean - observed
    return {
        **row,
        "replicate_mean": mean,
        "shift": shift,
        "replicate_sd": float(defined.std(ddof=1)) if defined.size > 1 else float("nan"),
        "low": min(low, low - shift),
        "high": max(high, high - shift),
        "percentile_low": low,
        "percentile_high": high,
        "centered_low": low - shift,
        "centered_high": high - shift,
    }


def signed(value: float, digits: int = 3) -> str:
    text = f"{value:+.{digits}f}"
    return text[1:] if float(text) == 0.0 else text


def separated(reach: Mapping[str, float]) -> bool:
    return reach["low"] > 0.0 or reach["high"] < 0.0


def level_rows(identity: Mapping[str, Any], result: Bootstrap) -> list[dict[str, Any]]:
    return [
        {
            **identity,
            "model": label,
            "reading": name,
            "n_cells": result.n_cells,
            "n_replicates": result.n_replicates,
            **spread(result.reading(label, name), result.score(label, name)),
            "share_zero": float(np.mean(result.reading(label, name) == 0.0)),
        }
        for label in result.observed
        for name in READINGS
    ]


def difference_rows(
    identity: Mapping[str, Any],
    result: Bootstrap,
    pairs: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, reference in pairs:
        for name in READINGS:
            gap = result.gap(model, reference, name)
            defined = gap[np.isfinite(gap)]
            blank = float("nan")
            reach = spread(gap, result.difference(model, reference, name))
            rows.append(
                {
                    **identity,
                    "model": model,
                    "reference": reference,
                    "reading": name,
                    "n_cells": result.n_cells,
                    "n_replicates": result.n_replicates,
                    **reach,
                    "share_above": float(np.mean(defined > 0.0)) if defined.size else blank,
                    "share_below": float(np.mean(defined < 0.0)) if defined.size else blank,
                    "share_tied": float(np.mean(defined == 0.0)) if defined.size else blank,
                    "separated": separated(reach),
                }
            )
    return rows


def binary_cells(shares: Sequence[float]) -> FloatArray:
    chosen = np.asarray(shares, dtype=np.float64)
    return np.column_stack([1.0 - chosen, chosen])


def sampled_cells(shares: Sequence[float], responses: int) -> Cells:
    cells = binary_cells(shares)
    return Cells(cells, np.full(cells.shape[0], float(responses)))


def synthetic_inputs() -> pd.DataFrame:
    return pd.DataFrame({"group": SYNTHETIC_GROUPS, "survey": SYNTHETIC_SURVEY, **SYNTHETIC_MODELS})


def synthetic_bootstrap(
    replicates: int = REPLICATES,
    seed: int | np.random.SeedSequence = GLOBAL_SEED,
    responses: int = SYNTHETIC_RESPONSES,
) -> Bootstrap:
    models = {label: sampled_cells(shares, responses) for label, shares in SYNTHETIC_MODELS.items()}
    return paired_bootstrap(sampled_cells(SYNTHETIC_SURVEY, responses), models, replicates, seed)


def survey_cells(question: Question) -> tuple[pd.DataFrame, pd.Series]:
    wvs, cells = load_wvs(), load_subpops()["subpop"]
    answers = one_hot(
        question.normalize(wvs[question.var]), question.wvs_labels, question.answer_columns
    )
    shares = group_responses(answers, cells)
    counts = answers.notna().any(axis=1).groupby(cells.to_numpy()).sum().reindex(shares.index)
    answered = shares.notna().all(axis=1)
    return shares[answered], counts[answered].astype(np.float64)


def response_files(
    keys: Sequence[str] = RELEASED_MODELS,
    arm: str = BASE_ARM,
    questions: Sequence[Question] | None = None,
) -> list[str]:
    return [
        f"data/{run_slug(key, arm)}/{question.var}/{stem}"
        for key in keys
        for question in questions or resolve_questions(None)
        for stem in csv_stems(key, arm, question)
    ]


def released_run(root: Path, key: str, question: Question, arm: str = BASE_ARM) -> PreparedData:
    ntp_path, fa_path = canonical_run_paths(
        root / run_slug(key, arm) / question.var, question, csv_stems(key, arm, question)
    )
    return prepare_data(pd.read_csv(ntp_path), pd.read_csv(fa_path), question)


def answer_counts(prepared: PreparedData) -> pd.Series:
    if prepared.fa_raw is None:
        raise ValueError("no sampled answers in this run")
    question = prepared.question
    matched = prepared.fa_raw.set_index("id").reindex(prepared.wvs["id"]).reset_index(drop=True)
    answers = one_hot(
        question.normalize(matched[question.var]), question.fa_answers, question.answer_columns
    )
    counted = answers.notna().any(axis=1).groupby(prepared.subpops["subpop"].to_numpy()).sum()
    return counted.reindex(prepared.names).astype(np.float64)


def model_cells(prepared: PreparedData, mode: str, names: pd.Index) -> Cells:
    shares = answer_array(prepared, prepared.props(mode), names)
    if mode != SAMPLED_MODE:
        return Cells(shares)
    answers = answer_counts(prepared).reindex(names).to_numpy(dtype=np.float64)
    whole = shares * answers[:, None]
    if not np.allclose(whole, np.round(whole), atol=1e-6):
        raise ValueError("the sampled shares are not counts over the valid answers")
    return Cells(shares, answers)


def surveyed_cells(prepared: PreparedData, names: pd.Index) -> Cells:
    shares = answer_array(prepared, prepared.wvs_props, names)
    return Cells(shares, respondent_counts(prepared).reindex(names).to_numpy(dtype=np.float64))


def model_label(key: str) -> str:
    return CULTURE_MODELS[key].label


def released_cells(
    root: Path, keys: Sequence[str], question: Question, mode: str
) -> tuple[Cells, dict[str, Cells]]:
    runs = {key: released_run(root, key, question) for key in keys}
    names = pd.Index(sorted(set.intersection(*(set(run.names) for run in runs.values()))))
    models = {model_label(key): model_cells(run, mode, names) for key, run in runs.items()}
    return surveyed_cells(runs[keys[0]], names), models


ReleasedTask = tuple[Path, tuple[str, ...], str, str, int, np.random.SeedSequence]


def released_task(task: ReleasedTask) -> tuple[str, str, Bootstrap]:
    root, keys, variable, mode, replicates, seed = task
    survey, models = released_cells(root, keys, resolve_question(variable), mode)
    return variable, mode, paired_bootstrap(survey, models, replicates, seed)


def _finished(task: Any, tasks: Sequence[Any], workers: int) -> list[Any]:
    if workers <= 1:
        return [task(entry) for entry in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(task, tasks))


def released_cases(
    questions: Sequence[Question] | None, modes: Sequence[str]
) -> list[tuple[str, str]]:
    return [
        (question.var, mode) for question in questions or resolve_questions(None) for mode in modes
    ]


def released_bootstrap(
    root: Path,
    keys: Sequence[str] = RELEASED_MODELS,
    questions: Sequence[Question] | None = None,
    modes: Sequence[str] = RELEASED_MODES,
    replicates: int = REPLICATES,
    seed: int = GLOBAL_SEED,
    workers: int = WORKERS,
) -> dict[tuple[str, str], Bootstrap]:
    tasks = [
        (root, tuple(keys), variable, mode, replicates, case_seed(seed, variable, mode))
        for variable, mode in released_cases(questions, modes)
    ]
    finished = _finished(released_task, tasks, workers)
    return {(variable, mode): result for variable, mode, result in finished}


def released_identity(question: Question, mode: str) -> dict[str, Any]:
    return {
        "study": SURVEYED,
        "question": question.var,
        "question_label": question.label,
        "mode": mode,
    }


def bootstrap_tables(
    synthetic: Bootstrap,
    released: Mapping[tuple[str, str], Bootstrap],
    keys: Sequence[str] = RELEASED_MODELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    levels = level_rows(SYNTHETIC_IDENTITY, synthetic)
    differences = difference_rows(SYNTHETIC_IDENTITY, synthetic, SYNTHETIC_PAIRS)
    pair = (model_label(keys[0]), model_label(keys[1]))
    for (variable, mode), result in released.items():
        identity = released_identity(resolve_question(variable), mode)
        levels += level_rows(identity, result)
        differences += difference_rows(identity, result, (pair,))
    return pd.DataFrame(levels), pd.DataFrame(differences)


def run_differences(groups: pd.DataFrame, keys: Sequence[str] = RELEASED_MODELS) -> pd.DataFrame:
    pooled = groups[
        groups["group"].eq(POPULATION)
        & groups["arm"].eq(BASE_ARM)
        & groups["mode"].eq(SAMPLED_MODE)
        & groups["model_key"].isin(list(keys))
    ]
    runs = ["question", "question_label", "mode", REPLICATE]
    tables: list[pd.DataFrame] = []
    for name in CALIBRATED:
        wide = pooled.pivot(index=runs, columns="model_key", values=[name, "n_cells"])
        table = pd.DataFrame(
            {
                "model": model_label(keys[0]),
                "reference": model_label(keys[1]),
                "reading": name,
                "n_cells_model": wide[("n_cells", keys[0])],
                "n_cells_reference": wide[("n_cells", keys[1])],
                "score_model": wide[(name, keys[0])],
                "score_reference": wide[(name, keys[1])],
            }
        )
        table["difference"] = table["score_model"] - table["score_reference"]
        tables.append(table.reset_index())
    return pd.concat(tables, ignore_index=True)


Pairs = tuple[tuple[str, str], ...]

CalibrationTask = tuple[Cells, dict[str, Cells], Pairs, int, np.random.SeedSequence, dict[str, Any]]


def calibration_task(task: CalibrationTask) -> list[dict[str, Any]]:
    survey, models, pairs, replicates, seed, identity = task
    drawing, resampling = child_seed(seed, 0), child_seed(seed, 1)
    rng = np.random.default_rng(drawing)
    sampled = Cells(redrawn(rng, survey), survey.answers)
    answered = {label: Cells(redrawn(rng, cells), cells.answers) for label, cells in models.items()}
    result = paired_bootstrap(sampled, answered, replicates, resampling)
    rows = difference_rows(identity, result, pairs)
    return [row for row in rows if row["reading"] in CALIBRATED]


def calibration_tasks(
    identity: Mapping[str, Any],
    survey: Cells,
    models: dict[str, Cells],
    pairs: Pairs,
    datasets: int,
    replicates: int,
    seed: np.random.SeedSequence,
) -> tuple[list[CalibrationTask], list[dict[str, Any]]]:
    scored = {label: group_fidelity(survey.shares, cells.shares) for label, cells in models.items()}
    populations = [
        {
            **identity,
            "model": model,
            "reference": reference,
            "reading": name,
            "population": float(scored[model][name] - scored[reference][name]),
        }
        for model, reference in pairs
        for name in CALIBRATED
    ]
    tasks: list[CalibrationTask] = [
        (
            survey,
            models,
            pairs,
            replicates,
            child_seed(seed, dataset),
            {**identity, "dataset": dataset},
        )
        for dataset in range(datasets)
    ]
    return tasks, populations


def calibrated(
    tasks: Sequence[CalibrationTask], populations: Sequence[Mapping[str, Any]], workers: int
) -> pd.DataFrame:
    rows = [row for chunk in _finished(calibration_task, tasks, workers) for row in chunk]
    return pd.DataFrame(rows).merge(
        pd.DataFrame(populations), on=list(CALIBRATION_KEYS), validate="many_to_one"
    )


def synthetic_calibration(
    datasets: int = SYNTHETIC_DATASETS,
    replicates: int = SYNTHETIC_CALIBRATION_REPLICATES,
    seed: int = GLOBAL_SEED,
    workers: int = WORKERS,
    responses: int = SYNTHETIC_RESPONSES,
) -> pd.DataFrame:
    models = {label: sampled_cells(shares, responses) for label, shares in SYNTHETIC_MODELS.items()}
    tasks, populations = calibration_tasks(
        SYNTHETIC_IDENTITY,
        sampled_cells(SYNTHETIC_SURVEY, responses),
        models,
        SYNTHETIC_PAIRS,
        datasets,
        replicates,
        np.random.SeedSequence(seed),
    )
    return calibrated(tasks, populations, workers)


def released_calibration(
    root: Path,
    keys: Sequence[str] = RELEASED_MODELS,
    questions: Sequence[Question] | None = None,
    modes: Sequence[str] = RELEASED_MODES,
    datasets: int = CALIBRATION_DATASETS,
    replicates: int = CALIBRATION_REPLICATES,
    seed: int = GLOBAL_SEED,
    workers: int = WORKERS,
) -> pd.DataFrame:
    pairs: Pairs = ((model_label(keys[0]), model_label(keys[1])),)
    tasks: list[CalibrationTask] = []
    populations: list[dict[str, Any]] = []
    for variable, mode in released_cases(questions, modes):
        question = resolve_question(variable)
        survey, models = released_cells(root, tuple(keys), question, mode)
        added, known = calibration_tasks(
            released_identity(question, mode),
            survey,
            models,
            pairs,
            datasets,
            replicates,
            case_seed(seed, variable, mode),
        )
        tasks += added
        populations += known
    return calibrated(tasks, populations, workers)


def calibration_summary(drawn: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = list(CALIBRATION_KEYS)
    for identity, block in drawn.groupby(keys, sort=False):
        observed = block["observed"].to_numpy(dtype=np.float64)
        expected = float(observed.mean())
        population = float(block["population"].iloc[0])
        row: dict[str, Any] = {
            **dict(zip(keys, cast(tuple[Any, ...], identity), strict=True)),
            "n_datasets": len(block),
            "n_replicates": int(block["n_replicates"].iloc[0]),
            "population": population,
            "expected": expected,
            "dataset_sd": float(observed.std(ddof=1)),
            "replicate_sd_mean": float(block["replicate_sd"].mean()),
            "shift_mean": float(block["shift"].mean()),
        }
        for interval, (low, high) in CALIBRATED_INTERVALS.items():
            lower = block[low].to_numpy(dtype=np.float64)
            upper = block[high].to_numpy(dtype=np.float64)
            row[f"{interval}_covers_expected"] = float(
                np.mean((lower <= expected) & (expected <= upper))
            )
            row[f"{interval}_covers_population"] = float(
                np.mean((lower <= population) & (population <= upper))
            )
            row[f"{interval}_excludes_zero"] = float(np.mean((lower > 0.0) | (upper < 0.0)))
            row[f"{interval}_width_mean"] = float(np.mean(upper - lower))
        rows.append(row)
    return pd.DataFrame(rows)


def check_bootstrap(
    levels: pd.DataFrame,
    groups: pd.DataFrame,
    keys: Sequence[str] = RELEASED_MODELS,
) -> int:
    assert not levels.empty
    ordered = levels["low"].le(levels["high"]) | levels["low"].isna()
    assert ordered.all(), "an interval is inverted"
    scored = levels[levels["study"].eq(SURVEYED)]
    reference = groups[
        groups["group"].eq(POPULATION)
        & groups["arm"].eq(BASE_ARM)
        & groups[REPLICATE].eq(FIRST_REPLICATE)
        & groups["model_key"].isin(list(keys))
    ].assign(model=lambda frame: frame["model_label"])
    matched = 0
    for name in READINGS:
        ours = scored[scored["reading"].eq(name)]
        merged = ours.merge(
            reference[["question", "mode", "model", "n_cells", name]],
            on=["question", "mode", "model"],
            how="inner",
            suffixes=("", "_published"),
            validate="one_to_one",
        )
        assert len(merged) == len(ours), "a bootstrapped run is missing from the groups table"
        same = merged[merged["n_cells"].eq(merged["n_cells_published"])]
        assert np.allclose(
            same["observed"].to_numpy(dtype=np.float64),
            same[name].to_numpy(dtype=np.float64),
            atol=1e-12,
            equal_nan=True,
        ), f"{name} drifted from the groups table"
        matched += len(same)
    return matched


def strip_handles() -> list[Line2D]:
    marker: dict[str, Any] = {
        "linestyle": "none",
        "marker": "D",
        "markersize": MARKER_SIZE,
        "markeredgewidth": MARKER_EDGE,
        "markeredgecolor": INK,
    }
    return [
        Line2D([], [], markerfacecolor=INK, label=SEPARATED_LABEL, **marker),
        Line2D([], [], markerfacecolor=SURFACE, label=UNSEPARATED_LABEL, **marker),
        Line2D([], [], color=INK, linewidth=2.2, label=SPREAD_LABEL),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="s",
            markersize=MARKER_SIZE + 2.0,
            markeredgewidth=0.0,
            markerfacecolor=tinted(INK, STRIP_TINT),
            label=REPLICATE_LABEL,
        ),
        Line2D([], [], color=MUTED_INK, linestyle=(0, (4, 2)), label=ZERO_LABEL),
    ]


def strip_plate(strips: Sequence[Strip], stem: Path, xlabel: str = DIFFERENCE_LABEL) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    with rc_context(cast(Any, PLATE_TEXT)):
        handles = strip_handles()
        figure = Figure(
            figsize=(STRIP_WIDTH, STRIP_ROW * len(strips) + STRIP_MARGIN), layout="constrained"
        )
        axis = figure.add_subplot()
        drawn = [strip.gap[np.isfinite(strip.gap)] for strip in strips]
        reaches = [spread(strip.gap, strip.observed) for strip in strips]
        lowest = min(0.0, *(float(values.min()) for values in drawn))
        highest = max(0.0, *(float(values.max()) for values in drawn))
        lowest = min(lowest, *(reach["low"] for reach in reaches))
        highest = max(highest, *(reach["high"] for reach in reaches))
        edges = np.linspace(lowest, highest, STRIP_BINS + 1)
        for row, (strip, values, reach) in enumerate(zip(strips, drawn, reaches, strict=True)):
            low, high = reach["low"], reach["high"]
            apart = separated(reach)
            counts, _ = np.histogram(values, bins=edges)
            height = STRIP_HEIGHT * counts / counts.max()
            axis.fill_between(
                edges,
                row,
                row - np.append(height, height[-1]),
                step="post",
                color=tinted(strip.ink, STRIP_TINT),
                linewidth=0.0,
                zorder=2.0,
            )
            axis.hlines(row, edges[0], edges[-1], color=GRID, linewidth=0.8, zorder=1.0)
            axis.hlines(row + STRIP_BAR, low, high, color=strip.ink, linewidth=2.2, zorder=3.0)
            axis.plot(
                strip.observed,
                row + STRIP_BAR,
                marker="D",
                markersize=MARKER_SIZE,
                markeredgewidth=MARKER_EDGE,
                markeredgecolor=strip.ink,
                markerfacecolor=strip.ink if apart else SURFACE,
                linestyle="none",
                zorder=4.0,
            )
            axis.annotate(
                STRIP_NOTE.format(
                    observed=signed(strip.observed), low=signed(low), high=signed(high)
                ),
                xy=(1.0, row),
                xycoords=("axes fraction", "data"),
                xytext=(8.0, 0.0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=NOTE_SIZE + 1.5,
                color=INK,
                annotation_clip=False,
            )
        axis.axvline(0.0, color=MUTED_INK, linestyle=(0, (4, 2)), linewidth=1.1, zorder=2.5)
        axis.set_yticks(range(len(strips)), [strip.label for strip in strips])
        axis.set_ylim(len(strips) - 1 + 0.55, -STRIP_HEIGHT - 0.25)
        axis.tick_params(axis="y", length=0.0)
        axis.spines["left"].set_visible(False)
        axis.grid(axis="x", color=GRID, linewidth=0.6)
        axis.set_xlabel(xlabel)
        legend_below(figure, handles, len(handles))
        return save_plate(figure, stem)


def synthetic_strips(result: Bootstrap) -> list[Strip]:
    return [
        Strip(
            f"Model {model} minus Model {reference}",
            result.gap(model, reference, "pfs"),
            result.difference(model, reference, "pfs"),
        )
        for model, reference in SYNTHETIC_PAIRS
    ]


def released_strips(
    released: Mapping[tuple[str, str], Bootstrap], keys: Sequence[str] = RELEASED_MODELS
) -> list[Strip]:
    model, reference = model_label(keys[0]), model_label(keys[1])
    strips: list[Strip] = []
    for (variable, mode), result in released.items():
        name = mode.upper()
        strips.append(
            Strip(
                f"{resolve_question(variable).label}, {name}",
                result.gap(model, reference, "pfs"),
                result.difference(model, reference, "pfs"),
                MODE_TONES[name].ink,
            )
        )
    return strips


def bootstrap_plates(
    synthetic: Bootstrap,
    released: Mapping[tuple[str, str], Bootstrap],
    keys: Sequence[str] = RELEASED_MODELS,
    destination: Path = BOOTSTRAP_FIGURES,
) -> list[Path]:
    label = f"PFS of {model_label(keys[0])} minus PFS of {model_label(keys[1])}"
    return [
        *strip_plate(synthetic_strips(synthetic), destination / "fig_fidelity_bootstrap_synthetic"),
        *strip_plate(
            released_strips(released, keys), destination / "fig_fidelity_bootstrap_models", label
        ),
    ]
