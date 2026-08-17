from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

from .config import (
    GLOBAL_SEED,
    PROJECT_ROOT,
    RunPaths,
    paths_for,
)
from .data import (
    PreparedData,
    archived_fa,
    archived_ntp,
    canonical_run_paths,
    country_of,
    load_linear_baseline,
    prepare_data,
)
from .figures import generate_figures
from .io_utils import atomic_write_json, hardware_summary, hash_paths, write_csv
from .metrics import (
    DISTANCES,
    KL_ZERO_PROPORTION_SMOOTHING,
    MMD_BANDWIDTH_IN_CATEGORY_STEPS,
    QUALITY_LABELS,
    nemd,
    pairwise_nemd,
    quality_classes,
)
from .questions import Question, one_hot, resolve_question
from .r_rng import center_holdout_indices

SOCIAL_COLUMNS = (
    "countryAustralia",
    "countryGermany",
    "countryMexico",
    "countryRussia",
    "decade200X",
    "decade199X",
    "Sex_Female",
    "Age_<25",
    "Age_25-34",
    "Age_45-54",
    "Age_55-64",
    "Age_65-74",
    "Age_75+",
    "Education_Low",
    "Education_Middle",
    "Employment_Homemaker",
    "Employment_Retired",
    "Employment_Student",
    "Employment_Unemployed",
    "Marstat_Cohabiting",
    "Marstat_Divorced or separated",
    "Marstat_Single",
    "Marstat_Widowed",
)


@dataclass(frozen=True, slots=True)
class RegressionArtifacts:
    fit: pd.DataFrame
    f_tests: pd.DataFrame
    social_coefficients: pd.DataFrame
    full_coefficients: pd.DataFrame
    standardized_coefficients: pd.DataFrame
    holdouts: pd.DataFrame


def _response_distribution(values: pd.Series, categories: tuple[str, ...]) -> np.ndarray:
    counts = values.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=np.float64)
    return np.asarray(counts / counts.sum(), dtype=np.float64)


def _distance_rows(
    method: str,
    names: pd.Index,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame({"method": method, "subpopulation": names})
    for name, function in DISTANCES.items():
        frame[name] = np.atleast_1d(function(reference, candidate))
    return frame


def social_design(names: pd.Index, predictors: pd.DataFrame) -> pd.DataFrame:
    labels = names.to_series(index=names)
    country = country_of(labels)
    year = pd.to_numeric(labels.str.replace(r"^.*? (\d+) .*$", r"\1", regex=True))
    decade = year.floordiv(10).mul(10)

    design = pd.DataFrame(index=names)
    for country_value in ("Australia", "Germany", "Mexico", "Russia"):
        design[f"country{country_value}"] = (country.to_numpy() == country_value).astype(np.float64)
    for decade_value in (2000, 1990):
        design[f"decade{str(decade_value)[:3]}X"] = (decade.to_numpy() == decade_value).astype(
            np.float64
        )
    for column in SOCIAL_COLUMNS[6:]:
        design[column] = predictors[column].to_numpy(dtype=np.float64)
    return design.loc[:, list(SOCIAL_COLUMNS)]


def _r_bic(result: RegressionResultsWrapper) -> float:
    parameter_count_including_sigma = int(result.df_model) + 2
    return float(-2 * result.llf + math.log(result.nobs) * parameter_count_including_sigma)


def _coefficient_table(
    result: RegressionResultsWrapper,
    *,
    mode: str,
    model: str,
) -> pd.DataFrame:
    confidence = result.conf_int(alpha=0.05)
    return pd.DataFrame(
        {
            "mode": mode,
            "model": model,
            "predictor": result.params.index,
            "estimate": result.params.to_numpy(dtype=np.float64),
            "std_error": result.bse.to_numpy(dtype=np.float64),
            "ci_low": confidence.iloc[:, 0].to_numpy(dtype=np.float64),
            "ci_high": confidence.iloc[:, 1].to_numpy(dtype=np.float64),
            "p_value": result.pvalues.to_numpy(dtype=np.float64),
        }
    )


def regressions(
    names: pd.Index,
    wvs_props: pd.DataFrame,
    llm_props: pd.DataFrame,
    predictors: pd.DataFrame,
    mode: str,
) -> RegressionArtifacts:
    design = social_design(names, predictors)
    indices = center_holdout_indices(len(names), mode)
    holdout_names = names.take(indices)
    llm_center = llm_props.iloc[indices].mean(axis=0).to_numpy(dtype=np.float64)
    response = np.log1p(
        nemd(
            wvs_props.to_numpy(dtype=np.float64),
            llm_props.to_numpy(dtype=np.float64),
        )
    )
    center_distance = np.log1p(nemd(wvs_props.to_numpy(dtype=np.float64), llm_center))
    keep = np.ones(len(names), dtype=bool)
    keep[indices] = False

    response_series = pd.Series(response[keep], name="nEMD_log")
    social_x = sm.add_constant(design.loc[keep].reset_index(drop=True), has_constant="add")
    center_frame = pd.DataFrame({"nEMD_center": center_distance[keep]})
    center_x = sm.add_constant(center_frame, has_constant="add")
    full_x = sm.add_constant(
        pd.concat(
            [center_frame.reset_index(drop=True), design.loc[keep].reset_index(drop=True)],
            axis=1,
        ),
        has_constant="add",
    )

    social_fit = sm.OLS(response_series, social_x).fit()
    center_fit = sm.OLS(response_series, center_x).fit()
    full_fit = sm.OLS(response_series, full_x).fit()

    unscaled = pd.concat(
        [center_frame.reset_index(drop=True), design.loc[keep].reset_index(drop=True)],
        axis=1,
    )
    standardized = (unscaled - unscaled.mean()) / unscaled.std(ddof=1)
    standardized_x = sm.add_constant(standardized, has_constant="add")
    standardized_fit = sm.OLS(response_series, standardized_x).fit()

    fit_rows = []
    for name, result in (
        ("social", social_fit),
        ("center", center_fit),
        ("full", full_fit),
    ):
        fit_rows.append(
            {
                "mode": mode,
                "model": name,
                "n": int(result.nobs),
                "parameters": int(result.df_model) + 1,
                "adjusted_r_squared": float(result.rsquared_adj),
                "bic_r": _r_bic(result),
                "aic": float(result.aic),
            }
        )

    center_vs_full = full_fit.compare_f_test(center_fit)
    social_vs_full = full_fit.compare_f_test(social_fit)
    f_tests = pd.DataFrame(
        [
            {
                "mode": mode,
                "reduced_model": "center",
                "added_terms": "social",
                "f_statistic": float(center_vs_full[0]),
                "p_value": float(center_vs_full[1]),
                "df_difference": int(center_vs_full[2]),
                "residual_df": int(full_fit.df_resid),
            },
            {
                "mode": mode,
                "reduced_model": "social",
                "added_terms": "center",
                "f_statistic": float(social_vs_full[0]),
                "p_value": float(social_vs_full[1]),
                "df_difference": int(social_vs_full[2]),
                "residual_df": int(full_fit.df_resid),
            },
        ]
    )
    holdouts = pd.DataFrame(
        {
            "mode": mode,
            "zero_based_index": indices,
            "subpopulation": holdout_names.to_numpy(),
        }
    )
    return RegressionArtifacts(
        fit=pd.DataFrame(fit_rows),
        f_tests=f_tests,
        social_coefficients=_coefficient_table(social_fit, mode=mode, model="social"),
        full_coefficients=_coefficient_table(full_fit, mode=mode, model="full"),
        standardized_coefficients=_coefficient_table(
            standardized_fit,
            mode=mode,
            model="full_standardized",
        ),
        holdouts=holdouts,
    )


def _random_baselines(data: PreparedData) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    question = data.question
    columns = question.answer_columns
    answers = question.normalize(data.wvs[question.var]).to_numpy(copy=True)
    rng = np.random.default_rng(GLOBAL_SEED)
    frames: list[pd.DataFrame] = []
    distances: list[pd.DataFrame] = []
    reference = data.wvs_props.to_numpy(dtype=np.float64)
    for iteration in range(1, 21):
        shuffled = pd.Series(rng.permutation(answers))
        responses = one_hot(shuffled, question.wvs_labels, columns)
        responses["name"] = data.subpops["subpop"].to_numpy()
        props = responses.groupby("name", sort=True)[list(columns)].mean().reindex(data.names)
        frames.append(props)
        distances.append(
            _distance_rows(
                f"Random_{iteration:02d}",
                data.names,
                reference,
                props.to_numpy(dtype=np.float64),
            )
        )
    return pd.concat(distances, ignore_index=True), frames


def _quality_table(distances: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = distances.copy()
    frame["quality"] = quality_classes(frame["nEMD"].to_numpy())
    nonrandom = frame[~frame["method"].str.startswith("Random_")]
    for method, group in nonrandom.groupby("method", sort=False):
        proportions = group["quality"].value_counts(normalize=True)
        for label in QUALITY_LABELS:
            rows.append(
                {
                    "method": method,
                    "quality": label,
                    "percent": 100 * float(proportions.get(label, 0.0)),
                }
            )
    random_rows: list[dict[str, Any]] = []
    for method, group in frame[frame["method"].str.startswith("Random_")].groupby("method"):
        proportions = group["quality"].value_counts(normalize=True)
        for label in QUALITY_LABELS:
            random_rows.append(
                {
                    "replicate": method,
                    "quality": label,
                    "proportion": float(proportions.get(label, 0.0)),
                }
            )
    random_frame = pd.DataFrame(random_rows)
    for label in QUALITY_LABELS:
        values = random_frame.loc[random_frame["quality"] == label, "proportion"]
        rows.append(
            {
                "method": "Random",
                "quality": label,
                "percent": 100 * float(values.mean()),
            }
        )
    return pd.DataFrame(rows)


def _paper_regression_checkpoints(
    regression_fit: pd.DataFrame,
    f_tests: pd.DataFrame,
) -> pd.DataFrame:
    checks = (
        ("adjusted_r_squared", "social", 0.387, 0.005),
        ("adjusted_r_squared", "center", 0.829, 0.002),
        ("adjusted_r_squared", "full", 0.858, 0.002),
        ("bic_r", "social", -2276.0, 10.0),
        ("bic_r", "center", -3250.0, 10.0),
        ("bic_r", "full", -3245.0, 10.0),
    )
    rows: list[dict[str, Any]] = []
    ntp_fit = regression_fit[regression_fit["mode"] == "ntp"].set_index("model")
    for metric, model, expected, tolerance in checks:
        actual = float(cast(Any, ntp_fit.loc[model, metric]))
        rows.append(
            {
                "paper_table": "Table 5",
                "metric": metric,
                "model": model,
                "expected": expected,
                "actual": actual,
                "tolerance": tolerance,
                "passed": abs(actual - expected) <= tolerance,
            }
        )
    f_expected = (("social", 6.73, 0.5), ("center", 2126.66, 0.5))
    ntp_tests = f_tests[f_tests["mode"] == "ntp"].set_index("added_terms")
    for added_terms, expected, tolerance in f_expected:
        actual = float(cast(Any, ntp_tests.loc[added_terms, "f_statistic"]))
        rows.append(
            {
                "paper_table": "Table 6",
                "metric": "f_statistic",
                "model": f"add_{added_terms}",
                "expected": expected,
                "actual": actual,
                "tolerance": tolerance,
                "passed": abs(actual - expected) <= tolerance,
            }
        )
    return pd.DataFrame(rows)


def load_source(
    source: str,
    paths: RunPaths,
    question: Question,
    stems: tuple[str, str] | None,
) -> PreparedData:
    if source == "archived":
        return prepare_data(archived_ntp(question), archived_fa(question), question)
    ntp_path, fa_path = canonical_run_paths(paths.outputs, question, stems)
    if not ntp_path.is_file() or not fa_path.is_file():
        raise FileNotFoundError(
            f"canonical NTP and FA CSVs are required before analysis: {paths.outputs}"
        )
    return prepare_data(pd.read_csv(ntp_path), pd.read_csv(fa_path), question)


def run_analysis(
    source: str,
    question: str | Question = "d_happy",
    *,
    model_description: str = "Mixtral-8x7B-v0.1.Q4_K_M",
    csv_stems: tuple[str, str] | None = None,
    paper_checkpoints: bool = True,
) -> dict[str, Any]:
    outcome = resolve_question(question)
    columns = outcome.answer_columns
    paths = paths_for(source, outcome.var)
    paths.ensure()
    data = load_source(source, paths, outcome, csv_stems)
    if not data.names.size:
        return _empty_analysis(source, outcome, paths, data, model_description)

    wvs_average = _response_distribution(
        outcome.normalize(data.wvs[outcome.var]), outcome.wvs_labels
    )
    ntp_average = outcome.ntp_answers(data.ntp_raw).mean().to_numpy()
    fa_average = _response_distribution(
        outcome.normalize(data.fa_raw[outcome.var]), outcome.fa_answers
    )
    distributions = pd.DataFrame(
        [
            {"method": "WVS", **dict(zip(columns, wvs_average, strict=True))},
            {"method": "NTP", **dict(zip(columns, ntp_average, strict=True))},
            {"method": "FA", **dict(zip(columns, fa_average, strict=True))},
        ]
    )

    reference = data.wvs_props.to_numpy(dtype=np.float64)
    method_props = {"NTP": data.ntp_props, "FA": data.fa_props}
    distance_frames: list[pd.DataFrame] = [
        _distance_rows(method, data.names, reference, props.to_numpy(dtype=np.float64))
        for method, props in method_props.items()
    ]

    linear = load_linear_baseline(outcome).set_index("name").reindex(data.names)
    if linear[list(columns)].isna().any().any():
        raise ValueError(f"linear baseline does not align with {outcome.var} subpopulations")
    distance_frames.append(
        _distance_rows(
            "Linear",
            data.names,
            reference,
            linear.loc[:, list(columns)].to_numpy(dtype=np.float64),
        )
    )
    random_distances, _ = _random_baselines(data)
    distance_frames.append(random_distances)
    distances = pd.concat(distance_frames, ignore_index=True)
    quality = _quality_table(distances)

    pairwise_rows = []
    for method, props in {"WVS": data.wvs_props, **method_props}.items():
        values = pairwise_nemd(props.to_numpy(dtype=np.float64))
        pairwise_rows.append(
            {
                "method": method,
                "pairs": len(values),
                "median_pairwise_nEMD": float(np.median(values)),
                "mean_pairwise_nEMD": float(np.mean(values)),
            }
        )
    pairwise = pd.DataFrame(pairwise_rows)

    ntp_overall = float(nemd(wvs_average, ntp_average))
    fa_overall = float(nemd(wvs_average, fa_average))
    metrics = pd.DataFrame(
        [
            {"metric": "wvs_rows", "method": "WVS", "value": float(len(data.wvs))},
            {
                "metric": "ntp_profiles",
                "method": "NTP",
                "value": float(len(data.ntp_raw)),
            },
            {
                "metric": "subpopulations",
                "method": "WVS",
                "value": float(len(data.names)),
            },
            {
                "metric": "valid_token_mass_mean",
                "method": "NTP",
                "value": float(data.ntp_raw["mass"].mean()),
            },
            {
                "metric": "subpopulations_dropped",
                "method": "WVS",
                "value": float(data.coverage.subpopulations_dropped),
            },
            {"metric": "overall_nEMD", "method": "NTP", "value": ntp_overall},
            {"metric": "overall_nEMD", "method": "FA", "value": fa_overall},
            *[
                {
                    "metric": "median_pairwise_nEMD",
                    "method": row["method"],
                    "value": row["median_pairwise_nEMD"],
                }
                for _, row in pairwise.iterrows()
            ],
        ]
    )

    regressions_by_mode = [
        regressions(data.names, data.wvs_props, props, data.social_predictors, mode)
        for mode, props in (("ntp", data.ntp_props), ("fa", data.fa_props))
    ]
    regression_fit = pd.concat([item.fit for item in regressions_by_mode], ignore_index=True)
    f_tests = pd.concat([item.f_tests for item in regressions_by_mode], ignore_index=True)
    social_coefficients = pd.concat(
        [item.social_coefficients for item in regressions_by_mode],
        ignore_index=True,
    )
    full_coefficients = pd.concat(
        [item.full_coefficients for item in regressions_by_mode],
        ignore_index=True,
    )
    standardized_coefficients = pd.concat(
        [item.standardized_coefficients for item in regressions_by_mode],
        ignore_index=True,
    )
    holdouts = pd.concat([item.holdouts for item in regressions_by_mode], ignore_index=True)

    artifacts: dict[str, pd.DataFrame] = {
        "response_distributions.csv": distributions,
        "summary_metrics.csv": metrics,
        "subpopulation_distances.csv": distances,
        "quality_bands.csv": quality,
        "pairwise_dispersion.csv": pairwise,
        "regression_fit.csv": regression_fit,
        "f_tests.csv": f_tests,
        "social_coefficients.csv": social_coefficients,
        "full_coefficients.csv": full_coefficients,
        "full_standardized_coefficients.csv": standardized_coefficients,
        "center_holdouts.csv": holdouts,
    }
    if paper_checkpoints:
        artifacts["paper_regression_checkpoints.csv"] = _paper_regression_checkpoints(
            regression_fit, f_tests
        )
    artifact_paths: list[Path] = []
    for filename, frame in artifacts.items():
        destination = paths.outputs / filename
        write_csv(frame, destination)
        artifact_paths.append(destination)

    figure_paths = generate_figures(
        paths,
        data,
        distances,
        quality,
        regression_fit,
        social_coefficients,
        full_coefficients,
        standardized_coefficients,
    )
    artifact_paths.extend(figure_paths)

    manifest = _manifest(source, outcome, data, model_description)
    manifest["artifacts"] = hash_paths(artifact_paths, PROJECT_ROOT)
    atomic_write_json(paths.outputs / "run_manifest.json", manifest)
    return {
        "source": source,
        "question": outcome.var,
        "outputs": str(paths.outputs),
        "figures": str(paths.figures),
        "overall_nemd_ntp": ntp_overall,
        "overall_nemd_fa": fa_overall,
        "coverage": data.coverage.as_dict(),
        "artifact_count": len(artifact_paths),
    }


def _manifest(
    source: str,
    question: Question,
    data: PreparedData,
    model_description: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "source": source,
        "seed": GLOBAL_SEED,
        "hardware": hardware_summary(),
        "parameters": {
            "question": question.var,
            "question_label": question.label,
            "answer_levels": question.levels,
            "model": model_description,
            "generation_modes": ["ntp", "fa"],
            "random_baseline_replicates": 20,
            "center_holdout_size": 20,
            "distances": list(DISTANCES),
            "kl_epsilon": KL_ZERO_PROPORTION_SMOOTHING,
            "mmd_bandwidth": MMD_BANDWIDTH_IN_CATEGORY_STEPS,
        },
        "coverage": data.coverage.as_dict(),
        "artifacts": {},
    }


def _empty_analysis(
    source: str,
    question: Question,
    paths: RunPaths,
    data: PreparedData,
    model_description: str,
) -> dict[str, Any]:
    manifest = _manifest(source, question, data, model_description)
    manifest["status"] = "no_scorable_subpopulations"
    atomic_write_json(paths.outputs / "run_manifest.json", manifest)
    return {
        "source": source,
        "question": question.var,
        "outputs": str(paths.outputs),
        "figures": str(paths.figures),
        "overall_nemd_ntp": None,
        "overall_nemd_fa": None,
        "coverage": data.coverage.as_dict(),
        "artifact_count": 0,
        "status": "no_scorable_subpopulations",
    }
