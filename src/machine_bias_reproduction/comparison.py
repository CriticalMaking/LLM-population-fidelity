from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from .config import OUTPUTS_ROOT
from .figures import save_plate
from .io_utils import sha256_file, write_csv
from .plates import GRID, INK, MUTED_INK, slot_tone
from .questions import DEFAULT_QUESTION, QUESTION_NAMES

AGREEMENT_TONE = slot_tone(3)
DIVERGENCE_TONE = slot_tone(0)
POOLED_TONE = slot_tone(4)

FloatArray = npt.NDArray[np.float64]

DEFAULT_METHODS = ("NTP", "FA")
DEFAULT_EQUIVALENCE_MARGIN = 0.005
DEFAULT_ALPHA = 0.05
DEFAULT_PERMUTATIONS = 100_000
DEFAULT_SEED = 20_240_110
EXACT_PERMUTATION_MAX_CLUSTERS = 18


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"required result table not found: {path}")
    frame = pd.read_csv(path)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return frame


def country_wave_labels(subpopulations: pd.Series) -> pd.Series:
    extracted = subpopulations.astype(str).str.extract(r"^(?P<country>.+?) (?P<year>\d{4})(?: |$)")
    missing = extracted.isna().any(axis=1)
    if missing.any():
        examples = subpopulations.loc[missing].astype(str).head(3).tolist()
        raise ValueError(f"could not parse country and wave from subpopulations: {examples}")
    return extracted["country"] + " " + extracted["year"]


def holm_adjust(p_values: npt.ArrayLike) -> FloatArray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.isnan(values).any():
        raise ValueError("p-values must be a one-dimensional array without missing values")
    order = np.argsort(values)
    adjusted_sorted = np.empty(len(values), dtype=np.float64)
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * float(values[index]))
        running_max = max(running_max, candidate)
        adjusted_sorted[rank] = running_max
    adjusted = np.empty(len(values), dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return adjusted


def _cluster_robust_mean(
    differences: FloatArray,
    clusters: pd.Series,
    *,
    alpha: float,
) -> dict[str, float]:
    if len(differences) != len(clusters):
        raise ValueError("differences and clusters must have the same length")
    cluster_count = int(clusters.nunique())
    if cluster_count < 3:
        raise ValueError("at least three country-wave clusters are required")
    mean = float(np.mean(differences))
    residuals = differences - mean
    residual_frame = pd.DataFrame({"cluster": clusters.to_numpy(), "residual": residuals})
    cluster_sums = residual_frame.groupby("cluster", sort=True)["residual"].sum().to_numpy()
    variance = cluster_count / (cluster_count - 1) * float(np.sum(cluster_sums**2))
    variance /= len(differences) ** 2
    standard_error = math.sqrt(max(0.0, variance))
    degrees_freedom = cluster_count - 1
    critical = float(stats.t.ppf(1 - alpha / 2, degrees_freedom))
    if standard_error == 0:
        p_value = 1.0 if mean == 0 else 0.0
    else:
        p_value = float(2 * stats.t.sf(abs(mean / standard_error), degrees_freedom))
    return {
        "mean": mean,
        "cluster_robust_se": standard_error,
        "cluster_degrees_freedom": float(degrees_freedom),
        "cluster_t_p_value": p_value,
        "ci95_low": mean - critical * standard_error,
        "ci95_high": mean + critical * standard_error,
    }


def _cluster_sign_flip_test(
    differences: FloatArray,
    clusters: pd.Series,
    *,
    permutations: int,
    seed: int,
) -> tuple[float, int, str]:
    frame = pd.DataFrame({"cluster": clusters.to_numpy(), "difference": differences})
    totals = frame.groupby("cluster", sort=True)["difference"].sum().to_numpy(dtype=np.float64)
    cluster_count = len(totals)
    observed = abs(float(np.sum(totals) / len(differences)))
    tolerance = np.finfo(np.float64).eps * max(1.0, observed) * 16

    if cluster_count <= EXACT_PERMUTATION_MAX_CLUSTERS:
        exceedances = 0
        permutation_count = 2**cluster_count
        for exact_signs in itertools.product((-1.0, 1.0), repeat=cluster_count):
            statistic = abs(float(np.dot(totals, exact_signs) / len(differences)))
            exceedances += int(statistic >= observed - tolerance)
        return exceedances / permutation_count, permutation_count, "exact"

    if permutations < 1:
        raise ValueError("permutations must be positive")
    rng = np.random.default_rng(seed)
    exceedances = 0
    completed = 0
    batch_size = 10_000
    while completed < permutations:
        current = min(batch_size, permutations - completed)
        random_signs = rng.choice((-1.0, 1.0), size=(current, cluster_count))
        statistics = np.abs(random_signs @ totals / len(differences))
        exceedances += int(np.count_nonzero(statistics >= observed - tolerance))
        completed += current
    return (exceedances + 1) / (permutations + 1), permutations, "monte_carlo"


def _tost(
    mean: float,
    standard_error: float,
    degrees_freedom: int,
    margin: float,
) -> tuple[float, float, float]:
    if standard_error == 0:
        lower_p = 0.0 if mean > -margin else 1.0
        upper_p = 0.0 if mean < margin else 1.0
    else:
        lower_statistic = (mean + margin) / standard_error
        upper_statistic = (mean - margin) / standard_error
        lower_p = float(stats.t.sf(lower_statistic, degrees_freedom))
        upper_p = float(stats.t.cdf(upper_statistic, degrees_freedom))
    return lower_p, upper_p, max(lower_p, upper_p)


def _safe_wilcoxon(differences: FloatArray) -> float:
    if np.allclose(differences, 0):
        return 1.0
    return float(
        stats.wilcoxon(
            differences,
            zero_method="wilcox",
            alternative="two-sided",
        ).pvalue
    )


def _paired_effect_size(differences: FloatArray) -> float:
    standard_deviation = float(np.std(differences, ddof=1))
    mean = float(np.mean(differences))
    if standard_deviation == 0:
        return 0.0 if mean == 0 else math.copysign(math.inf, mean)
    return mean / standard_deviation


def _verdict(different: bool, equivalent: bool) -> str:
    if different and equivalent:
        return "detectable_but_within_equivalence_margin"
    if different:
        return "statistically_different"
    if equivalent:
        return "no_detectable_change_and_practically_equivalent"
    return "inconclusive"


def paired_method_inference(
    pairs: pd.DataFrame,
    *,
    method: str,
    alpha: float,
    equivalence_margin: float,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    differences = pairs["nEMD_fresh"].to_numpy(dtype=np.float64) - pairs["nEMD_archived"].to_numpy(
        dtype=np.float64
    )
    clusters = pairs["country_wave"]
    robust = _cluster_robust_mean(differences, clusters, alpha=alpha)
    permutation_p, permutation_count, permutation_type = _cluster_sign_flip_test(
        differences,
        clusters,
        permutations=permutations,
        seed=seed,
    )
    degrees_freedom = int(robust["cluster_degrees_freedom"])
    tost_lower, tost_upper, tost_p = _tost(
        robust["mean"],
        robust["cluster_robust_se"],
        degrees_freedom,
        equivalence_margin,
    )
    archived = pairs["nEMD_archived"].to_numpy(dtype=np.float64)
    fresh = pairs["nEMD_fresh"].to_numpy(dtype=np.float64)
    ties = np.isclose(differences, 0.0, rtol=0.0, atol=1e-12)
    archive_mean = float(np.mean(archived))
    relative_change = 100 * float(robust["mean"]) / archive_mean if archive_mean != 0 else math.nan
    paired_t = stats.ttest_rel(fresh, archived)
    return {
        "method": method,
        "n_pairs": len(pairs),
        "n_country_wave_clusters": int(clusters.nunique()),
        "archived_mean_nEMD": archive_mean,
        "fresh_mean_nEMD": float(np.mean(fresh)),
        "mean_change_fresh_minus_archived": robust["mean"],
        "relative_change_percent": relative_change,
        "median_change": float(np.median(differences)),
        "cluster_robust_se": robust["cluster_robust_se"],
        "cluster_degrees_freedom": degrees_freedom,
        "ci95_low": robust["ci95_low"],
        "ci95_high": robust["ci95_high"],
        "cluster_t_p_value": robust["cluster_t_p_value"],
        "cluster_permutation_p_value": permutation_p,
        "permutation_type": permutation_type,
        "permutations_evaluated": permutation_count,
        "equivalence_margin": equivalence_margin,
        "tost_lower_p_value": tost_lower,
        "tost_upper_p_value": tost_upper,
        "tost_p_value": tost_p,
        "practically_equivalent": tost_p < alpha,
        "paired_effect_size_dz": _paired_effect_size(differences),
        "archived_fresh_pearson_r": float(np.corrcoef(archived, fresh)[0, 1]),
        "archived_fresh_spearman_rho": float(stats.spearmanr(archived, fresh).statistic),
        "fresh_better_count": int(np.count_nonzero((differences < 0) & ~ties)),
        "unchanged_count": int(np.count_nonzero(ties)),
        "fresh_worse_count": int(np.count_nonzero((differences > 0) & ~ties)),
        "unclustered_paired_t_p_value": float(paired_t.pvalue),
        "unclustered_wilcoxon_p_value": _safe_wilcoxon(differences),
    }


def _align_subpopulation_distances(
    archived_dir: Path,
    fresh_dir: Path,
    methods: Sequence[str],
) -> pd.DataFrame:
    columns = ["method", "subpopulation", "nEMD"]
    required = set(columns)
    archived = _read_csv(archived_dir / "subpopulation_distances.csv", required)
    fresh = _read_csv(fresh_dir / "subpopulation_distances.csv", required)
    archived = archived.loc[archived["method"].isin(methods), columns].copy()
    fresh = fresh.loc[fresh["method"].isin(methods), columns].copy()
    keys = ["method", "subpopulation"]
    for label, frame in (("archived", archived), ("fresh", fresh)):
        duplicates = frame.duplicated(keys, keep=False)
        if duplicates.any():
            raise ValueError(f"{label} subpopulation table has duplicate method/name pairs")
    pairs = archived.merge(
        fresh,
        on=keys,
        how="outer",
        suffixes=("_archived", "_fresh"),
        validate="one_to_one",
        indicator=True,
    )
    unmatched = pairs["_merge"] != "both"
    if unmatched.any():
        examples = pairs.loc[unmatched, [*keys, "_merge"]].head(5).to_dict("records")
        raise ValueError(f"archived and fresh subpopulation rows do not align: {examples}")
    found_methods = set(pairs["method"])
    missing_methods = set(methods).difference(found_methods)
    if missing_methods:
        raise ValueError(f"methods missing from comparison: {', '.join(sorted(missing_methods))}")
    pairs = pairs.drop(columns="_merge")
    pairs["country_wave"] = country_wave_labels(pairs["subpopulation"])
    pairs["change_fresh_minus_archived"] = pairs["nEMD_fresh"] - pairs["nEMD_archived"]
    tolerance = 1e-12
    pairs["direction"] = np.select(
        [
            pairs["change_fresh_minus_archived"] < -tolerance,
            pairs["change_fresh_minus_archived"] > tolerance,
        ],
        ["fresh_better", "fresh_worse"],
        default="unchanged",
    )
    return pairs.sort_values(["method", "subpopulation"]).reset_index(drop=True)


def _cluster_differences(pairs: pd.DataFrame) -> pd.DataFrame:
    return (
        pairs.groupby(["method", "country_wave"], sort=True)["change_fresh_minus_archived"]
        .agg(subpopulations="size", mean_change="mean", median_change="median", sd_change="std")
        .reset_index()
    )


def _summary_metric_changes(archived_dir: Path, fresh_dir: Path) -> pd.DataFrame:
    required = {"metric", "method", "value"}
    archived = _read_csv(archived_dir / "summary_metrics.csv", required)
    fresh = _read_csv(fresh_dir / "summary_metrics.csv", required)
    merged = archived.merge(
        fresh,
        on=["metric", "method"],
        how="outer",
        suffixes=("_archived", "_fresh"),
        validate="one_to_one",
        indicator=True,
    )
    if (merged["_merge"] != "both").any():
        raise ValueError("archived and fresh summary metric rows do not align")
    merged = merged.drop(columns="_merge")
    merged["absolute_change"] = merged["value_fresh"] - merged["value_archived"]
    merged["relative_change_percent"] = np.where(
        merged["value_archived"] != 0,
        100 * merged["absolute_change"] / merged["value_archived"],
        np.nan,
    )
    return merged


def _response_distribution_changes(archived_dir: Path, fresh_dir: Path) -> pd.DataFrame:
    archived_frame = _read_csv(archived_dir / "response_distributions.csv", {"method"})
    fresh_frame = _read_csv(fresh_dir / "response_distributions.csv", {"method"})
    answer_columns = sorted(column for column in archived_frame.columns if column != "method")
    fresh_answer_columns = sorted(column for column in fresh_frame.columns if column != "method")
    if answer_columns != fresh_answer_columns:
        raise ValueError(
            "archived and fresh response_distributions.csv have different answer "
            f"columns: {answer_columns} vs {fresh_answer_columns}"
        )
    archived = archived_frame.melt(
        id_vars="method",
        value_vars=answer_columns,
        var_name="answer",
        value_name="proportion_archived",
    )
    fresh = fresh_frame.melt(
        id_vars="method",
        value_vars=answer_columns,
        var_name="answer",
        value_name="proportion_fresh",
    )
    merged = archived.merge(
        fresh,
        on=["method", "answer"],
        validate="one_to_one",
    )
    merged["change_fresh_minus_archived"] = (
        merged["proportion_fresh"] - merged["proportion_archived"]
    )
    return merged


def _regression_fit_changes(archived_dir: Path, fresh_dir: Path) -> pd.DataFrame:
    keys = ["mode", "model"]
    metrics = ["adjusted_r_squared", "bic_r", "aic"]
    required = set(keys + metrics)
    archived = _read_csv(archived_dir / "regression_fit.csv", required)
    fresh = _read_csv(fresh_dir / "regression_fit.csv", required)
    archived_long = archived.melt(
        id_vars=keys,
        value_vars=metrics,
        var_name="metric",
        value_name="value_archived",
    )
    fresh_long = fresh.melt(
        id_vars=keys,
        value_vars=metrics,
        var_name="metric",
        value_name="value_fresh",
    )
    merged = archived_long.merge(
        fresh_long,
        on=[*keys, "metric"],
        validate="one_to_one",
    )
    merged["change_fresh_minus_archived"] = merged["value_fresh"] - merged["value_archived"]
    return merged


def _coefficient_changes(archived_dir: Path, fresh_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for table in (
        "social_coefficients.csv",
        "full_coefficients.csv",
        "full_standardized_coefficients.csv",
    ):
        required = {
            "mode",
            "model",
            "predictor",
            "estimate",
            "ci_low",
            "ci_high",
            "p_value",
        }
        archived = _read_csv(archived_dir / table, required)
        fresh = _read_csv(fresh_dir / table, required)
        keys = ["mode", "model", "predictor"]
        columns = [*keys, "estimate", "ci_low", "ci_high", "p_value"]
        merged = archived.loc[:, columns].merge(
            fresh.loc[:, columns],
            on=keys,
            suffixes=("_archived", "_fresh"),
            validate="one_to_one",
        )
        merged.insert(0, "table", table)
        merged["estimate_change_fresh_minus_archived"] = (
            merged["estimate_fresh"] - merged["estimate_archived"]
        )
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def _format_p(value: float) -> str:
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def _comparison_figure(
    pairs: pd.DataFrame,
    clusters: pd.DataFrame,
    tests: pd.DataFrame,
    path_stem: Path,
) -> list[Path]:
    methods = tests["method"].tolist()
    figure, axes = plt.subplots(
        len(methods),
        2,
        figsize=(13, 5.7 * len(methods)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, method in enumerate(methods):
        method_pairs = pairs[pairs["method"] == method]
        method_clusters = clusters[clusters["method"] == method].sort_values("mean_change")
        result = tests[tests["method"] == method].iloc[0]

        scatter = axes[row_index, 0]
        scatter.scatter(
            method_pairs["nEMD_archived"],
            method_pairs["nEMD_fresh"],
            s=18,
            alpha=0.7,
            color=AGREEMENT_TONE.fill,
            edgecolors=AGREEMENT_TONE.ink,
            linewidths=0.4,
        )
        limits = (
            float(
                min(
                    method_pairs["nEMD_archived"].min(),
                    method_pairs["nEMD_fresh"].min(),
                )
            ),
            float(
                max(
                    method_pairs["nEMD_archived"].max(),
                    method_pairs["nEMD_fresh"].max(),
                )
            ),
        )
        padding = max(0.005, (limits[1] - limits[0]) * 0.04)
        lower, upper = limits[0] - padding, limits[1] + padding
        scatter.plot(
            [lower, upper], [lower, upper], color=MUTED_INK, linestyle=(0, (4, 2)), linewidth=0.8
        )
        scatter.set(xlim=(lower, upper), ylim=(lower, upper))
        scatter.set_aspect("equal", adjustable="box")
        scatter.set_xlabel("Archived subpopulation nEMD")
        scatter.set_ylabel("Fresh subpopulation nEMD")
        scatter.text(
            0.03,
            0.97,
            f"{method}\n"
            f"Pearson r = {result['archived_fresh_pearson_r']:.4f}\n"
            f"Spearman $\\rho$ = {result['archived_fresh_spearman_rho']:.4f}\n"
            f"mean $\\Delta$ = {result['mean_change_fresh_minus_archived']:.6f}\n"
            f"Holm p = {_format_p(result['holm_adjusted_p_value'])}",
            transform=scatter.transAxes,
            va="top",
            fontsize=8.5,
            color=MUTED_INK,
        )

        cluster_axis = axes[row_index, 1]
        improved = method_clusters["mean_change"] < 0
        fills = np.where(improved, AGREEMENT_TONE.fill, DIVERGENCE_TONE.fill)
        inks = np.where(improved, AGREEMENT_TONE.ink, DIVERGENCE_TONE.ink)
        y_positions = np.arange(len(method_clusters))
        cluster_axis.scatter(
            method_clusters["mean_change"],
            y_positions,
            s=42,
            c=fills,
            edgecolors=inks,
            linewidths=0.7,
            zorder=3,
        )
        cluster_axis.hlines(
            y_positions,
            0,
            method_clusters["mean_change"],
            colors=inks,
            alpha=0.55,
            linewidth=1.2,
        )
        cluster_axis.axvline(0, color=INK, linewidth=0.8)
        cluster_axis.axvline(
            result["mean_change_fresh_minus_archived"],
            color=POOLED_TONE.ink,
            linestyle=(0, (4, 2)),
            linewidth=1.2,
            label=f"{method} all-subpopulation mean",
        )
        cluster_axis.set_yticks(y_positions, method_clusters["country_wave"])
        cluster_axis.set_xlabel("Mean change (fresh - archived nEMD)")
        cluster_axis.legend(loc="best")
        cluster_axis.grid(axis="x", color=GRID, linewidth=0.6)

    return save_plate(figure, path_stem, pdf_dpi=300)


def compare_runs(
    archived_dir: Path,
    fresh_dir: Path,
    output_dir: Path,
    *,
    methods: Sequence[str] = DEFAULT_METHODS,
    alpha: float = DEFAULT_ALPHA,
    equivalence_margin: float = DEFAULT_EQUIVALENCE_MARGIN,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5")
    if equivalence_margin <= 0:
        raise ValueError("equivalence margin must be positive")
    if not methods:
        raise ValueError("at least one method is required")

    archived_dir = archived_dir.resolve()
    fresh_dir = fresh_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = _align_subpopulation_distances(archived_dir, fresh_dir, methods)
    inference_rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(methods):
        method_pairs = pairs[pairs["method"] == method].reset_index(drop=True)
        inference_rows.append(
            paired_method_inference(
                method_pairs,
                method=method,
                alpha=alpha,
                equivalence_margin=equivalence_margin,
                permutations=permutations,
                seed=seed + method_index,
            )
        )
    tests = pd.DataFrame(inference_rows)
    tests["holm_adjusted_p_value"] = holm_adjust(tests["cluster_permutation_p_value"].to_numpy())
    tests["difference_significant"] = tests["holm_adjusted_p_value"] < alpha
    tests["verdict"] = [
        _verdict(bool(different), bool(equivalent))
        for different, equivalent in zip(
            tests["difference_significant"],
            tests["practically_equivalent"],
            strict=True,
        )
    ]

    clusters = _cluster_differences(pairs)
    summary = _summary_metric_changes(archived_dir, fresh_dir)
    distributions = _response_distribution_changes(archived_dir, fresh_dir)
    regression = _regression_fit_changes(archived_dir, fresh_dir)
    coefficients = _coefficient_changes(archived_dir, fresh_dir)

    artifacts = {
        "statistical_tests.csv": tests,
        "subpopulation_differences.csv": pairs,
        "cluster_differences.csv": clusters,
        "summary_metric_changes.csv": summary,
        "response_distribution_changes.csv": distributions,
        "regression_fit_changes.csv": regression,
        "coefficient_changes.csv": coefficients,
    }
    artifact_paths: list[Path] = []
    for filename, frame in artifacts.items():
        destination = output_dir / filename
        write_csv(frame, destination)
        artifact_paths.append(destination)

    artifact_paths.extend(
        _comparison_figure(
            pairs,
            clusters,
            tests,
            output_dir / "fig_archived_vs_fresh",
        )
    )

    input_names = (
        "subpopulation_distances.csv",
        "summary_metrics.csv",
        "response_distributions.csv",
        "regression_fit.csv",
        "social_coefficients.csv",
        "full_coefficients.csv",
        "full_standardized_coefficients.csv",
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "parameters": {
            "methods": list(methods),
            "alpha": alpha,
            "equivalence_margin_nEMD": equivalence_margin,
            "cluster": "country-survey wave",
            "permutations_requested_for_monte_carlo": permutations,
            "seed": seed,
        },
        "inputs": {
            "archived_directory": str(archived_dir),
            "fresh_directory": str(fresh_dir),
            "sha256": {f"archived/{name}": sha256_file(archived_dir / name) for name in input_names}
            | {f"fresh/{name}": sha256_file(fresh_dir / name) for name in input_names},
        },
        "artifacts": {path.name: sha256_file(path) for path in artifact_paths},
    }
    manifest_path = output_dir / "comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "output_directory": str(output_dir),
        "methods": {
            str(row["method"]): {
                "mean_change": float(row["mean_change_fresh_minus_archived"]),
                "holm_p_value": float(row["holm_adjusted_p_value"]),
                "equivalent": bool(row["practically_equivalent"]),
                "verdict": str(row["verdict"]),
            }
            for row in tests.to_dict(orient="records")
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare archived and fresh subpopulation nEMD with paired, "
            "country-wave-clustered inference."
        )
    )
    parser.add_argument(
        "--question",
        choices=QUESTION_NAMES,
        default=DEFAULT_QUESTION,
        help=f"outcome question to compare (default: {DEFAULT_QUESTION})",
    )
    parser.add_argument(
        "--archived",
        type=Path,
        help="archived analysis output directory (default: outputs/archived/<question>)",
    )
    parser.add_argument(
        "--fresh",
        type=Path,
        help="fresh analysis output directory (default: outputs/fresh/<question>)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="destination for comparison tables and figure",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="methods from subpopulation_distances.csv to compare",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--equivalence-margin",
        type=float,
        default=DEFAULT_EQUIVALENCE_MARGIN,
        help=(
            "smallest practically important absolute nEMD change "
            f"(default: {DEFAULT_EQUIVALENCE_MARGIN:g})"
        ),
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=DEFAULT_PERMUTATIONS,
        help="Monte Carlo draws when there are too many clusters for exact enumeration",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parser().parse_args(arguments)
    parsed.archived = parsed.archived or OUTPUTS_ROOT / "archived" / parsed.question
    parsed.fresh = parsed.fresh or OUTPUTS_ROOT / "fresh" / parsed.question
    parsed.output = parsed.output or OUTPUTS_ROOT / "comparison" / parsed.question
    result = compare_runs(
        parsed.archived,
        parsed.fresh,
        parsed.output,
        methods=parsed.methods,
        alpha=parsed.alpha,
        equivalence_margin=parsed.equivalence_margin,
        permutations=parsed.permutations,
        seed=parsed.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
