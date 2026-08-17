from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from machine_bias_reproduction.data import group_responses, load_subpops, load_wvs
from machine_bias_reproduction.figures import (
    NEMD_BREAKS,
    annotate_quality_bands,
    classical_mds,
    log_nemd_density,
    mds_block,
    save_plate,
)
from machine_bias_reproduction.metrics import nemd, pairwise_nemd_matrix
from machine_bias_reproduction.plates import MODEL_TONE, SURVEY_TONE, magnitude_steps
from machine_bias_reproduction.questions import QUESTIONS, Question, one_hot

from .load import CSV_ROOT, robustness_root

PROMPT_STRATEGIES = {
    "Interview": None,
    "1st person": "NTP-GPT-4T-1stPers",
    "Chat": "NTP-GPT-4T-Chat",
}
TEMPERATURES = {
    "0.3": "FA-Mistral-7B-t03",
    "0.7": "FA-Mistral-7B-t07",
    "1.0": "FA-Mistral-7B-t10",
    "1.2": "FA-Mistral-7B-t12",
    "1.5": "FA-Mistral-7B-t15",
}

Embeddings = dict[str, tuple[pd.DataFrame, dict[str, pd.DataFrame]]]


def _profile_key(stem: str) -> str:
    return stem.split("§", 1)[1] if stem.startswith("en§") else stem


def _read_ntp_directory(directory: Path, question: Question) -> pd.DataFrame | None:
    root = directory / question.var
    if not root.is_dir():
        return None
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("*.txt")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        record: dict[str, object] = {str(key): value for key, value in frame.iloc[0].items()}
        record["profile"] = _profile_key(path.stem)
        rows.append(record)
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    answers = [column for column in frame.columns if column not in {"profile", "mass"}]
    frame = frame.rename(columns=dict(zip(answers, question.answer_columns, strict=False)))
    return frame


def _read_fa_directory(directory: Path, question: Question) -> pd.DataFrame | None:
    root = directory / question.var
    if not root.is_dir():
        return None
    rows = [
        {"id": path.stem, question.var: path.read_text(encoding="utf-8").strip()}
        for path in sorted(root.glob("*.txt"))
    ]
    return pd.DataFrame(rows) if rows else None


def _subpopulation_props(responses: pd.DataFrame, subpops: pd.Series) -> pd.DataFrame:
    return group_responses(responses, subpops).dropna()


def _wvs_props(question: Question, mask: pd.Series | None = None) -> pd.DataFrame:
    wvs = load_wvs()
    subpops = load_subpops()["subpop"]
    if mask is not None:
        wvs = wvs[mask]
        subpops = subpops[mask]
    responses = one_hot(
        question.normalize(wvs[question.var]), question.wvs_labels, question.answer_columns
    )
    return _subpopulation_props(responses, subpops)


def _strategy_props(question: Question) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    wvs = load_wvs()
    subpops = load_subpops()["subpop"]
    frames: dict[str, pd.DataFrame] = {}
    for label, directory in PROMPT_STRATEGIES.items():
        if directory is None:
            path = CSV_ROOT / f"NTP-GPT-4T-{question.var}.csv"
            frame = pd.read_csv(path) if path.is_file() else None
            if frame is not None:
                frame = question.ntp_answers(frame.set_index("profile")).reset_index()
        else:
            frame = _read_ntp_directory(robustness_root() / directory, question)
        if frame is not None and not frame.empty:
            frames[label] = frame.set_index("profile")
    if len(frames) < 2:
        return pd.DataFrame(), {}
    shared = set.intersection(*(set(frame.index) for frame in frames.values()))
    if not shared:
        return pd.DataFrame(), {}
    mask = wvs["profile"].isin(shared)
    wvs_props = _wvs_props(question, mask)
    columns = list(question.answer_columns)
    resolved: dict[str, pd.DataFrame] = {}
    for label, frame in frames.items():
        matched = frame.reindex(wvs.loc[mask, "profile"])[columns].reset_index(drop=True)
        resolved[label] = (
            _subpopulation_props(matched, subpops[mask]).reindex(wvs_props.index).dropna()
        )
    return wvs_props, resolved


def table_s6() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for var, question in QUESTIONS.items():
        wvs_props, resolved = _strategy_props(question)
        for label, props in resolved.items():
            common = wvs_props.index.intersection(props.index)
            if common.empty:
                continue
            values = nemd(
                wvs_props.loc[common].to_numpy(dtype=np.float64),
                props.loc[common].to_numpy(dtype=np.float64),
            )
            rows.append(
                {
                    "question": var,
                    "strategy": label,
                    "subpopulations": len(common),
                    "median_nEMD": float(np.median(np.atleast_1d(values))),
                    "mean_nEMD": float(np.mean(np.atleast_1d(values))),
                }
            )
    return pd.DataFrame(rows)


def table_s7() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for var, question in QUESTIONS.items():
        quant = _read_ntp_directory(robustness_root() / "NTP-Mistral-7B-quant", question)
        unquant = _read_ntp_directory(robustness_root() / "NTP-Mistral-7B-unquant", question)
        if quant is None or unquant is None:
            continue
        columns = list(question.answer_columns)
        left = quant.set_index("profile")
        right = unquant.set_index("profile")
        shared = left.index.intersection(right.index)
        if shared.empty:
            continue
        values = np.atleast_1d(
            nemd(
                left.loc[shared, columns].to_numpy(dtype=np.float64),
                right.loc[shared, columns].to_numpy(dtype=np.float64),
            )
        )
        rows.append(
            {
                "question": var,
                "prompts": int(shared.size),
                "mean_nEMD": float(values.mean()),
                "median_nEMD": float(np.median(values)),
                "p90_nEMD": float(np.percentile(values, 90)),
                "max_nEMD": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def _temperature_props(question: Question) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    wvs = load_wvs()
    subpops = load_subpops()["subpop"]
    props: dict[str, pd.DataFrame] = {}
    covered: set[str] | None = None
    for label, directory in TEMPERATURES.items():
        frame = _read_fa_directory(robustness_root() / directory, question)
        if frame is None or frame.empty:
            continue
        ids = set(frame["id"])
        covered = ids if covered is None else covered & ids
        props[label] = frame.set_index("id")
    if not props or covered is None:
        return pd.DataFrame(), {}
    mask = wvs["id"].isin(covered)
    wvs_props = _wvs_props(question, mask)
    resolved: dict[str, pd.DataFrame] = {}
    for label, frame in props.items():
        matched = frame.reindex(wvs.loc[mask, "id"])[question.var].reset_index(drop=True)
        responses = one_hot(
            question.normalize(matched), question.fa_answers, question.answer_columns
        )
        grouped = _subpopulation_props(responses, subpops[mask])
        resolved[label] = grouped.reindex(wvs_props.index).dropna()
    return wvs_props, resolved


def _strategy_mds_grid(
    embeddings: Embeddings,
    labels: list[str],
    stem: Path,
    *,
    marker_size: int,
    survey_alpha: float,
    cell_inches: float,
    header: str,
) -> list[Path]:
    questions = [var for var in QUESTIONS if var in embeddings]
    figure, axes = plt.subplots(
        len(questions),
        len(labels),
        figsize=(cell_inches * len(labels), cell_inches * len(questions)),
        squeeze=False,
    )
    for row, var in enumerate(questions):
        wvs_props, props = embeddings[var]
        present = [label for label in labels if label in props and not props[label].empty]
        common = wvs_props.index
        for label in present:
            common = common.intersection(props[label].index)
        if common.empty:
            for axis in axes[row]:
                axis.set_axis_off()
            continue
        blocks = [wvs_props.loc[common].to_numpy(dtype=np.float64)] * len(present)
        blocks.extend(props[label].loc[common].to_numpy(dtype=np.float64) for label in present)
        coordinates = classical_mds(pairwise_nemd_matrix(np.vstack(blocks)))
        count = len(common)
        for column, label in enumerate(labels):
            axis = axes[row][column]
            if label not in present:
                axis.set_axis_off()
                continue
            index = present.index(label)
            survey = mds_block(coordinates, index, count)
            axis.scatter(
                survey[:, 0],
                survey[:, 1],
                s=marker_size,
                alpha=survey_alpha,
                marker="^",
                color=SURVEY_TONE.fill,
                edgecolors=SURVEY_TONE.ink,
                linewidths=0.3,
            )
            model = mds_block(coordinates, len(present) + index, count)
            axis.scatter(
                model[:, 0],
                model[:, 1],
                s=marker_size,
                alpha=0.6,
                color=MODEL_TONE.fill,
                edgecolors=MODEL_TONE.ink,
                linewidths=0.3,
            )
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_xlabel(header.format(label), fontsize=9, labelpad=6)
                axis.xaxis.set_label_position("top")
            if column == 0:
                axis.set_ylabel(QUESTIONS[var].label, fontsize=9)
    figure.tight_layout()
    return save_plate(figure, stem)


def figures_s13_s14(destination: Path) -> tuple[list[Path], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    embeddings: Embeddings = {}
    for var, question in QUESTIONS.items():
        wvs_props, props = _temperature_props(question)
        if not props:
            continue
        embeddings[var] = (wvs_props, props)
        for label, frame in props.items():
            common = wvs_props.index.intersection(frame.index)
            values = np.atleast_1d(
                nemd(
                    wvs_props.loc[common].to_numpy(dtype=np.float64),
                    frame.loc[common].to_numpy(dtype=np.float64),
                )
            )
            rows.extend(
                {"question": var, "temperature": label, "subpopulation": name, "nEMD": float(value)}
                for name, value in zip(common, values, strict=True)
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return [], frame

    questions = [var for var in QUESTIONS if var in set(frame["question"])]
    palette = magnitude_steps(len(TEMPERATURES), span=(0.1, 0.9))
    figure, axes = plt.subplots(
        1, len(questions), figsize=(4.6 * len(questions), 5.0), sharey=True, squeeze=False
    )
    for axis, var in zip(axes[0], questions, strict=True):
        subset = frame[frame["question"] == var]
        for index, label in enumerate(TEMPERATURES):
            values = subset.loc[subset["temperature"] == label, "nEMD"].to_numpy(dtype=np.float64)
            if not np.any(values > 0):
                continue
            grid, density = log_nemd_density(values)
            axis.plot(grid, density, label=f"T={label}", color=palette[index], linewidth=1.5)
        axis.set_xscale("log")
        axis.set_xlim(0.01, 1.0)
        axis.set_xticks(NEMD_BREAKS, [f"{threshold:.2f}" for threshold in NEMD_BREAKS])
        axis.set_ylim(bottom=0)
        annotate_quality_bands(axis)
        axis.set_xlabel(f"nEMD (log scale) — {QUESTIONS[var].label}")
    axes[0][0].set_ylabel("Density (per natural-log unit)")
    axes[0][0].legend(loc="upper left")
    figure.tight_layout()
    produced = save_plate(figure, destination / "Figure-S13-temperature-nEMD-density")

    produced.extend(
        _strategy_mds_grid(
            embeddings,
            list(TEMPERATURES),
            destination / "Figure-S14-temperature-MDS",
            marker_size=9,
            survey_alpha=0.35,
            cell_inches=2.9,
            header="T={}",
        )
    )
    return produced, frame


def figure_s12(destination: Path) -> list[Path]:
    embeddings: Embeddings = {}
    for var, question in QUESTIONS.items():
        wvs_props, resolved = _strategy_props(question)
        if resolved:
            embeddings[var] = (wvs_props, resolved)
    if not embeddings:
        return []
    return _strategy_mds_grid(
        embeddings,
        list(PROMPT_STRATEGIES),
        destination / "Figure-S12-prompting-strategy-MDS",
        marker_size=10,
        survey_alpha=0.6,
        cell_inches=3.1,
        header="{}",
    )
