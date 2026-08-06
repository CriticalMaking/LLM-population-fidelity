"""The paper's robustness checks: Tables S6, S7 and Figures S12, S13, S14."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from machine_bias_reproduction.data import load_subpops, load_wvs
from machine_bias_reproduction.figures import (
    NEMD_BREAKS,
    _annotate_quality_bands,
    _classical_mds,
    _log_nemd_density,
    _save,
)
from machine_bias_reproduction.metrics import nemd, pairwise_nemd_matrix
from machine_bias_reproduction.questions import QUESTIONS, Question, one_hot

from .load import robustness_root

matplotlib.use("Agg")
from matplotlib import pyplot as plt

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


def _profile_key(stem: str) -> str:
    """Strip the language prefix the GPT-4T variants add to their filenames.

    That prefix is the only difference from the canonical key. The rest of the
    stem is left verbatim — multi-word countries keep their underscore, exactly
    as ``LLM-outputs/csv`` and ``wvs["profile"]`` spell them.
    """
    return stem.split("§", 1)[1] if stem.startswith("en§") else stem


def _read_ntp_directory(directory: Path, question: Question) -> pd.DataFrame | None:
    """Read a directory of one-row NTP probability tables into one frame."""
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
    """Read a directory of single-answer FA files into one frame."""
    root = directory / question.var
    if not root.is_dir():
        return None
    rows = [
        {"id": path.stem, question.var: path.read_text(encoding="utf-8").strip()}
        for path in sorted(root.glob("*.txt"))
    ]
    return pd.DataFrame(rows) if rows else None


def _subpopulation_props(
    responses: pd.DataFrame,
    keys: pd.Series,
    subpops: pd.Series,
) -> pd.DataFrame:
    frame = responses.copy()
    frame["name"] = subpops.to_numpy()
    answers = [column for column in frame.columns if column != "name"]
    grouped = frame.groupby("name", sort=True)[answers].mean()
    return grouped.dropna()


def _wvs_props(question: Question, mask: pd.Series | None = None) -> pd.DataFrame:
    wvs = load_wvs()
    subpops = load_subpops()["subpop"]
    if mask is not None:
        wvs = wvs[mask]
        subpops = subpops[mask]
    responses = one_hot(
        question.normalize(wvs[question.var]), question.wvs_labels, question.answer_columns
    )
    return _subpopulation_props(responses, wvs["id"], subpops)


def table_s6() -> pd.DataFrame:
    """Table S6 — median one-to-one nEMD across GPT-4T prompting strategies.

    The Interview strategy is the paper's standard prompt, so its output is the
    main ``NTP-GPT-4T`` table restricted to the profiles the variants cover.
    """
    from .load import CSV_ROOT

    wvs = load_wvs()
    subpops = load_subpops()["subpop"]
    rows: list[dict[str, object]] = []
    for var, question in QUESTIONS.items():
        frames: dict[str, pd.DataFrame] = {}
        for label, directory in PROMPT_STRATEGIES.items():
            if directory is None:
                path = CSV_ROOT / f"NTP-GPT-4T-{var}.csv"
                frame = pd.read_csv(path) if path.is_file() else None
                if frame is not None:
                    frame = question.ntp_answers(frame.set_index("profile")).reset_index()
            else:
                frame = _read_ntp_directory(robustness_root() / directory, question)
            if frame is not None and not frame.empty:
                frames[label] = frame.set_index("profile")
        if len(frames) < 2:
            continue
        shared = set.intersection(*(set(frame.index) for frame in frames.values()))
        if not shared:
            continue
        mask = wvs["profile"].isin(shared)
        wvs_props = _wvs_props(question, mask)
        columns = list(question.answer_columns)
        for label, frame in frames.items():
            matched = frame.reindex(wvs.loc[mask, "profile"])[columns].reset_index(drop=True)
            props = _subpopulation_props(matched, wvs.loc[mask, "id"], subpops[mask])
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
    """Table S7 — nEMD between quantized and unquantized Mistral-7B, per prompt."""
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
        grouped = _subpopulation_props(responses, wvs.loc[mask, "id"], subpops[mask])
        resolved[label] = grouped.reindex(wvs_props.index).dropna()
    return wvs_props, resolved


def figures_s13_s14(destination: Path) -> tuple[list[Path], pd.DataFrame]:
    """Figures S13 and S14 — nEMD density and MDS across five FA temperatures."""
    rows: list[dict[str, object]] = []
    embeddings: dict[str, tuple[pd.DataFrame, dict[str, pd.DataFrame]]] = {}
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
    palette = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(TEMPERATURES)))
    figure, axes = plt.subplots(
        1, len(questions), figsize=(4.6 * len(questions), 5.0), sharey=True, squeeze=False
    )
    for axis, var in zip(axes[0], questions, strict=True):
        subset = frame[frame["question"] == var]
        for index, label in enumerate(TEMPERATURES):
            values = subset.loc[subset["temperature"] == label, "nEMD"].to_numpy(dtype=np.float64)
            if not np.any(values > 0):
                continue
            grid, density = _log_nemd_density(values)
            axis.plot(grid, density, label=f"T={label}", color=palette[index], linewidth=1.5)
        axis.set_xscale("log")
        axis.set_xlim(0.01, 1.0)
        axis.set_xticks(NEMD_BREAKS, [f"{break_:.2f}" for break_ in NEMD_BREAKS])
        axis.set_ylim(bottom=0)
        _annotate_quality_bands(axis)
        axis.set(xlabel="nEMD (log scale)", title=QUESTIONS[var].label)
    axes[0][0].set_ylabel("Density (per natural-log unit)")
    axes[0][0].legend(frameon=False, fontsize=8, loc="upper left")
    figure.tight_layout()
    produced = _save(figure, destination / "Figure-S13-temperature-nEMD-density")

    labels = list(TEMPERATURES)
    figure, axes = plt.subplots(
        len(questions),
        len(labels),
        figsize=(2.9 * len(labels), 2.9 * len(questions)),
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
        coordinates = _classical_mds(pairwise_nemd_matrix(np.vstack(blocks)))
        count = len(common)
        for column, label in enumerate(labels):
            axis = axes[row][column]
            if label not in present:
                axis.set_axis_off()
                continue
            index = present.index(label)
            axis.scatter(
                coordinates[index * count : (index + 1) * count, 0],
                coordinates[index * count : (index + 1) * count, 1],
                s=9,
                alpha=0.35,
                marker="^",
                color="#fde725",
                edgecolors="#8a8a2a",
                linewidths=0.3,
            )
            model = coordinates[(len(present) + index) * count : (len(present) + index + 1) * count]
            axis.scatter(model[:, 0], model[:, 1], s=9, alpha=0.35, color="#440154")
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"T={label}", fontsize=9)
            if column == 0:
                axis.set_ylabel(QUESTIONS[var].label, fontsize=9)
    figure.tight_layout()
    produced.extend(_save(figure, destination / "Figure-S14-temperature-MDS"))
    return produced, frame


def figure_s12(destination: Path) -> list[Path]:
    """Figure S12 — MDS across GPT-4T prompting strategies.

    Upstream writes this plate under the wrong name
    (``Figure-S10-prompting-strategy-MDS.png``, ``6-results.R:1438``) and into
    the working directory; it is emitted here under its correct S12 name.
    """
    from .load import CSV_ROOT

    wvs = load_wvs()
    subpops = load_subpops()["subpop"]
    questions: list[str] = []
    embeddings: dict[str, tuple[pd.DataFrame, dict[str, pd.DataFrame]]] = {}
    for var, question in QUESTIONS.items():
        frames: dict[str, pd.DataFrame] = {}
        for label, directory in PROMPT_STRATEGIES.items():
            if directory is None:
                path = CSV_ROOT / f"NTP-GPT-4T-{var}.csv"
                frame = pd.read_csv(path) if path.is_file() else None
                if frame is not None:
                    frame = question.ntp_answers(frame.set_index("profile")).reset_index()
            else:
                frame = _read_ntp_directory(robustness_root() / directory, question)
            if frame is not None and not frame.empty:
                frames[label] = frame.set_index("profile")
        if len(frames) < 2:
            continue
        shared = set.intersection(*(set(frame.index) for frame in frames.values()))
        if not shared:
            continue
        mask = wvs["profile"].isin(shared)
        wvs_props = _wvs_props(question, mask)
        columns = list(question.answer_columns)
        resolved: dict[str, pd.DataFrame] = {}
        for label, frame in frames.items():
            matched = frame.reindex(wvs.loc[mask, "profile"])[columns].reset_index(drop=True)
            resolved[label] = (
                _subpopulation_props(matched, wvs.loc[mask, "id"], subpops[mask])
                .reindex(wvs_props.index)
                .dropna()
            )
        embeddings[var] = (wvs_props, resolved)
        questions.append(var)
    if not questions:
        return []

    labels = list(PROMPT_STRATEGIES)
    figure, axes = plt.subplots(
        len(questions),
        len(labels),
        figsize=(3.1 * len(labels), 3.1 * len(questions)),
        squeeze=False,
    )
    for row, var in enumerate(questions):
        wvs_props, resolved = embeddings[var]
        present = [label for label in labels if label in resolved and not resolved[label].empty]
        common = wvs_props.index
        for label in present:
            common = common.intersection(resolved[label].index)
        if common.empty:
            for axis in axes[row]:
                axis.set_axis_off()
            continue
        blocks = [wvs_props.loc[common].to_numpy(dtype=np.float64)] * len(present)
        blocks.extend(resolved[label].loc[common].to_numpy(dtype=np.float64) for label in present)
        coordinates = _classical_mds(pairwise_nemd_matrix(np.vstack(blocks)))
        count = len(common)
        for column, label in enumerate(labels):
            axis = axes[row][column]
            if label not in present:
                axis.set_axis_off()
                continue
            index = present.index(label)
            axis.scatter(
                coordinates[index * count : (index + 1) * count, 0],
                coordinates[index * count : (index + 1) * count, 1],
                s=10,
                alpha=0.35,
                marker="^",
                color="#fde725",
                edgecolors="#8a8a2a",
                linewidths=0.3,
            )
            model = coordinates[(len(present) + index) * count : (len(present) + index + 1) * count]
            axis.scatter(model[:, 0], model[:, 1], s=10, alpha=0.35, color="#440154")
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(label, fontsize=9)
            if column == 0:
                axis.set_ylabel(QUESTIONS[var].label, fontsize=9)
    figure.tight_layout()
    return _save(figure, destination / "Figure-S12-prompting-strategy-MDS")
