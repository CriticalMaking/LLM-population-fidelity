from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUPS = ("A", "B", "C", "D")
GROUP_COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#6250d6"}
POOLED_FILL = "#dcdbd6"
INK = "#2b2b2a"
INK_MUTED = "#6f6e69"
K = 101
X = np.linspace(0.0, 10.0, K)
SURVEY_MEANS = (2.5, 3.5, 4.5, 7.5)
SPREAD = 1.3
LINE = 72.27 / 25.4 * 0.75
OUTPUT = Path("figures") / "stylized_example" / "fig_pfs_toy_gate_cap.pdf"


def nemd(p, q):
    return float(np.abs(np.cumsum(p - q))[:-1].sum() / (len(p) - 1))


def make_dist(mean, spread):
    density = np.exp(-((X - mean) ** 2) / (2 * spread**2))
    return density / density.sum()


def mean_of(p):
    return float((X * p).sum())


def pair_dists(dists):
    return np.round([nemd(a, b) for a, b in combinations(dists, 2)], 10)


def spearman(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def scores(survey, model):
    survey_distance, model_distance = pair_dists(survey), pair_dists(model)
    ratio = float(np.median(model_distance) / np.median(survey_distance))
    rho = spearman(survey_distance, model_distance)
    orientation = spearman(
        np.round([mean_of(p) for p in survey], 10), np.round([mean_of(p) for p in model], 10)
    )
    accuracy = 1 - float(np.mean([nemd(a, b) for a, b in zip(survey, model, strict=True)]))
    adaptability = 0.0 if ratio == 0 else min(ratio, 1 / ratio)
    structure = 0.0 if np.isnan(rho) else max(0.0, rho)
    credit = 0.0 if np.isnan(orientation) else max(0.0, orientation)
    gated = structure if credit > 0 else 0.0
    capped = min(structure, credit)
    return {
        "accuracy": accuracy,
        "adaptability": adaptability,
        "structure": structure,
        "orientation": orientation,
        "pfs": (accuracy * adaptability * structure) ** (1 / 3),
        "pfs_gated": (accuracy * adaptability * gated) ** (1 / 3),
        "pfs_capped": (accuracy * adaptability * capped) ** (1 / 3),
    }


survey = [make_dist(mean, SPREAD) for mean in SURVEY_MEANS]
models = {
    "M4: mirrored": [p[::-1] for p in survey],
    "M5: partly reversed": [survey[2], survey[1], survey[0], survey[3]],
    "M6: ordered": [make_dist(mean, SPREAD) for mean in (3.0, 3.9, 4.8, 6.9)],
}
metrics = {name: scores(survey, model) for name, model in models.items()}
Y_TOP = max(np.mean(model, axis=0).max() for model in [survey, *models.values()]) * 1.2

ROWS = (
    ("Accuracy", "accuracy", "normal"),
    ("Adaptability", "adaptability", "normal"),
    ("Structure", "structure", "normal"),
    ("Orientation", "orientation", "normal"),
    ("PFS", "pfs", "bold"),
    ("PFS, gate", "pfs_gated", "bold"),
    ("PFS, cap", "pfs_capped", "bold"),
)


def shown(value):
    return "–" if np.isnan(value) else f"{value:.2f}".replace("-", "−")


def plot_panel(ax, dists, title, y_label=None):
    ax.fill_between(X, np.mean(dists, axis=0), color=POOLED_FILL, linewidth=0)
    placed = []
    for group, p in zip(GROUPS, dists, strict=True):
        share = p / len(dists)
        color = GROUP_COLORS[group]
        mean = mean_of(p)
        ax.fill_between(X, share, color=color, alpha=0.16, linewidth=0)
        ax.plot(X, share, color=color, linewidth=0.45 * LINE)
        ax.plot(
            [mean, mean],
            [0, np.interp(mean, X, share)],
            color=color,
            linewidth=0.35 * LINE,
            alpha=0.7,
            dashes=(2, 2),
            dash_capstyle="butt",
        )
        placed.append((mean, share[np.abs(X - mean) <= 0.5].max(), group))
    for mean, height, group in placed:
        ax.text(
            mean,
            height + 0.035 * Y_TOP,
            group,
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            color=INK,
        )
    ax.plot([0, 10], [0, 0], color=INK_MUTED, linewidth=0.3 * LINE, clip_on=False)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, Y_TOP)
    ax.set_title(title, fontsize=7.5, fontweight="bold", color=INK, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    if y_label:
        ax.plot([0, 0], [0, Y_TOP], color=INK_MUTED, linewidth=0.3 * LINE, clip_on=False)
        ax.set_ylabel(y_label, fontsize=6.5, color=INK_MUTED, labelpad=2)


def score_table(ax, row):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.45, len(ROWS) + 0.75)
    for index, (label, key, weight) in enumerate(ROWS):
        y = len(ROWS) - index + (0.2 if index < 4 else 0.0)
        ax.text(0.14, y, label, ha="left", va="center", fontsize=6.5, fontweight=weight, color=INK)
        ax.text(
            0.86,
            y,
            shown(row[key]),
            ha="right",
            va="center",
            fontsize=6.5,
            fontweight=weight,
            color=INK,
        )
    ax.plot([0.14, 0.86], [3.6, 3.6], color=INK_MUTED, linewidth=0.3 * LINE, alpha=0.6)


def build(path):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Nimbus Sans", "Liberation Sans", "Arial"],
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(5.5, 2.0),
        gridspec_kw={"height_ratios": [1, 0.78], "wspace": 0.14, "hspace": 0.06},
    )
    plot_panel(axes[0, 0], survey, "Survey (reference)", y_label="Population share")
    for ax, (name, dists) in zip(axes[0, 1:], models.items(), strict=True):
        plot_panel(ax, dists, name)
    ends = axes[1, 0]
    ends.axis("off")
    ends.set_xlim(0, 10)
    ends.set_ylim(0, 1)
    ends.text(0.05, 0.94, "Low", ha="left", va="top", fontsize=6.5, color=INK_MUTED)
    ends.text(9.95, 0.94, "High", ha="right", va="top", fontsize=6.5, color=INK_MUTED)
    for ax, name in zip(axes[1, 1:], models, strict=True):
        score_table(ax, metrics[name])
    fig.subplots_adjust(left=0.035, right=0.995, top=0.91, bottom=0.03)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    for name, row in metrics.items():
        print(name, {key: round(value, 4) for key, value in row.items()})
    build(OUTPUT)
    print("wrote", OUTPUT)
