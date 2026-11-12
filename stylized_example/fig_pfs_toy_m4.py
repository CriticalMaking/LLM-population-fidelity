from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm, spearmanr

GROUPS = ("A", "B", "C")
GROUP_COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
POOLED_FILL = "#dcdbd6"
INK = "#2b2b2a"
INK_MUTED = "#6f6e69"

K = 101
X = np.linspace(0.0, 10.0, K)

LW = 72.27 / 25.4 * 0.75


def nemd(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.abs(np.cumsum(p - q))[:-1].sum() / (len(p) - 1))


def make_dist(mu: float, sd: float) -> np.ndarray:
    d = norm.pdf(X, mu, sd)
    return d / d.sum()


def mean_of(p: np.ndarray) -> float:
    return float((X * p).sum())


def sd_of(p: np.ndarray) -> float:
    return float(np.sqrt((((X - mean_of(p)) ** 2) * p).sum()))


def sharpen_groups(p: np.ndarray, tau: float) -> tuple[float, list[np.ndarray]]:
    mid = mean_of(p)

    def weights(offset: float) -> np.ndarray:
        centres = mid + np.array([-offset, 0.0, offset])
        w = np.exp(-((X[:, None] - centres[None, :]) ** 2) / (2 * tau**2))
        return w / w.sum(axis=1, keepdims=True)

    offset = brentq(lambda o: (p * weights(o)[:, 1]).sum() - 1 / 3, 0.01, 5, xtol=1e-15)
    w = weights(offset)
    return offset, [3 * p * w[:, g] for g in range(3)]


def pair_dists(dists: list[np.ndarray]) -> np.ndarray:
    return np.round([nemd(a, b) for a, b in combinations(dists, 2)], 10)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def pfs_metrics(s: list[np.ndarray], m: list[np.ndarray]) -> dict[str, float]:
    errors = [nemd(a, b) for a, b in zip(s, m, strict=True)]
    d_s, d_m = pair_dists(s), pair_dists(m)
    ratio = float(np.median(d_m) / np.median(d_s))
    rho = spearman(d_s, d_m)
    s_acc = 1 - float(np.mean(errors))
    s_adapt = 0.0 if ratio == 0 else min(ratio, 1 / ratio)
    s_struct = 0.0 if np.isnan(rho) else max(0.0, rho)
    rho_orient = spearman(
        np.round([mean_of(p) for p in s], 10), np.round([mean_of(p) for p in m], 10)
    )
    credit = 0.0 if np.isnan(rho_orient) else max(0.0, rho_orient)
    s_capped = min(s_struct, credit)
    return {
        "errors": errors,
        "d_m": d_m,
        "S_acc": s_acc,
        "A": ratio,
        "S_adapt": s_adapt,
        "rho": rho,
        "S_struct": s_struct,
        "S_center": 1 - nemd(np.mean(s, axis=0), np.mean(m, axis=0)),
        "PFS": (s_acc * s_adapt * s_struct) ** (1 / 3),
        "rho_orient": rho_orient,
        "S_struct_capped": s_capped,
        "PFS_capped": (s_acc * s_adapt * s_capped) ** (1 / 3),
    }


survey = [make_dist(mu, 1.3) for mu in (4, 5, 6)]
pooled = np.mean(survey, axis=0)
offset, exaggerated = sharpen_groups(pooled, tau=0.4)

models = {
    "M1: collapsed": [pooled, pooled, pooled],
    "M2: exaggerated": exaggerated,
    "M3: swapped": [survey[0], survey[2], survey[1]],
    "M4: reversed": [survey[2], survey[1], survey[0]],
}

metrics = {name: pfs_metrics(survey, m) for name, m in models.items()}

d_survey = pair_dists(survey)
assert d_survey[0] == d_survey[2], "survey: AB = BC"
assert np.array_equal(metrics["M4: reversed"]["d_m"], d_survey), "M4 keeps every distance"
assert all(abs(row["S_center"] - 1) < 1e-12 for row in metrics.values())


def report() -> None:
    print(f"M2 sorting: tau = 0.4, c = {offset:.4f}")
    print("survey pairwise (AB, AC, BC):", np.round(d_survey, 4))
    for name, row in metrics.items():
        means = [round(mean_of(p), 2) for p in models[name]]
        print(
            f"{name:16s} means={means} errors={np.round(row['errors'], 3)} "
            f"d={np.round(row['d_m'], 3)} S_acc={row['S_acc']:.4f} A={row['A']:.3f} "
            f"S_adapt={row['S_adapt']:.3f} rho={row['rho']:.3f} S_struct={row['S_struct']:.2f} "
            f"PFS={row['PFS']:.4f} rho_orient={row['rho_orient']:.2f} "
            f"S_struct_capped={row['S_struct_capped']:.2f} PFS_capped={row['PFS_capped']:.4f}"
        )


Y_TOP = pooled.max() * 1.2
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Nimbus Sans", "Liberation Sans", "Arial"],
        "pdf.fonttype": 42,
    }
)


def plot_panel(ax, dists, title, collapsed=False, y_label=None) -> None:
    ax.fill_between(X, pooled, color=POOLED_FILL, linewidth=0)
    curves = {g: p / 3 for g, p in zip(GROUPS, dists, strict=True)}
    if collapsed:
        dash = 0.35
        seg = np.floor(X / dash + 1e-9).astype(int)
        for i, g in enumerate(GROUPS):
            for s in np.unique(seg[seg % 3 == i]):
                keep = seg == s
                ax.plot(
                    X[keep],
                    curves[g][keep],
                    color=GROUP_COLORS[g],
                    linewidth=0.55 * LW,
                    solid_capstyle="butt",
                )
        mean, height = mean_of(dists[0]), np.interp(mean_of(dists[0]), X, curves["A"])
        step = height / 12
        for i in range(12):
            ax.plot(
                [mean, mean],
                [i * step, (i + 0.5) * step],
                color=GROUP_COLORS[GROUPS[i % 3]],
                linewidth=0.35 * LW,
                alpha=0.7,
                solid_capstyle="butt",
            )
        ax.text(
            5,
            curves["A"].max() + 0.035 * Y_TOP,
            "A = B = C",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            color=INK,
        )
    else:
        for g in GROUPS:
            y = curves[g]
            ax.fill_between(X, y, color=GROUP_COLORS[g], alpha=0.16, linewidth=0)
            ax.plot(X, y, color=GROUP_COLORS[g], linewidth=0.45 * LW)
            mean = mean_of(y * 3)
            ax.plot(
                [mean, mean],
                [0, np.interp(mean, X, y)],
                color=GROUP_COLORS[g],
                linewidth=0.35 * LW,
                alpha=0.7,
                dashes=(2, 2),
                dash_capstyle="butt",
            )
            ax.text(
                mean,
                y[np.abs(X - mean) <= 0.5].max() + 0.035 * Y_TOP,
                g,
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                color=INK,
            )
    ax.plot([0, 10], [0, 0], color=INK_MUTED, linewidth=0.3 * LW, clip_on=False)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, Y_TOP)
    ax.set_title(title, fontsize=7.5, fontweight="bold", color=INK, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    if y_label:
        ax.plot([0, 0], [0, Y_TOP], color=INK_MUTED, linewidth=0.3 * LW, clip_on=False)
        ax.set_ylabel(y_label, fontsize=6.5, color=INK_MUTED, labelpad=2)


def fmt(value: float) -> str:
    if np.isnan(value):
        return "–"
    return f"{value:.2f}".replace("-", "−")


ROWS = (
    ("Accuracy", "S_acc", "normal"),
    ("Adaptability", "S_adapt", "normal"),
    ("Structure", "S_struct_capped", "normal"),
    ("PFS", "PFS_capped", "bold"),
)


def score_table(ax, scores) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.45, len(ROWS) + 0.55)
    for i, (label, key, weight) in enumerate(ROWS):
        y = len(ROWS) - i
        ax.text(0.10, y, label, ha="left", va="center", fontsize=6.5, fontweight=weight, color=INK)
        ax.text(
            0.90,
            y,
            fmt(scores[key]),
            ha="right",
            va="center",
            fontsize=6.5,
            fontweight=weight,
            color=INK,
        )


def build(path: Path) -> None:
    fig, axes = plt.subplots(
        2,
        5,
        figsize=(5.5, 1.6),
        gridspec_kw={"height_ratios": [1, 0.44], "wspace": 0.14, "hspace": 0.06},
    )
    plot_panel(axes[0, 0], survey, "Survey (reference)", y_label="Population share")
    for ax, (name, dists) in zip(axes[0, 1:], models.items(), strict=True):
        plot_panel(ax, dists, name, collapsed="collapsed" in name)

    ends = axes[1, 0]
    ends.axis("off")
    ends.set_xlim(0, 10)
    ends.set_ylim(0, 1)
    ends.text(0.05, 0.93, "Low", ha="left", va="top", fontsize=6.5, color=INK_MUTED)
    ends.text(9.95, 0.93, "High", ha="right", va="top", fontsize=6.5, color=INK_MUTED)
    for ax, name in zip(axes[1, 1:], models, strict=True):
        score_table(ax, metrics[name])

    fig.subplots_adjust(left=0.035, right=0.995, top=0.90, bottom=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    report()
    out = Path("figures") / "stylized_example" / "fig_pfs_toy_m4.pdf"
    build(out)
    print("wrote", out)
