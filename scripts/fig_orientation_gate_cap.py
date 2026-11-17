from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOURCE = Path("outputs") / "culture" / "population_fidelity_orientation_groups.csv"
OUTPUT = Path("figures") / "fidelity" / "orientation" / "fig_fidelity_orientation_gate_cap.pdf"
INK = "#2b2b2a"
INK_MUTED = "#6f6e69"
RULE = "#d9d8d3"
BAND = "#efeeea"
KEPT = "#a9a8a2"
REVERSED = "#eb6834"
LOWERED = "#2a78d6"
REFERENCE = "pfs_original"
REFERENCE_STRUCTURE = "score_structure_original"
VARIANTS = (("pfs_gated", "Orientation gate"), ("pfs", "Orientation cap"))


def population_rows():
    frame = pd.read_csv(SOURCE)
    rows = frame[(frame["group"] == "population") & (frame["replicate"] == 1)].copy()
    rows["standard_error"] = 1 / np.sqrt(rows["n_cells"] - 1)
    rows["reversed"] = (rows["rho_orientation"] <= 0) & (rows[REFERENCE_STRUCTURE] > 0)
    rows["cap_only"] = (rows["rho_orientation"] > 0) & (
        rows[REFERENCE_STRUCTURE] > rows["rho_orientation"]
    )
    return rows


def report(rows):
    scored = rows[rows[REFERENCE] > 0]
    band = 1.96 * rows["standard_error"]
    reversed_rows, cap_rows = rows[rows["reversed"]], rows[rows["cap_only"]]
    reversed_band, cap_band = band[reversed_rows.index], band[cap_rows.index]
    reversed_near = int((reversed_rows["rho_orientation"].abs() < reversed_band).sum())
    reversed_loss = (reversed_rows[REFERENCE] - reversed_rows["pfs_gated"]).mean()
    cap_near = int((cap_rows["rho_orientation"] < cap_band).sum())
    cap_gap = cap_rows[REFERENCE_STRUCTURE] - cap_rows["rho_orientation"]
    loss = cap_rows[REFERENCE] - cap_rows["pfs"]
    relative = loss / cap_rows[REFERENCE]
    print(f"combinations {len(rows)}, original PFS above zero {len(scored)}")
    print(
        f"reversed {len(reversed_rows)}, within 1.96 SE of zero {reversed_near}, "
        f"mean PFS lost {reversed_loss:.3f}"
    )
    print(
        f"cap only {len(cap_rows)}, orientation within 1.96 SE of zero {cap_near}, "
        f"gap below 1.96 SE {int((cap_gap < cap_band).sum())}"
    )
    print(
        f"cap only PFS lost mean {loss.mean():.3f} max {loss.max():.3f} "
        f"relative mean {relative.mean():.3f} max {relative.max():.3f}"
    )
    print(
        f"standard error {rows['standard_error'].min():.4f} to {rows['standard_error'].max():.4f}"
    )


def draw(ax, rows, column, title, band):
    scored = rows[rows[REFERENCE] > 0]
    kept_share = scored[column] / scored[REFERENCE]
    ax.axvspan(-band, band, color=BAND, linewidth=0, zorder=0)
    ax.axhline(1, color=RULE, linewidth=0.6, zorder=1)
    ax.axvline(0, color=INK_MUTED, linewidth=0.5, zorder=1)
    layers = (
        (~scored["reversed"] & ~scored["cap_only"], KEPT, "o", 9, "Unchanged by both"),
        (scored["reversed"], REVERSED, "s", 13, "Reversed ordering"),
        (scored["cap_only"], LOWERED, "^", 15, "Lowered by the cap only"),
    )
    for mask, color, marker, size, label in layers:
        ax.scatter(
            scored.loc[mask, "rho_orientation"],
            kept_share[mask],
            s=size,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.4,
            label=label,
            zorder=3,
            clip_on=False,
        )
    ax.set_title(title, fontsize=7.5, fontweight="bold", color=INK, pad=4)
    ax.set_xlim(-0.4, 0.8)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(labelsize=6.5, colors=INK_MUTED, length=2, width=0.5)
    ax.set_xlabel(
        r"Orientation $\rho_{\mathrm{orient}}$", fontsize=6.5, color=INK_MUTED, labelpad=2
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.5)


def build(rows, path):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Nimbus Sans", "Liberation Sans", "Arial"],
            "pdf.fonttype": 42,
        }
    )
    band = float(1.96 * rows["standard_error"].mean())
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.35), sharey=True)
    for ax, (column, title) in zip(axes, VARIANTS, strict=True):
        draw(ax, rows, column, title, band)
    axes[0].set_ylabel("Share of original PFS kept", fontsize=6.5, color=INK_MUTED, labelpad=3)
    axes[0].text(
        band + 0.02,
        0.5,
        "Zero or full credit,\ndecided by the sign",
        fontsize=6.5,
        color=INK,
        va="center",
        ha="left",
    )
    axes[1].text(
        band + 0.02,
        0.5,
        "Credit falls as\n" + r"$\rho_{\mathrm{orient}}$ nears 0",
        fontsize=6.5,
        color=INK,
        va="center",
        ha="left",
    )
    for ax in axes:
        ax.text(
            -band - 0.02,
            0.5,
            "Within 1.96 SE\nof zero",
            fontsize=6.5,
            color=INK_MUTED,
            va="center",
            ha="right",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=6.5,
        handletextpad=0.2,
        columnspacing=1.4,
        labelcolor=INK,
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.91, bottom=0.27, wspace=0.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    population = population_rows()
    report(population)
    build(population, OUTPUT)
    print("wrote", OUTPUT)
