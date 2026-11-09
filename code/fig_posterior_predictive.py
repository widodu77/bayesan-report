"""Figure: observed test statistics against their posterior predictive distributions.

Input:  tables/posterior_predictive.csv, tables/posterior_predictive_replicates.csv
        (from posterior_predictive.py)
Output: figures/posterior_predictive.pdf, .png and .tex
"""
from pathlib import Path

import numpy as np
import pandas as pd

from plotting import COLORS, INK, MUTED, TEXT_WIDTH, plt, save

TABLES = Path(__file__).resolve().parent / "tables"
GAP_PREFIX = "white win share, rating gap "


def histogram(ax, replicated, observed, p_value, xlabel, title):
    ax.hist(replicated, bins=40, color=COLORS[0], alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axvline(observed, color=INK, linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("replicated data sets")
    ax.set_title(title, loc="left")
    ax.text(0.98, 0.95, f"$p$ = {p_value:.2f}", transform=ax.transAxes, ha="right", va="top", color=INK)


def main():
    table = pd.read_csv(TABLES / "posterior_predictive.csv").set_index("statistic")
    replicates = pd.read_csv(TABLES / "posterior_predictive_replicates.csv")
    overall = "white win share, all games"
    spread = next(s for s in table.index if s.startswith("sd of white win share"))
    min_games = int(spread.split("at least ")[1].split()[0])
    chi2 = "chi-square discrepancy over the bins of rating gap"
    gap = table[table.index.str.startswith(GAP_PREFIX)]

    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 5.0), layout="constrained")
    histogram(axes[0, 0], replicates[overall], table.loc[overall, "observed"], table.loc[overall, "p_value"],
              "White win share", "(a) all games")
    histogram(axes[0, 1], replicates[spread], table.loc[spread, "observed"], table.loc[spread, "p_value"],
              "standard deviation of White win share", f"(b) between openings with $\\geq${min_games} games")

    ax = axes[1, 0]
    x = np.arange(len(gap))
    centre = gap["replicated_mean"].to_numpy()
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.vlines(x, 100 * (gap["replicated_q05"] - centre), 100 * (gap["replicated_q95"] - centre),
              color=COLORS[0], linewidth=4, alpha=0.6, label="replicated, 90% interval")
    ax.plot(x, 100 * (gap["observed"] - centre), "o", color=INK, markersize=4, label="observed")
    labels = [s.removeprefix(GAP_PREFIX) for s in gap.index]
    ax.set_xticks(x, labels, fontsize=6.5, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel("rating gap, White minus Black (points)")
    ax.set_ylabel("White win share minus\nreplicated mean (percentage points)")
    ax.set_title("(c) by rating gap", loc="left")
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.45 * (high - low))
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    observed_chi2 = replicates["chi-square discrepancy, observed results"]
    replicated_chi2 = replicates["chi-square discrepancy, replicated results"]
    ax.scatter(observed_chi2, replicated_chi2, s=3, color=COLORS[0], alpha=0.3, linewidths=0)
    top = max(observed_chi2.max(), replicated_chi2.max())
    ax.plot([0, top], [0, top], color=INK, linewidth=1)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("$X^2$(observed results, parameters)")
    ax.set_ylabel("$X^2$(replicated results, parameters)")
    ax.set_title("(d) discrepancy over the bins of (c)", loc="left")
    ax.text(0.98, 0.95, f"$p$ = {table.loc[chi2, 'p_value']:.3f}", transform=ax.transAxes, ha="right", va="top")

    worst = gap["p_value_two_sided"].idxmin()
    caption = (
        f"Posterior predictive checks, from {len(replicates):,} data sets simulated from the posterior for the "
        "real rating gaps and openings. (a), (b): histograms of the replicated statistic; the black line is "
        "the observed value and $p$ the share of replicated values at least as large. "
        "(c): observed White win share in each decile of rating gap, minus the mean of its replicated values, "
        "with the central 90\\% interval of the replicated values. "
        f"(d): chi-square discrepancy between wins and expected wins over the deciles of (c), for the observed "
        f"and the replicated results under the same posterior draw; points below the line are draws where the observed results fit worse than the replicated ones. "
        f"The model reproduces the overall win share ($p = {table.loc[overall, 'p_value']:.2f}$) and the spread "
        f"between openings ($p = {table.loc[spread, 'p_value']:.2f}$), but not the shape of the win share "
        f"against the rating gap ($p = {table.loc[chi2, 'p_value']:.3f}$): the stronger player wins more often "
        "than predicted at moderate gaps, for example "
        f"{100 * gap.loc[worst, 'observed']:.1f}\\% observed against "
        f"{100 * gap.loc[worst, 'replicated_mean']:.1f}\\% replicated for gaps of "
        f"{worst.removeprefix(GAP_PREFIX)} points."
    )
    save(fig, "posterior_predictive", caption)


if __name__ == "__main__":
    main()
