"""Figure: validation of the sampler on simulated games.

Input:  tables/validation_recovery.csv, tables/validation_recovery_theta.csv,
        tables/validation_repeated.csv, tables/validation_coverage.csv (from validate_simulated.py)
Output: figures/validation.pdf, .png and .tex
"""
from pathlib import Path

import numpy as np
import pandas as pd

from plotting import COLORS, INK, MUTED, TEXT_WIDTH, plt, save
from validate_simulated import N_DATASETS, N_REPEATS

TABLES = Path(__file__).resolve().parent / "tables"
MIN_GAMES = 30
SYMBOLS = {"beta": r"$\beta$", "mu": r"$\mu$", "sigma": r"$\sigma$", "theta": r"$\theta_o$"}


def main():
    recovery = pd.read_csv(TABLES / "validation_recovery.csv").set_index("quantity")
    openings = pd.read_csv(TABLES / "validation_recovery_theta.csv")
    repeated = pd.read_csv(TABLES / "validation_repeated.csv").set_index("quantity")
    coverage = pd.read_csv(TABLES / "validation_coverage.csv").set_index("quantity")

    fig = plt.figure(figsize=(TEXT_WIDTH, 5.0), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 2.1])
    shared_axes = top.subplots(1, 3)
    theta_ax, coverage_ax = bottom.subplots(1, 2)

    for ax, (name, row) in zip(shared_axes, recovery.iterrows(), strict=True):
        ax.hlines(0, row["q05"], row["q95"], color=COLORS[0], linewidth=4, alpha=0.7, label="90% interval")
        ax.plot(row["mean"], 0, "o", color=INK, markersize=4, label="posterior mean")
        ax.axvline(row["truth"], color=COLORS[1], linewidth=1.5, label="true value")
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(SYMBOLS[name])
        span = row["q95"] - row["q05"]
        ax.set_xlim(min(row["q05"], row["truth"]) - 0.4 * span, max(row["q95"], row["truth"]) + 0.4 * span)
    shared_axes[0].set_title("(a) recovery of the shared parameters", loc="left")
    top.legend(*shared_axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=3)

    many = openings["games"] >= MIN_GAMES
    for subset, color, name in [(~many, COLORS[1], f"fewer than {MIN_GAMES} games"),
                                (many, COLORS[0], f"{MIN_GAMES} games or more")]:
        part = openings[subset]
        theta_ax.scatter(part["truth"], part["mean"], s=3 + 0.08 * part["games"], color=color, alpha=0.7,
                         linewidths=0, label=name)
    limits = [openings["truth"].min(), openings["truth"].max()]
    theta_ax.plot(limits, limits, color=INK, linewidth=0.8)
    theta_ax.axhline(recovery.loc["mu", "truth"], color=MUTED, linewidth=0.8, linestyle="--")
    theta_ax.set_aspect("equal")
    theta_ax.set_xlabel(r"true $\theta_o$")
    theta_ax.set_ylabel(r"posterior mean of $\theta_o$")
    theta_ax.set_title(f"(b) recovery of the {len(openings)} opening effects", loc="left")
    legend = theta_ax.legend(loc="upper left")
    for handle in legend.legend_handles:
        handle.set_sizes([20])

    names = list(coverage.index)
    x = np.arange(len(names))
    for offset, level, color in [(-0.12, 50, COLORS[0]), (0.12, 90, COLORS[1])]:
        coverage_ax.axhline(level, color=color, linewidth=0.8, linestyle="--")
        coverage_ax.errorbar(x + offset, 100 * coverage[f"coverage_{level}"], yerr=200 * coverage[f"se_{level}"],
                             fmt="o", markersize=4, color=color, elinewidth=1, capsize=0,
                             label=f"{level}% intervals")
    coverage_ax.set_xticks(x, [SYMBOLS[n] for n in names], fontsize=10)
    coverage_ax.set_ylim(0, 100)
    coverage_ax.set_ylabel("share of intervals containing\nthe true value (%)")
    coverage_ax.set_title("(c) coverage over data sets drawn from the prior", loc="left")
    coverage_ax.legend(loc="lower right")

    beta = recovery.loc["beta"]
    caption = (
        "Validation on simulated games, with the real rating gaps and openings and the results drawn from the "
        "model. (a): one data set simulated with chosen values (orange lines); the posterior mean (black) and "
        "the central 90\\% interval (blue) of $\\beta$, $\\mu$ and $\\sigma$. "
        f"The 90\\% interval of $\\beta$ misses the true value {beta['truth']:g} "
        f"(posterior mean {beta['mean']:.3f}); over {N_REPEATS} "
        f"repeated data sets with the same values, the 90\\% interval of $\\beta$ contains it "
        f"{100 * repeated.loc['beta', 'share_inside_90']:.0f}\\% of the time and the posterior mean is off by "
        f"{repeated.loc['beta', 'mean_error']:.3f} on average (standard error "
        f"{repeated.loc['beta', 'se_mean_error']:.3f}), so the miss is chance and not a bias. "
        "(b): true opening effects of the same data set against their posterior means, with the area of each "
        "point proportional to the number of games and the identity line in black. Openings with few games are "
        "shrunk towards the true $\\mu$ (dashed line), and openings with many games lie near the identity line; "
        f"{100 * openings['inside_90'].mean():.1f}\\% of the 90\\% intervals contain the true value. "
        f"(c): share of the 50\\% and 90\\% posterior intervals that contain the true value, over {N_DATASETS} "
        "data sets "
        "whose parameters are drawn from the prior, with $\\pm 2$ standard errors; for $\\theta_o$ the share is "
        "over all openings of all data sets. The shares match the nominal levels (dashed lines), as they "
        "should when the sampler is correct."
    )
    save(fig, "validation", caption)


if __name__ == "__main__":
    main()
