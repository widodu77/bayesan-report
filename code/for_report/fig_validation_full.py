"""Figure: validation of the final model (sampler_full.py) on simulated games.

Input:  tables/validation_full_recovery.csv, tables/validation_full_coverage.csv (from validate_full.py)
Output: figures/validation_full.pdf, .png and .tex
"""
from pathlib import Path

import numpy as np
import pandas as pd

from core.plotting import COLORS, INK, MUTED, TEXT_WIDTH, plt, save
from validation.validate_full import N_DATASETS

TABLES = Path(__file__).resolve().parents[1] / "tables"
SYMBOLS = {"beta, rated games": r"$\beta_{\mathrm{rated}}$", "beta, unrated games": r"$\beta_{\mathrm{unrated}}$",
           "mu": r"$\mu$", "sigma_move": r"$\sigma_{\mathrm{m}}$", "sigma_reply": r"$\sigma_{\mathrm{r}}$",
           "sigma_player": r"$\sigma_{\mathrm{p}}$", "gamma": r"$\gamma_m$", "theta": r"$\theta_c$", "u": r"$u_p$"}


def main():
    recovery = pd.read_csv(TABLES / "validation_full_recovery.csv").set_index("quantity")
    coverage = pd.read_csv(TABLES / "validation_full_coverage.csv").set_index("quantity")

    fig = plt.figure(figsize=(TEXT_WIDTH, 5.4), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.25, 1])
    recovery_axes = top.subplots(2, 3).ravel()
    coverage_ax = bottom.subplots()

    for ax, (name, row) in zip(recovery_axes, recovery.iterrows(), strict=True):
        ax.hlines(0, row["q05"], row["q95"], color=COLORS[0], linewidth=4, alpha=0.7, label="90% interval")
        ax.plot(row["mean"], 0, "o", color=INK, markersize=4, label="posterior mean")
        ax.axvline(row["truth"], color=COLORS[1], linewidth=1.5, label="chosen value")
        if not np.isnan(row["realized"]):
            ax.plot(row["realized"], 0.45, "v", color=MUTED, markersize=5, label="realized in the data set")
        ax.set_yticks([])
        ax.set_ylim(-0.6, 0.8)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(SYMBOLS[name])
        values = [row["q05"], row["q95"], row["truth"]] + ([] if np.isnan(row["realized"]) else [row["realized"]])
        span = max(values) - min(values)
        ax.set_xlim(min(values) - 0.25 * span, max(values) + 0.25 * span)
    recovery_axes[0].set_title("(a) one simulated data set", loc="left")
    handles, labels = [], []
    for ax in recovery_axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    top.legend(handles, labels, loc="outside lower center", ncols=4, fontsize=7)

    names = list(coverage.index)
    x = np.arange(len(names))
    for offset, level, color in [(-0.12, 50, COLORS[0]), (0.12, 90, COLORS[1])]:
        coverage_ax.axhline(level, color=color, linewidth=0.8, linestyle="--")
        coverage_ax.errorbar(x + offset, 100 * coverage[f"coverage_{level}"], yerr=200 * coverage[f"se_{level}"],
                             fmt="o", markersize=4, color=color, elinewidth=1, capsize=0, label=f"{level}% intervals")
    coverage_ax.set_xticks(x, [SYMBOLS[n] for n in names], fontsize=9)
    coverage_ax.set_ylim(0, 100)
    coverage_ax.set_ylabel("share of intervals containing\nthe true value (%)")
    coverage_ax.set_title(f"(b) coverage over {N_DATASETS} data sets drawn from the prior", loc="left")
    coverage_ax.legend(loc="lower right", ncols=2)

    caption = (
        "Validation of the final model on simulated games, with the real rating gaps, rated status, pairs and "
        "players and the results drawn from the model. (a): one data set simulated with chosen values (orange "
        "lines), with the posterior mean (black) and central 90\\% interval (blue); the gray triangles mark the "
        "values of the spreads and of $\\mu$ realized in this data set, which the posterior targets. "
        "(b): share of the 50\\% and 90\\% intervals that contain the true value, with $\\pm 2$ standard errors, "
        f"over {N_DATASETS} data sets whose parameters are drawn from the prior; for $\\gamma_m$, $\\theta_c$ and "
        "$u_p$ the share is over all first moves, pairs or players of all data sets."
    )
    save(fig, "validation_full", caption)


if __name__ == "__main__":
    main()
