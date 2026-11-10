"""Figure: ranking of the openings with uncertainty, and the shrinkage of their win shares.

Input:  tables/posterior_openings.csv, tables/posterior_shared.csv (from fit_real.py)
Output: figures/openings.pdf, .png and .tex
"""
from pathlib import Path

import numpy as np
import pandas as pd

from plotting import COLORS, INK, MUTED, TEXT_WIDTH, plt, save

TABLES = Path(__file__).resolve().parent / "tables"
N_SHOWN = 30
MAX_LABEL = 42


def logistic(x):
    return 1 / (1 + np.exp(-x))


def label(row):
    name = row["opening_name"].replace(" |  ", ", ")
    if len(name) > MAX_LABEL:
        name = name[:MAX_LABEL - 1].rstrip(" ,:") + "…"
    return f"{row['opening_eco']}  {name}"


def main():
    openings = pd.read_csv(TABLES / "posterior_openings.csv")
    shared = pd.read_csv(TABLES / "posterior_shared.csv").set_index("quantity")
    average = shared.loc["P(white wins) at equal ratings, average opening", "median"]

    shown = openings.nlargest(N_SHOWN, "games").sort_values("theta_median")
    fig, (ranking, shrinkage) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 5.4), layout="constrained",
                                             gridspec_kw={"width_ratios": [1.15, 1]})

    y = np.arange(len(shown))
    ranking.axvline(100 * average, color=MUTED, linewidth=0.8, linestyle="--")
    ranking.hlines(y, 100 * logistic(shown["theta_q05"]), 100 * logistic(shown["theta_q95"]),
                   color=COLORS[0], linewidth=2.5, alpha=0.7)
    ranking.plot(100 * logistic(shown["theta_median"]), y, "o", color=INK, markersize=3)
    ranking.set_yticks(y, [label(row) for _, row in shown.iterrows()], fontsize=6)
    ranking.set_ylim(-0.7, len(shown) - 0.3)
    ranking.grid(axis="y", visible=False)
    ranking.set_xlabel("P(White wins) at equal ratings (%)")
    ranking.set_title(f"(a) the {N_SHOWN} most played openings", loc="left")

    many = openings["games"] >= 30
    for subset, color, name in [(~many, COLORS[1], "fewer than 30 games"), (many, COLORS[0], "30 games or more")]:
        part = openings[subset]
        shrinkage.scatter(100 * part["white_win_share"], 100 * logistic(part["theta_median"]),
                          s=4 + 0.12 * part["games"], color=color, alpha=0.6, linewidths=0, label=name)
    posterior = 100 * logistic(openings["theta_median"])
    low, high = np.floor(posterior.min() / 5) * 5 - 5, np.ceil(posterior.max() / 5) * 5 + 5
    shrinkage.plot([low, high], [low, high], color=MUTED, linewidth=0.8)
    shrinkage.axhline(100 * average, color=MUTED, linewidth=0.8, linestyle="--")
    shrinkage.set_xlim(-3, 103)
    shrinkage.set_ylim(low, high)
    shrinkage.set_xlabel("observed White win share (%)")
    shrinkage.set_ylabel("posterior median of P(White wins)\nat equal ratings (%)")
    shrinkage.set_title(f"(b) all {len(openings)} openings", loc="left")
    legend = shrinkage.legend(loc="upper left")
    for handle in legend.legend_handles:
        handle.set_sizes([20])

    above = int((logistic(shown["theta_q05"]) > average).sum())
    below = int((logistic(shown["theta_q95"]) < average).sum())
    caption = (
        "Effect of the opening on White's chance of winning. "
        f"(a): posterior median and central 90\\% interval of $\\operatorname{{logit}}^{{-1}}(\\theta_o)$, the "
        f"probability that White wins when both players have the same rating, for the {N_SHOWN} openings with "
        "the most games, ordered by posterior median. The dashed line is the average opening, "
        f"$\\operatorname{{logit}}^{{-1}}(\\mu)$, at {100 * average:.1f}\\%. Each label is an ECO code with the "
        "most frequent opening name among its games; one ECO code groups several variations. "
        f"The 90\\% interval lies entirely above the average for {above} of these {N_SHOWN} openings and "
        f"entirely below it for {below}; the others cannot be told apart from the average opening. "
        "(b): observed White win share of every opening against its posterior median, with the area of each "
        "point proportional to the number of games, the identity line in gray and the dashed line the average "
        "opening. Openings with few games are "
        "pulled almost all the way to the average, and openings with many games stay near their observed share; "
        "the remaining gap from the identity line also reflects the rating gaps of the games, which the model "
        "accounts for and the observed share does not."
    )
    save(fig, "openings", caption.replace("…", "\\ldots{}").replace("#", "\\#"))


if __name__ == "__main__":
    main()
