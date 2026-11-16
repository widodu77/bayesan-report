"""Figure: posterior predictive checks of the final model, against the same model without player effects.

Input:  tables/posterior_predictive_full.csv, tables/posterior_predictive_full_replicates.csv
        (from posterior_predictive_full.py)
Output: figures/posterior_predictive_full.pdf, .png and .tex
"""
from pathlib import Path

import numpy as np
import pandas as pd

from plotting import COLORS, INK, MUTED, TEXT_WIDTH, plt, save

TABLES = Path(__file__).resolve().parent / "tables"
MODELS = [("without player effects", "without player effects", COLORS[1], "o"),
          ("with player effects", "with player effects (final)", COLORS[0], "s")]
SHORT_NAMES = {  # statistic in the table: label in panel (b)
    "white win share, all games": "White win share",
    "sd of white win share between pairs": "spread between pairs",
    "chi-square over deciles of rating gap, all games": "rating-gap shape, all",
    "chi-square over deciles of rating gap, rated games": "rating-gap shape, rated",
    "chi-square over deciles of rating gap, unrated games": "rating-gap shape, unrated",
    "player dispersion": "player dispersion",
}


def short_name(statistic):
    return next(label for prefix, label in SHORT_NAMES.items() if statistic.startswith(prefix))


def main():
    table = pd.read_csv(TABLES / "posterior_predictive_full.csv")
    replicates = pd.read_csv(TABLES / "posterior_predictive_full_replicates.csv")
    dispersion_name = next(s for s in table["statistic"] if s.startswith("player dispersion"))

    fig, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.0), layout="constrained",
                                      gridspec_kw={"width_ratios": [1, 1.25]})

    for row, (model, label, color, _) in enumerate(MODELS):
        observed, replicated = replicates[f"{model}, observed"], replicates[f"{model}, replicated"]
        y = -row
        left.hlines(y, *np.percentile(replicated, [5, 95]), color=color, linewidth=4, alpha=0.5)
        left.plot(np.median(replicated), y, "|", color=color, markersize=10, markeredgewidth=1.5)
        left.plot(observed.mean(), y, "o", color=INK, markersize=5)
        p = table.query("model == @model and statistic == @dispersion_name")["p_value"].iloc[0]
        left.text(observed.mean() + 0.15, y, f"$p$ = {p:.3f}" if p >= 0.001 else "$p$ < 0.001", ha="left",
                  va="center", fontsize=7, color=INK)
    left.axvline(1, color=MUTED, linewidth=0.6, linestyle="--")
    left.set_xlim(0.7, 2.9)
    left.set_yticks([0, -1], [label.replace(" (final)", "\n(final model)") for _, label, _, _ in MODELS])
    left.set_ylim(-1.6, 0.7)
    left.grid(axis="y", visible=False)
    left.set_xlabel("variance of the standardized player residual")
    left.set_title("(a) player dispersion", loc="left")

    statistics = list(dict.fromkeys(table["statistic"]))
    y_positions = -np.arange(len(statistics))
    right.axvspan(0, 0.05, color=MUTED, alpha=0.12, linewidth=0)
    right.axvspan(0.95, 1, color=MUTED, alpha=0.12, linewidth=0)
    for offset, (model, label, color, marker) in zip((0.15, -0.15), MODELS, strict=True):
        p = table[table["model"] == model].set_index("statistic").loc[statistics, "p_value"]
        right.scatter(p, y_positions + offset, color=color, marker=marker, s=22, label=label, zorder=3)
    right.set_yticks(y_positions, [short_name(s) for s in statistics])
    right.set_xlim(-0.02, 1.02)
    right.grid(axis="y", visible=False)
    right.set_xlabel("posterior predictive $p$-value")
    right.set_title("(b) all checks", loc="left")
    right.legend(loc="lower center", bbox_to_anchor=(0.5, 1.07), ncols=2, fontsize=7, handletextpad=0.3,
                 columnspacing=1.0)

    without = table[table["model"] == "without player effects"].set_index("statistic")
    final = table[table["model"] == "with player effects"].set_index("statistic")
    caption = (
        "Posterior predictive checks of the final model (blue) and of the same model without player effects "
        f"(orange), each from {len(replicates):,} data sets simulated from the posterior for the real games. "
        "(a): for each player with at least 10 games, the difference between their score and the score the "
        "model expects, divided by its standard deviation; the statistic is the variance of these "
        "standardized residuals over the players, close to 1 when a player's games are independent given the "
        "model. Black dots are the observed values averaged over posterior draws, bars the 90\\% interval and "
        "ticks the median of the replicated values. Without player effects the observed dispersion is "
        f"{without.loc[dispersion_name, 'observed']:.2f}, far above what the model can produce "
        f"({without.loc[dispersion_name, 'replicated_q05']:.2f} to "
        f"{without.loc[dispersion_name, 'replicated_q95']:.2f}): the same players keep beating, or missing, their "
        f"rating. With player effects it is {final.loc[dispersion_name, 'observed']:.2f} against "
        f"{final.loc[dispersion_name, 'replicated_q05']:.2f} to {final.loc[dispersion_name, 'replicated_q95']:.2f}. "
        "(b): posterior predictive $p$-values of all the statistics, the share of replicated values at least as "
        "large as the observed one; values in the gray bands, below 0.05 or above 0.95, flag a feature of the "
        "data the model does not reproduce."
    )
    save(fig, "posterior_predictive_full", caption)


if __name__ == "__main__":
    main()
