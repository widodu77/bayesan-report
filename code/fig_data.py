"""Figure: descriptive view of the decisive games.

Input:  data/processed/games_clean.csv (from prepare_data.py)
Output: figures/data.pdf, .png and .tex
"""
import numpy as np

from plotting import COLORS, INK, MUTED, TEXT_WIDTH, plt, save
from sampler import load_games

GAP_EDGES = np.arange(-600, 601, 50)  # rating points; wider gaps are rare and pooled in the end bins


def main():
    data, _ = load_games()
    games = np.sort(data.games_per_opening)[::-1]
    gap = 400 * data.delta

    fig, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.8), layout="constrained")

    rank = np.arange(1, len(games) + 1)
    left.bar(rank, games, width=1.0, color=COLORS[0], linewidth=0)
    left.set_yscale("log")
    left.set_xlim(0, len(games) + 1)
    left.axhline(np.median(games), color=INK, linewidth=0.8, linestyle="--")
    left.text(len(games), np.median(games) * 1.25, f"median {np.median(games):.0f} games", ha="right", va="bottom",
              fontsize=7, color=INK)
    left.set_xlabel("opening (ECO code), from most to least played")
    left.set_ylabel("decisive games")
    left.set_title(f"(a) games per opening, {len(games)} openings", loc="left")

    clipped = np.clip(gap, GAP_EDGES[0], GAP_EDGES[-1] - 1e-9)
    bins = np.digitize(clipped, GAP_EDGES) - 1
    n = np.bincount(bins, minlength=len(GAP_EDGES) - 1)
    share = np.bincount(bins, data.y, len(GAP_EDGES) - 1) / np.maximum(n, 1)
    centre = (GAP_EDGES[:-1] + GAP_EDGES[1:]) / 2
    half_width = 1.645 * np.sqrt(share * (1 - share) / np.maximum(n, 1))
    curve = np.linspace(GAP_EDGES[0], GAP_EDGES[-1], 200)
    right.plot(curve, 100 / (1 + 10**(-curve / 400)), color=MUTED, linewidth=1, linestyle="--",
               label="Elo formula")
    right.errorbar(centre[n > 0], 100 * share[n > 0], yerr=100 * half_width[n > 0], fmt="o", markersize=3,
                   color=COLORS[0], ecolor=COLORS[0], elinewidth=1, capsize=0, label="observed, 90% interval")
    right.axhline(50, color=MUTED, linewidth=0.6)
    right.axvline(0, color=MUTED, linewidth=0.6)
    right.set_xlabel("rating gap, White minus Black (points)")
    right.set_ylabel("White win share (%)")
    right.set_title("(b) result against rating gap", loc="left")
    right.legend(loc="upper left")

    singles = int(np.sum(games == 1))
    level = np.abs(gap) < 25
    caption = (
        f"The {len(data.y):,} decisive games. (a): number of games for each of the {len(games)} openings, "
        f"identified by their ECO code, on a log scale. The data are very unbalanced: the most played opening "
        f"has {games[0]} games, the median {np.median(games):.0f}, and {singles} openings have a single game, "
        "which motivates a hierarchical model that pools information between openings. "
        "(b): share of games won by White in bins of 50 rating points, with 90\\% binomial intervals; gaps "
        f"beyond $\\pm${GAP_EDGES[-1]} points are pooled in the end bins. The dashed line is the expected score "
        "of the Elo formula, $1/(1 + 10^{-\\text{gap}/400})$. The rating gap is the main driver of the result, "
        "but the observed curve is flatter than the Elo formula, so the slope $\\beta$ is left free in the "
        "model. White wins more than half of the games between players of nearly equal rating "
        f"({100 * data.y[level].mean():.1f}\\% of the {level.sum():,} games with a gap under 25 points), "
        "which the model attributes to the first move and to the opening."
    )
    save(fig, "data", caption)


if __name__ == "__main__":
    main()
