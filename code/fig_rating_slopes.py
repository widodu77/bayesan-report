"""Figure: separate rating slopes for rated and unrated games, and the posterior predictive checks
of the models with one slope and with two.

Input:  data/processed/games_clean.csv, results/draws_rating_slopes.npz,
        tables/posterior_rating_slopes.csv, tables/favourite_win_probability.csv,
        tables/posterior_predictive_rating_slopes.csv (from fit_rating_slopes.py)
Output: figures/rating_slopes.pdf, .png and .tex
"""
from pathlib import Path

import numpy as np
import pandas as pd

from plotting import COLORS, INK, MUTED, TEXT_WIDTH, plt, save
from sampler import load_games

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"
RESULTS = ROOT / "results"
GAP_EDGES = np.arange(-600, 601, 50)  # rating points; wider gaps are pooled in the end bins
MODELS = [("one slope", "one slope", "o", "none", MUTED), ("rated and unrated slopes", "two slopes", "o", INK, INK)]
SUBSETS = ["all games", "rated games", "unrated games"]
P_COLUMN = 100  # x position of the right-aligned p-values in panel (b), past the largest discrepancy


def logistic(x):
    return 1 / (1 + np.exp(-x))


def binned_share(gap, y):
    clipped = np.clip(gap, GAP_EDGES[0], GAP_EDGES[-1] - 1e-9)
    bins = np.digitize(clipped, GAP_EDGES) - 1
    n = np.bincount(bins, minlength=len(GAP_EDGES) - 1)
    share = np.bincount(bins, y, len(GAP_EDGES) - 1) / np.maximum(n, 1)
    half_width = 1.645 * np.sqrt(share * (1 - share) / np.maximum(n, 1))
    centre = (GAP_EDGES[:-1] + GAP_EDGES[1:]) / 2
    return centre[n > 0], share[n > 0], half_width[n > 0]


def main():
    data, _ = load_games(slope_by="rated")
    saved = np.load(RESULTS / "draws_rating_slopes.npz", allow_pickle=True)
    beta, mu = saved["beta"].reshape(-1, data.n_slopes), saved["mu"].ravel()
    posterior = pd.read_csv(TABLES / "posterior_rating_slopes.csv").set_index("quantity")
    favourite = pd.read_csv(TABLES / "favourite_win_probability.csv").set_index("rating_gap")
    checks = pd.read_csv(TABLES / "posterior_predictive_rating_slopes.csv")

    fig, (curve_ax, check_ax) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.0), layout="constrained",
                                             gridspec_kw={"width_ratios": [1.15, 1]})

    gap = 400 * data.delta
    grid = np.linspace(GAP_EDGES[0], GAP_EDGES[-1], 241)
    curve_ax.plot(grid, 100 / (1 + 10**(-grid / 400)), color=MUTED, linewidth=1, linestyle="--", label="Elo formula")
    for k, name in enumerate(data.slope_names):
        games = data.slope_index == k
        centre, share, half_width = binned_share(gap[games], data.y[games])
        curve_ax.errorbar(centre, 100 * share, yerr=100 * half_width, fmt="o", markersize=2.5, color=COLORS[k],
                          elinewidth=0.6, capsize=0, alpha=0.8)
        # Posterior of P(White wins) in the average opening; drawn in chunks to keep memory small.
        probability = np.concatenate([logistic(mu[i:i + 4000, None] + beta[i:i + 4000, k, None] * grid / 400)
                                      for i in range(0, len(mu), 4000)])
        low, mean, high = np.percentile(probability, 5, axis=0), probability.mean(axis=0), \
            np.percentile(probability, 95, axis=0)
        curve_ax.fill_between(grid, 100 * low, 100 * high, color=COLORS[k], alpha=0.25, linewidth=0)
        slope = posterior.loc[f"beta, {name}", "mean"]
        curve_ax.plot(grid, 100 * mean, color=COLORS[k], linewidth=1.5, label=f"{name.split()[0]}, slope {slope:.2f}")
    curve_ax.axhline(50, color=MUTED, linewidth=0.6)
    curve_ax.axvline(0, color=MUTED, linewidth=0.6)
    curve_ax.set_xlabel("rating gap, White minus Black (points)")
    curve_ax.set_ylabel("White win share (%)")
    curve_ax.set_title("(a) result against rating gap", loc="left")
    curve_ax.legend(loc="upper left")

    y_positions = {}
    for row, subset in enumerate(SUBSETS):
        for offset, (model, label, marker, face, color) in zip((0.17, -0.17), MODELS, strict=True):
            line = checks[(checks["model"] == model)
                          & (checks["statistic"] == f"chi-square over deciles of rating gap, {subset}")].iloc[0]
            y = -row + offset
            y_positions[(subset, model)] = y
            check_ax.hlines(y, line["replicated_q05"], line["replicated_q95"], color=color, linewidth=3, alpha=0.35)
            check_ax.plot(line["observed"], y, marker, markersize=5, markerfacecolor=face, markeredgecolor=color,
                          markeredgewidth=1.2)
            if row == 0:  # direct labels on the first row instead of a legend
                check_ax.text(line["replicated_q95"] + 7, y, label, va="center", ha="left", fontsize=7, color=color)
            check_ax.text(P_COLUMN, y, f"$p$ = {line['p_value']:.3f}" if line["p_value"] >= 0.001 else "$p$ < 0.001",
                          va="center", ha="right", fontsize=7, color=INK)
    check_ax.set_yticks([-row for row in range(len(SUBSETS))], SUBSETS)
    check_ax.grid(axis="y", visible=False)
    check_ax.set_xlim(0, P_COLUMN + 1)
    check_ax.set_xticks([0, 20, 40, 60])
    check_ax.set_xlabel("chi-square discrepancy")
    check_ax.set_title("(b) posterior predictive check", loc="left")

    difference = posterior.loc["beta, rated minus unrated"]
    p_one = checks.query("model == 'one slope' and statistic == 'chi-square over deciles of rating gap, all games'")
    p_two = checks.query("model == 'rated and unrated slopes' and "
                         "statistic == 'chi-square over deciles of rating gap, all games'")
    rated_200 = favourite.loc[200]
    caption = (
        "A separate rating slope for rated and unrated games. "
        f"(a): share of games won by White in bins of 50 rating points, with 90\\% binomial intervals, for the "
        f"{int(np.sum(data.slope_index == 0)):,} rated and {int(np.sum(data.slope_index == 1)):,} unrated decisive "
        "games; the lines and bands are the posterior mean and 90\\% interval of "
        "$\\operatorname{logit}^{-1}(\\mu + \\beta_g \\Delta)$ for the average opening, and the dashed line is "
        "the expected score of the Elo formula. The rating gap predicts the result much better in rated games: "
        f"the slopes differ by {difference['mean']:.2f} (90\\% interval {difference['q05']:.2f} to "
        f"{difference['q95']:.2f}). Both curves are flatter than the Elo formula: in rated games a player rated "
        f"200 points higher wins {100 * rated_200['rated games, mean']:.1f}\\% of decisive games "
        f"({100 * rated_200['rated games, q05']:.1f} to {100 * rated_200['rated games, q95']:.1f}\\%), not the "
        f"{100 * rated_200['elo_expected_score']:.0f}\\% of the Elo formula. "
        "(b): chi-square discrepancy between wins and expected wins over the deciles of rating gap, on all games "
        "and within each group, for the model with one slope (open circles) and with two (filled); the dot is "
        "the realized discrepancy averaged over posterior draws and the bar the 90\\% interval of its replicated "
        "values, with the posterior predictive $p$-value. With one slope the shape of the curve is rejected "
        f"($p = {p_one['p_value'].iloc[0]:.3f}$ on all games, and lower within each group); with two slopes the "
        f"misfit disappears ($p = {p_two['p_value'].iloc[0]:.2f}$)."
    )
    save(fig, "rating_slopes", caption)


if __name__ == "__main__":
    main()
