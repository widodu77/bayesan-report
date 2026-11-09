"""Data sensitivity: refit the model on other versions of the cleaned games and compare.

Draws scored as half a win: the likelihood term y * eta - log(1 + exp(eta)) with y = 1/2 is
log sqrt(p (1 - p)), so a draw counts as half a White win and half a Black win. This is not the
probability of a draw under any model of three outcomes, but it is the usual way to use the score
of a game, as in the Elo system.

Output: tables/sensitivity_data.csv
"""
from pathlib import Path

from sampler import Priors, load_games
from sensitivity import compare

TABLES = Path(__file__).resolve().parent / "tables"
VARIANTS = [
    ("baseline", "decisive games with at least 10 plies", {}),
    ("draws as half wins", "draws kept with y = 1/2", {"draws_as_half": True}),
    ("rated games only", "unrated games removed", {"rated_only": True}),
    ("no player rated 1500", "games with a player at the possible default rating removed", {"drop_rating": 1500}),
    ("at least 20 plies", "games shorter than 10 moves per side removed", {"min_plies": 20}),
]


def main():
    variants = []
    for name, description, options in VARIANTS:
        data, codes = load_games(**options)
        variants.append((name, description, data, codes, Priors()))
    table = compare(variants)
    TABLES.mkdir(exist_ok=True)
    table.to_csv(TABLES / "sensitivity_data.csv", index=False)
    columns = ["variant", "games", "beta_mean", "mu_mean", "sigma_mean", "advantage_mean", "advantage_q05",
               "advantage_q95", "rank_correlation_with_baseline", "top_10_shared_with_baseline", "largest_rhat",
               "smallest_ess"]
    print(table[columns].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
