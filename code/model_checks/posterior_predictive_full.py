"""Posterior predictive checks of the final model, against the same model without player effects.

Both models are the model of White's first move and Black's reply with one rating slope for rated and
one for unrated games (fit_moves.py); the final model adds one effect per player (fit_full.py).
For each of N_REPLICATES posterior draws, results are simulated for the real games, and discrepancies
that depend on the parameters (Gelman et al., Bayesian Data Analysis, Section 6.3) are computed on the
observed and on the replicated results:

- the White win share, and the spread of win shares between pairs with at least MIN_GAMES games;
- the chi-square over deciles of rating gap, on all games and within rated and unrated games;
- the player dispersion: for each player with at least MIN_PLAYER_GAMES games, the standardized
  difference between the score and the expected score, z = (score - E) / sqrt(V); the statistic is
  the variance of z over these players, close to 1 when the games of a player are independent given
  the model, and above 1 when players keep beating, or missing, what the model expects of them.

Input:  results/draws_moves.npz (from fit_moves.py), results/draws_full.npz (from fit_full.py)
Output: tables/posterior_predictive_full.csv, tables/posterior_predictive_full_replicates.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from core.sampler import load_games
from model_checks.posterior_predictive import N_REPLICATES, gap_bins, gap_discrepancy

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
RESULTS = ROOT / "results"
SEED = 20261120
MIN_GAMES = 30
MIN_PLAYER_GAMES = 10


def logistic(x):
    return 1 / (1 + np.exp(-x))


def model_draws(data, codes):
    """Aligned draws (beta, theta, u) of both models, N_REPLICATES each; u is zero without player effects."""
    moves = np.load(RESULTS / "draws_moves.npz", allow_pickle=True)
    full = np.load(RESULTS / "draws_full.npz", allow_pickle=True)
    for saved in (moves, full):
        if not np.array_equal(saved["codes"], codes):
            raise ValueError("saved draws do not match the data; rerun fit_moves.py and fit_full.py")

    # Without player effects: theta was kept every 10 iterations, so beta is thinned to match.
    theta = moves["theta"].reshape(-1, data.n_openings).astype(float)
    thin = moves["beta"].shape[1] // moves["theta"].shape[1]
    beta = moves["beta"][:, ::thin].reshape(-1, data.n_slopes)
    pick = np.sort(np.random.default_rng(SEED).choice(len(theta), size=N_REPLICATES, replace=False))
    without = (beta[pick], theta[pick], None)

    # Final model: the player effects were kept every u_every iterations; theta and beta are aligned to them.
    u_every, theta_thin = int(full["u_every"]), int(full["theta_thin"])
    u = full["u"].reshape(-1, data.n_players)
    theta = full["theta"][:, ::u_every // theta_thin].reshape(-1, data.n_openings).astype(float)
    beta = full["beta"][:, ::u_every].reshape(-1, data.n_slopes)
    if not len(theta) == len(beta) == len(u):
        raise ValueError("the kept draws of the final model are not aligned")
    pick = np.sort(np.random.default_rng(SEED + 1).choice(len(u), size=min(N_REPLICATES, len(u)), replace=False))
    return {"without player effects": without, "with player effects": (beta[pick], theta[pick], u[pick])}


def player_dispersion(y, p, data, active):
    """Variance over the active players of the standardized difference between score and expected score."""
    P = data.n_players
    score = np.bincount(data.white_player, y, P) + np.bincount(data.black_player, 1 - y, P)
    expected = np.bincount(data.white_player, p, P) + np.bincount(data.black_player, 1 - p, P)
    variance = np.bincount(data.white_player, p * (1 - p), P) + np.bincount(data.black_player, p * (1 - p), P)
    z = (score[active] - expected[active]) / np.sqrt(variance[active])
    return z.var(ddof=1)


def main():
    data, codes = load_games(openings="moves", slope_by="rated")
    games = np.bincount(data.white_player, minlength=data.n_players) + np.bincount(data.black_player,
                                                                                   minlength=data.n_players)
    active = games >= MIN_PLAYER_GAMES
    well_observed = data.games_per_opening >= MIN_GAMES
    subsets = {"all games": np.ones(len(data.y), dtype=bool)}
    subsets.update({name: data.slope_index == k for k, name in enumerate(data.slope_names)})
    bins = {name: gap_bins(data.delta[mask])[0] for name, mask in subsets.items()}

    def spread(y):
        wins = np.bincount(data.opening, y, data.n_openings)
        return np.std(wins[well_observed] / data.games_per_opening[well_observed], ddof=1)

    names = (["white win share, all games", f"sd of white win share between pairs with at least {MIN_GAMES} games"]
             + [f"chi-square over deciles of rating gap, {name}" for name in subsets]
             + [f"player dispersion, players with at least {MIN_PLAYER_GAMES} games"])
    rows, replicates = [], {}
    rng = np.random.default_rng(SEED + 2)
    for model, (beta, theta, u) in model_draws(data, codes).items():
        observed, replicated = [], []
        for s in range(len(beta)):
            eta = theta[s][data.opening] + beta[s][data.slope_index] * data.delta
            if u is not None:
                eta = eta + u[s][data.white_player] - u[s][data.black_player]
            p = logistic(eta)
            y_rep = (rng.uniform(size=len(p)) < p).astype(float)
            observed.append([data.y.mean(), spread(data.y)]
                            + [gap_discrepancy(data.y[m], p[m], bins[n]) for n, m in subsets.items()]
                            + [player_dispersion(data.y, p, data, active)])
            replicated.append([y_rep.mean(), spread(y_rep)]
                              + [gap_discrepancy(y_rep[m], p[m], bins[n]) for n, m in subsets.items()]
                              + [player_dispersion(y_rep, p, data, active)])
        observed, replicated = np.array(observed), np.array(replicated)
        for j, name in enumerate(names):
            rows.append({"model": model, "statistic": name, "observed": observed[:, j].mean(),
                         "replicated_mean": replicated[:, j].mean(),
                         "replicated_q05": np.percentile(replicated[:, j], 5),
                         "replicated_q95": np.percentile(replicated[:, j], 95),
                         "p_value": np.mean(replicated[:, j] >= observed[:, j])})
        replicates[f"{model}, observed"] = observed[:, -1]
        replicates[f"{model}, replicated"] = replicated[:, -1]
        print(f"{model}: done", flush=True)

    table = pd.DataFrame(rows)
    TABLES.mkdir(exist_ok=True)
    table.to_csv(TABLES / "posterior_predictive_full.csv", index=False)
    pd.DataFrame(replicates).to_csv(TABLES / "posterior_predictive_full_replicates.csv", index=False,
                                    float_format="%.6g")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
