"""Fit the model with one rating slope for rated games and one for unrated games, and check whether
it removes the misfit of the posterior predictive check against the rating gap.

The posterior predictive checks of both models use the same statistics as posterior_predictive.py
and the same random numbers, so their p-values can be compared directly. The chi-square
discrepancy over deciles of the rating gap is computed on all games and within each group.

Input:  results/draws.npz (from fit_real.py), for the model with one slope
Output: results/draws_rating_slopes.npz, tables/posterior_rating_slopes.csv,
tables/favourite_win_probability.csv, tables/posterior_predictive_rating_slopes.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from diagnostics import summarize
from fit_real import N_BURN, N_KEEP, SEED, THETA_THIN, logistic
from posterior_predictive import MIN_GAMES, N_REPLICATES, gap_bins, gap_discrepancy, posterior_draws
from sampler import Priors, fit, load_games

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"
RESULTS = ROOT / "results"
PPC_SEED = 20261115
GAPS = (100, 200, 400)  # rating points


def favourite_score(beta, mu, gap):
    """Chance that the higher-rated player wins a decisive game in an average opening, averaged
    over the two colours: logistic(mu + beta x) as White and logistic(beta x - mu) as Black."""
    x = gap / 400
    return 0.5 * (logistic(mu + beta * x) + logistic(beta * x - mu))


def posterior_table(data, draws):
    beta = draws["beta"]
    quantities = {f"beta, {name}": beta[..., k] for k, name in enumerate(data.slope_names)}
    quantities["beta, rated minus unrated"] = beta[..., 0] - beta[..., 1]
    quantities["mu"] = draws["mu"]
    quantities["sigma"] = draws["sigma"]
    quantities["white advantage in rating points, rated games (400 mu / beta)"] = 400 * draws["mu"] / beta[..., 0]
    table = pd.DataFrame([{"quantity": name, **summarize(q)} for name, q in quantities.items()])
    table["prob_positive"] = [np.mean(q > 0) for q in quantities.values()]
    return table


def favourite_table(data, draws):
    rows = []
    for gap in GAPS:
        row = {"rating_gap": gap, "elo_expected_score": 1 / (1 + 10 ** (-gap / 400))}
        for k, name in enumerate(data.slope_names):
            score = favourite_score(draws["beta"][..., k], draws["mu"], gap)
            low, high = np.percentile(score, [5, 95])
            row.update({f"{name}, mean": score.mean(), f"{name}, q05": low, f"{name}, q95": high,
                        f"{name}, P(below Elo)": np.mean(score < row["elo_expected_score"])})
        rows.append(row)
    return pd.DataFrame(rows)


def predictive_checks(data, beta, theta, model_group, chosen, rng, model):
    """Posterior predictive p-values of one model; beta has shape (draws, slopes of that model)."""
    groups = data.slope_index
    well_observed = data.games_per_opening >= MIN_GAMES
    subsets = {"all games": np.ones(len(data.y), dtype=bool)}
    subsets.update({name: groups == k for k, name in enumerate(data.slope_names)})
    bins = {name: gap_bins(data.delta[mask])[0] for name, mask in subsets.items()}

    def spread(y):
        wins = np.bincount(data.opening, y, data.n_openings)
        return np.std(wins[well_observed] / data.games_per_opening[well_observed], ddof=1)

    observed_simple = np.array([data.y.mean(), spread(data.y)])
    replicated_simple, chi2_observed, chi2_replicated = [], [], []
    for s in chosen:
        p = logistic(theta[s][data.opening] + beta[s][model_group] * data.delta)
        y_rep = (rng.uniform(size=len(p)) < p).astype(float)
        replicated_simple.append([y_rep.mean(), spread(y_rep)])
        chi2_observed.append([gap_discrepancy(data.y[m], p[m], bins[n]) for n, m in subsets.items()])
        chi2_replicated.append([gap_discrepancy(y_rep[m], p[m], bins[n]) for n, m in subsets.items()])
    replicated_simple = np.array(replicated_simple)
    chi2_observed, chi2_replicated = np.array(chi2_observed), np.array(chi2_replicated)

    rows = []
    for j, name in enumerate(["white win share, all games",
                              f"sd of white win share between openings with at least {MIN_GAMES} games"]):
        rows.append({"model": model, "statistic": name, "observed": observed_simple[j],
                     "replicated_mean": replicated_simple[:, j].mean(),
                     "replicated_q05": np.percentile(replicated_simple[:, j], 5),
                     "replicated_q95": np.percentile(replicated_simple[:, j], 95),
                     "p_value": np.mean(replicated_simple[:, j] >= observed_simple[j])})
    for j, name in enumerate(subsets):
        rows.append({"model": model, "statistic": f"chi-square over deciles of rating gap, {name}",
                     "observed": chi2_observed[:, j].mean(), "replicated_mean": chi2_replicated[:, j].mean(),
                     "replicated_q05": np.percentile(chi2_replicated[:, j], 5),
                     "replicated_q95": np.percentile(chi2_replicated[:, j], 95),
                     "p_value": np.mean(chi2_replicated[:, j] >= chi2_observed[:, j])})
    return rows


def main():
    data, codes = load_games(slope_by="rated")
    draws, acceptance, _ = fit(data, Priors(), SEED, n_burn=N_BURN, n_keep=N_KEEP, n_pilot=3000, parallel=True)

    RESULTS.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    theta = draws["theta"][:, ::THETA_THIN].astype(np.float32)
    np.savez_compressed(RESULTS / "draws_rating_slopes.npz", beta=draws["beta"], mu=draws["mu"],
                        sigma=draws["sigma"], theta=theta, codes=codes, slope_names=np.array(data.slope_names))

    posterior = posterior_table(data, draws)
    posterior.to_csv(TABLES / "posterior_rating_slopes.csv", index=False)
    favourite = favourite_table(data, draws)
    favourite.to_csv(TABLES / "favourite_win_probability.csv", index=False)
    print(posterior.round(4).to_string(index=False))
    print(f"acceptance: beta {np.round(acceptance['beta'], 3)}, theta median {np.median(acceptance['theta']):.3f}")
    print(favourite.round(3).to_string(index=False), flush=True)

    # Both models use the same thinned draws and the same random numbers in their checks.
    one_beta, one_theta, saved_codes = posterior_draws()
    if not np.array_equal(codes, saved_codes):
        raise ValueError("results/draws.npz does not match the data; rerun fit_real.py")
    two_beta = draws["beta"][:, ::THETA_THIN].reshape(-1, data.n_slopes)
    two_theta = theta.reshape(-1, data.n_openings).astype(float)
    chosen = np.sort(np.random.default_rng(PPC_SEED).choice(len(one_beta), size=N_REPLICATES, replace=False))
    rows = predictive_checks(data, one_beta[:, None], one_theta, np.zeros(len(data.y), dtype=int), chosen,
                             np.random.default_rng(PPC_SEED + 1), "one slope")
    rows += predictive_checks(data, two_beta, two_theta, data.slope_index, chosen,
                              np.random.default_rng(PPC_SEED + 1), "rated and unrated slopes")
    checks = pd.DataFrame(rows)
    checks.to_csv(TABLES / "posterior_predictive_rating_slopes.csv", index=False)
    print(checks.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
