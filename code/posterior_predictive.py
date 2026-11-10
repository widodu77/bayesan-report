"""Posterior predictive checks of the fitted model.

For each of N_REPLICATES posterior draws of (beta, theta), new results are simulated for the real
games (same rating gaps, same openings) and test statistics are computed on the replicated and on
the observed results. A posterior predictive p-value P(T_rep >= T_obs | y) near 0 or 1 shows a
feature of the data that the model does not reproduce.

The win shares by bin of rating gap are many tests at once, so they are also combined into one
chi-square discrepancy that depends on the parameters (Gelman et al., Bayesian Data Analysis,
Section 6.3): X2(y, theta) = sum over bins of (wins - expected wins)^2 / variance, with p-value
P(X2(y_rep, theta) >= X2(y, theta) | y).

Input:  results/draws.npz (from fit_real.py)
Output: tables/posterior_predictive.csv, tables/posterior_predictive_replicates.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from sampler import load_games, simulate_outcomes

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"
RESULTS = ROOT / "results"
SEED = 20261111
N_REPLICATES = 4000
MIN_GAMES = 30  # openings with fewer games have win shares dominated by binomial noise
N_GAP_BINS = 10


def gap_bins(delta):
    """Bin index of each game by decile of the rating gap, and the bin edges in rating points."""
    edges = np.quantile(delta, np.linspace(0, 1, N_GAP_BINS + 1))
    index = np.clip(np.searchsorted(edges, delta, side="right") - 1, 0, N_GAP_BINS - 1)
    return index, 400 * edges


def statistic_names(edges):
    names = ["white win share, all games",
             f"sd of white win share between openings with at least {MIN_GAMES} games"]
    names += [f"white win share, rating gap {edges[k]:.0f} to {edges[k + 1]:.0f}" for k in range(N_GAP_BINS)]
    return names


def statistics(y, data, well_observed, bins):
    """Test statistics of one set of results y."""
    wins = np.bincount(data.opening, y, data.n_openings)
    shares = wins[well_observed] / data.games_per_opening[well_observed]
    by_gap = np.bincount(bins, y, N_GAP_BINS) / np.bincount(bins, minlength=N_GAP_BINS)
    return np.r_[y.mean(), shares.std(ddof=1), by_gap]


def gap_discrepancy(y, p, bins):
    """Chi-square discrepancy between the wins in each bin of rating gap and their expectation."""
    expected = np.bincount(bins, p, N_GAP_BINS)
    variance = np.bincount(bins, p * (1 - p), N_GAP_BINS)
    return np.sum((np.bincount(bins, y, N_GAP_BINS) - expected)**2 / variance)


def posterior_draws(path=RESULTS / "draws.npz"):
    """Flattened draws of (beta, theta), aligned with the thinning of theta in fit_real.py."""
    saved = np.load(path, allow_pickle=True)
    theta = saved["theta"].astype(float)
    thin = saved["beta"].shape[1] // theta.shape[1]
    beta = saved["beta"][:, ::thin]
    return beta.ravel(), theta.reshape(-1, theta.shape[-1]), saved["codes"]


def main():
    data, codes = load_games()
    beta, theta, saved_codes = posterior_draws()
    if not np.array_equal(codes, saved_codes):
        raise ValueError("results/draws.npz does not match the data; rerun fit_real.py")

    rng = np.random.default_rng(SEED)
    chosen = np.sort(rng.choice(len(beta), size=N_REPLICATES, replace=False))
    well_observed = data.games_per_opening >= MIN_GAMES
    bins, edges = gap_bins(data.delta)
    names = statistic_names(edges)

    observed = statistics(data.y, data, well_observed, bins)
    replicated, discrepancy = [], []
    for s in chosen:
        y_rep = simulate_outcomes(data, beta[s], theta[s], rng).y
        p = 1 / (1 + np.exp(-(theta[s][data.opening] + beta[s] * data.delta)))
        replicated.append(statistics(y_rep, data, well_observed, bins))
        discrepancy.append([gap_discrepancy(data.y, p, bins), gap_discrepancy(y_rep, p, bins)])
    replicated, discrepancy = np.array(replicated), np.array(discrepancy)

    p_upper = np.mean(replicated >= observed, axis=0)
    low, high = np.percentile(replicated, [5, 95], axis=0)
    table = pd.DataFrame({
        "statistic": names,
        "observed": observed,
        "replicated_mean": replicated.mean(axis=0),
        "replicated_q05": low,
        "replicated_q95": high,
        "p_value": p_upper,
        "p_value_two_sided": 2 * np.minimum(p_upper, 1 - p_upper),
    })
    chi2_observed, chi2_replicated = discrepancy.T
    table.loc[len(table)] = {
        "statistic": "chi-square discrepancy over the bins of rating gap",
        "observed": chi2_observed.mean(),
        "replicated_mean": chi2_replicated.mean(),
        "replicated_q05": np.percentile(chi2_replicated, 5),
        "replicated_q95": np.percentile(chi2_replicated, 95),
        "p_value": np.mean(chi2_replicated >= chi2_observed),
        "p_value_two_sided": np.nan,  # only large values signal a misfit
    }
    replicates = pd.DataFrame(replicated, columns=names)
    replicates["chi-square discrepancy, observed results"] = chi2_observed
    replicates["chi-square discrepancy, replicated results"] = chi2_replicated

    TABLES.mkdir(exist_ok=True)
    table.to_csv(TABLES / "posterior_predictive.csv", index=False)
    replicates.to_csv(TABLES / "posterior_predictive_replicates.csv", index=False, float_format="%.6g")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
