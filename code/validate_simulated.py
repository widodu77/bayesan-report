"""Validation on simulated games, using the real rating gaps and openings.

Part 1 simulates one data set with values we choose and checks that the posterior recovers them.
Part 2 repeats part 1 on many data sets with the same chosen values, to separate a bias from the
chance miss that a 90% interval makes one time in ten.
Part 3 draws the parameters from the prior for many data sets and measures interval coverage,
which is exact on average over the prior when the sampler is correct.

Output: tables/validation_recovery.csv, tables/validation_recovery_theta.csv, tables/validation_repeated.csv,
tables/validation_coverage.csv
"""
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from diagnostics import summarize
from sampler import Priors, draw_from_prior, fit, load_games, simulate_outcomes

TABLES = Path(__file__).resolve().parent / "tables"
SEED = 20261109
TRUE_BETA, TRUE_MU, TRUE_SIGMA = 1.6, 0.08, 0.20
N_DATASETS = 100
N_REPEATS = 40
LEVELS = (0.5, 0.9)


def recovery(data):
    rng = np.random.default_rng(SEED)
    theta = rng.normal(TRUE_MU, TRUE_SIGMA, size=data.n_openings)
    simulated = simulate_outcomes(data, TRUE_BETA, theta, rng)
    draws, acceptance, _ = fit(simulated, Priors(), SEED + 1)

    rows = []
    for name, truth in [("beta", TRUE_BETA), ("mu", TRUE_MU), ("sigma", TRUE_SIGMA)]:
        rows.append({"quantity": name, "truth": truth, **summarize(draws[name])})
    table = pd.DataFrame(rows)
    table["inside_90"] = (table["truth"] >= table["q05"]) & (table["truth"] <= table["q95"])

    low, high = np.percentile(draws["theta"], [5, 95], axis=(0, 1))
    openings = pd.DataFrame({"games": data.games_per_opening, "truth": theta,
                             "mean": draws["theta"].mean(axis=(0, 1)), "q05": low, "q95": high})
    openings["inside_90"] = (theta >= low) & (theta <= high)
    return table, openings, acceptance


def one_repeat(args):
    data, seed = args
    rng = np.random.default_rng(seed)
    theta = rng.normal(TRUE_MU, TRUE_SIGMA, size=data.n_openings)
    simulated = simulate_outcomes(data, TRUE_BETA, theta, rng)
    draws, _, _ = fit(simulated, Priors(), int(rng.integers(2**32)), n_chains=2, n_burn=1500, n_keep=3000,
                      n_pilot=1500)
    result = {}
    for name, truth in [("beta", TRUE_BETA), ("mu", TRUE_MU), ("sigma", TRUE_SIGMA)]:
        result[f"{name}_inside"] = inside(draws[name].ravel(), truth, 0.9)
        result[f"{name}_error"] = draws[name].mean() - truth
    return result


def repeated_recovery(data):
    seeds = np.random.SeedSequence(SEED + 3).spawn(N_REPEATS)
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() - 2)) as pool:
        results = pd.DataFrame(pool.map(one_repeat, [(data, s) for s in seeds], chunksize=1)).astype(float)
    rows = []
    for name, truth in [("beta", TRUE_BETA), ("mu", TRUE_MU), ("sigma", TRUE_SIGMA)]:
        error = results[f"{name}_error"]
        rows.append({"quantity": name, "truth": truth, "share_inside_90": results[f"{name}_inside"].mean(),
                     "mean_error": error.mean(), "se_mean_error": error.std(ddof=1) / np.sqrt(len(error))})
    return pd.DataFrame(rows)


def inside(draws, truth, level):
    """Whether the truth lies in the central interval; draws have shape (draws, ...)."""
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(draws, [tail, 100 - tail], axis=0)
    return (truth >= low) & (truth <= high)


def one_dataset(args):
    data, seed = args
    rng = np.random.default_rng(seed)
    priors = Priors()
    beta, mu, tau, theta = draw_from_prior(data.n_openings, priors, rng)
    simulated = simulate_outcomes(data, beta, theta, rng)
    draws, _, _ = fit(simulated, priors, int(rng.integers(2**32)), n_chains=2, n_burn=1500, n_keep=3000, n_pilot=1500)

    result = {}
    for level in LEVELS:
        result[f"beta_{level}"] = inside(draws["beta"].ravel(), beta, level)
        result[f"mu_{level}"] = inside(draws["mu"].ravel(), mu, level)
        result[f"sigma_{level}"] = inside(draws["sigma"].ravel(), 1 / np.sqrt(tau), level)
        result[f"theta_{level}"] = inside(draws["theta"].reshape(-1, data.n_openings), theta, level).mean()
    return result


def coverage(data):
    seeds = np.random.SeedSequence(SEED + 2).spawn(N_DATASETS)
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() - 2)) as pool:
        results = pd.DataFrame(pool.map(one_dataset, [(data, s) for s in seeds], chunksize=1)).astype(float)

    rows = []
    for name in ("beta", "mu", "sigma", "theta"):
        row = {"quantity": name}
        for level in LEVELS:
            column = results[f"{name}_{level}"]
            row[f"coverage_{int(level * 100)}"] = column.mean()
            row[f"se_{int(level * 100)}"] = column.std(ddof=1) / np.sqrt(len(column))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    data, codes = load_games()
    table, openings, acceptance = recovery(data)
    TABLES.mkdir(exist_ok=True)
    table.to_csv(TABLES / "validation_recovery.csv", index=False)
    openings.insert(0, "opening_eco", codes)
    openings.to_csv(TABLES / "validation_recovery_theta.csv", index=False)
    print(table.round(4).to_string(index=False))
    print(f"share of true theta inside their 90% intervals: {openings['inside_90'].mean():.3f}")
    print(f"acceptance: beta {acceptance['beta']:.2f}, theta median {np.median(acceptance['theta']):.2f}, "
          f"range {acceptance['theta'].min():.2f}-{acceptance['theta'].max():.2f}", flush=True)

    repeated = repeated_recovery(data)
    repeated.to_csv(TABLES / "validation_repeated.csv", index=False)
    print(repeated.round(4).to_string(index=False), flush=True)

    result = coverage(data)
    result.to_csv(TABLES / "validation_coverage.csv", index=False)
    print(result.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
