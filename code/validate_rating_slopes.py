"""Validation of the model with one rating slope for rated games and one for unrated games.

Same design as validate_simulated.py, with the real rating gaps, openings and rated status:
part 1 recovers values we choose, part 2 measures interval coverage over data sets whose
parameters are drawn from the prior.

Output: tables/validation_rating_slopes_recovery.csv, tables/validation_rating_slopes_coverage.csv
"""
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from diagnostics import summarize
from sampler import Priors, draw_from_prior, fit, load_games, simulate_outcomes
from validate_simulated import LEVELS, inside

TABLES = Path(__file__).resolve().parent / "tables"
SEED = 20261114
TRUE_BETA = (1.75, 1.10)  # rated, unrated: close to the separate fits of the two groups
TRUE_MU, TRUE_SIGMA = 0.08, 0.15
N_DATASETS = 100


def recovery(data):
    rng = np.random.default_rng(SEED)
    theta = rng.normal(TRUE_MU, TRUE_SIGMA, size=data.n_openings)
    simulated = simulate_outcomes(data, np.array(TRUE_BETA), theta, rng)
    draws, acceptance, _ = fit(simulated, Priors(), SEED + 1)

    quantities = [(f"beta, {name}", truth, draws["beta"][..., k])
                  for k, (name, truth) in enumerate(zip(data.slope_names, TRUE_BETA, strict=True))]
    quantities += [("beta, rated minus unrated", TRUE_BETA[0] - TRUE_BETA[1],
                    draws["beta"][..., 0] - draws["beta"][..., 1]),
                   ("mu", TRUE_MU, draws["mu"]), ("sigma", TRUE_SIGMA, draws["sigma"])]
    table = pd.DataFrame([{"quantity": name, "truth": truth, **summarize(q)} for name, truth, q in quantities])
    table["inside_90"] = (table["truth"] >= table["q05"]) & (table["truth"] <= table["q95"])
    low, high = np.percentile(draws["theta"], [5, 95], axis=(0, 1))
    return table, np.mean((theta >= low) & (theta <= high)), acceptance


def one_dataset(args):
    data, seed = args
    rng = np.random.default_rng(seed)
    priors = Priors()
    beta, mu, tau, theta = draw_from_prior(data.n_openings, priors, rng, data.n_slopes)
    simulated = simulate_outcomes(data, beta, theta, rng)
    draws, _, _ = fit(simulated, priors, int(rng.integers(2**32)), n_chains=2, n_burn=1500, n_keep=3000,
                      n_pilot=1500)
    result = {}
    for level in LEVELS:
        for k in range(data.n_slopes):
            result[f"beta[{k}]_{level}"] = inside(draws["beta"][..., k].ravel(), beta[k], level)
        result[f"mu_{level}"] = inside(draws["mu"].ravel(), mu, level)
        result[f"sigma_{level}"] = inside(draws["sigma"].ravel(), 1 / np.sqrt(tau), level)
        result[f"theta_{level}"] = inside(draws["theta"].reshape(-1, data.n_openings), theta, level).mean()
    return result


def coverage(data):
    seeds = np.random.SeedSequence(SEED + 2).spawn(N_DATASETS)
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() - 2)) as pool:
        results = pd.DataFrame(pool.map(one_dataset, [(data, s) for s in seeds], chunksize=1)).astype(float)

    names = {f"beta[{k}]": f"beta, {name}" for k, name in enumerate(data.slope_names)}
    names.update({"mu": "mu", "sigma": "sigma", "theta": "theta"})
    rows = []
    for key, name in names.items():
        row = {"quantity": name}
        for level in LEVELS:
            column = results[f"{key}_{level}"]
            row[f"coverage_{int(level * 100)}"] = column.mean()
            row[f"se_{int(level * 100)}"] = column.std(ddof=1) / np.sqrt(len(column))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    data, _ = load_games(slope_by="rated")
    table, theta_coverage, acceptance = recovery(data)
    TABLES.mkdir(exist_ok=True)
    table.to_csv(TABLES / "validation_rating_slopes_recovery.csv", index=False)
    print(table.round(4).to_string(index=False))
    print(f"share of true theta inside their 90% intervals: {theta_coverage:.3f}")
    print(f"acceptance: beta {np.round(acceptance['beta'], 2)}, theta median {np.median(acceptance['theta']):.2f}",
          flush=True)

    result = coverage(data)
    result.to_csv(TABLES / "validation_rating_slopes_coverage.csv", index=False)
    print(result.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
