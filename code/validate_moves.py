"""Validation of the model of White's first move and Black's reply (sampler_moves.py).

Same design as validate_simulated.py, with the real rating gaps, rated status and (first move,
reply) pairs, and one rating slope for rated and one for unrated games as in fit_moves.py:
part 1 recovers values we choose, part 2 measures interval coverage over data sets whose
parameters are drawn from the prior.

Output: tables/validation_moves_recovery.csv, tables/validation_moves_coverage.csv
"""
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import sampler_moves
from diagnostics import summarize
from sampler import Priors, fit, load_games, simulate_outcomes
from validate_simulated import LEVELS, inside

TABLES = Path(__file__).resolve().parent / "tables"
SEED = 20261116
TRUE_BETA = (1.75, 1.10)  # rated, unrated
TRUE_MU, TRUE_SIGMA_MOVE, TRUE_SIGMA_REPLY = 0.08, 0.15, 0.10
N_DATASETS = 100


def recovery(data):
    rng = np.random.default_rng(SEED)
    gamma = rng.normal(TRUE_MU, TRUE_SIGMA_MOVE, size=len(data.family_names))
    theta = rng.normal(gamma[data.family], TRUE_SIGMA_REPLY)
    simulated = simulate_outcomes(data, np.array(TRUE_BETA), theta, rng)
    draws, acceptance, _ = fit(simulated, Priors(), SEED + 1, run=sampler_moves.run_chain)

    quantities = [(f"beta, {name}", truth, draws["beta"][..., k])
                  for k, (name, truth) in enumerate(zip(data.slope_names, TRUE_BETA, strict=True))]
    quantities += [("mu", TRUE_MU, draws["mu"]),
                  ("sigma_move", TRUE_SIGMA_MOVE, draws["sigma_move"]),
                  ("sigma_reply", TRUE_SIGMA_REPLY, draws["sigma_reply"])]
    table = pd.DataFrame([{"quantity": name, "truth": truth, **summarize(q)} for name, truth, q in quantities])
    table["inside_90"] = (table["truth"] >= table["q05"]) & (table["truth"] <= table["q95"])
    shares = {}
    for name, truth in [("gamma", gamma), ("theta", theta)]:
        low, high = np.percentile(draws[name], [5, 95], axis=(0, 1))
        shares[name] = np.mean((truth >= low) & (truth <= high))
    return table, shares, acceptance


def one_dataset(args):
    data, seed = args
    rng = np.random.default_rng(seed)
    priors = Priors()
    beta, mu, tau_move, tau_reply, gamma, theta = sampler_moves.draw_from_prior(data, priors, rng)
    simulated = simulate_outcomes(data, beta, theta, rng)
    draws, _, _ = fit(simulated, priors, int(rng.integers(2**32)), n_chains=2, n_burn=1500, n_keep=3000,
                      n_pilot=1500, run=sampler_moves.run_chain)
    truths = {"mu": mu, "sigma_move": 1 / np.sqrt(tau_move), "sigma_reply": 1 / np.sqrt(tau_reply)}
    result = {}
    for level in LEVELS:
        for k in range(data.n_slopes):
            result[f"beta[{k}]_{level}"] = inside(draws["beta"][..., k].ravel(), beta[k], level)
        for name, truth in truths.items():
            result[f"{name}_{level}"] = inside(draws[name].ravel(), truth, level)
        for name, truth in [("gamma", gamma), ("theta", theta)]:
            result[f"{name}_{level}"] = inside(draws[name].reshape(-1, len(truth)), truth, level).mean()
    return result


def coverage(data):
    seeds = np.random.SeedSequence(SEED + 2).spawn(N_DATASETS)
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() - 2)) as pool:
        results = pd.DataFrame(pool.map(one_dataset, [(data, s) for s in seeds], chunksize=1)).astype(float)
    names = {f"beta[{k}]": f"beta, {name}" for k, name in enumerate(data.slope_names)}
    names.update({name: name for name in ("mu", "sigma_move", "sigma_reply", "gamma", "theta")})
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
    data, _ = load_games(openings="moves", slope_by="rated")
    table, shares, acceptance = recovery(data)
    TABLES.mkdir(exist_ok=True)
    table.to_csv(TABLES / "validation_moves_recovery.csv", index=False)
    print(table.round(4).to_string(index=False))
    print(f"share of true values inside their 90% intervals: first moves {shares['gamma']:.3f}, "
          f"pairs {shares['theta']:.3f}")
    print(f"acceptance: beta {np.round(acceptance['beta'], 2)}, theta median {np.median(acceptance['theta']):.2f}", flush=True)

    result = coverage(data)
    result.to_csv(TABLES / "validation_moves_coverage.csv", index=False)
    print(result.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
