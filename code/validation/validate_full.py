"""Validation of the final model (sampler_full.py) on simulated games.

Same design as validate_simulated.py, with the real rating gaps, rated status, (first move, reply)
pairs and players: part 1 recovers values we choose, part 2 repeats it on many data sets with the
same values to separate a bias from chance misses, and part 3 measures interval coverage over data
sets whose parameters are drawn from the prior.

The spreads and mu describe effects that are drawn once per data set (20 first moves, 203 pairs,
14,832 players), so the posterior targets their realized values; part 1 reports those too.

Output: tables/validation_full_recovery.csv, tables/validation_full_repeated.csv,
tables/validation_full_coverage.csv
"""
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from core import sampler_full
from core.diagnostics import summarize
from core.sampler import Priors, load_games
from validation.validate_simulated import LEVELS, inside

TABLES = Path(__file__).resolve().parents[1] / "tables"
SEED = 20261118
TRUE_BETA = (1.95, 1.25)  # rated, unrated
TRUE_MU, TRUE_SIGMA_MOVE, TRUE_SIGMA_REPLY, TRUE_SIGMA_PLAYER = 0.08, 0.15, 0.10, 0.40
N_DATASETS = 100
N_REPEATS = 40
SCALARS = ("mu", "sigma_move", "sigma_reply", "sigma_player")
TRUTHS = dict(zip(SCALARS, (TRUE_MU, TRUE_SIGMA_MOVE, TRUE_SIGMA_REPLY, TRUE_SIGMA_PLAYER), strict=True))


def simulate_chosen(data, rng):
    """A data set simulated with the chosen values, and the realized effects."""
    gamma = rng.normal(TRUE_MU, TRUE_SIGMA_MOVE, size=len(data.family_names))
    theta = rng.normal(gamma[data.family], TRUE_SIGMA_REPLY)
    u = rng.normal(0, TRUE_SIGMA_PLAYER, size=data.n_players)
    simulated = sampler_full.simulate_outcomes(data, np.array(TRUE_BETA), theta, u, rng)
    realized = {"mu": gamma.mean(), "sigma_move": gamma.std(ddof=1),
                "sigma_reply": (theta - gamma[data.family]).std(ddof=1), "sigma_player": u.std(ddof=1)}
    return simulated, gamma, theta, u, realized


def recovery(data):
    rng = np.random.default_rng(SEED)
    simulated, gamma, theta, u, realized = simulate_chosen(data, rng)
    draws, players, acceptance = sampler_full.fit(simulated, Priors(), SEED + 1, parallel=True)

    quantities = [(f"beta, {name}", truth, np.nan, draws["beta"][..., k])
                  for k, (name, truth) in enumerate(zip(data.slope_names, TRUE_BETA, strict=True))]
    quantities += [(name, truth, realized[name], draws[name]) for name, truth in TRUTHS.items()]
    table = pd.DataFrame([{"quantity": name, "truth": truth, "realized": real, **summarize(q)}
                          for name, truth, real, q in quantities])
    table["inside_90"] = (table["truth"] >= table["q05"]) & (table["truth"] <= table["q95"])
    table["realized_inside_90"] = (table["realized"] >= table["q05"]) & (table["realized"] <= table["q95"])

    shares = {}
    for name, truth, sample in [("gamma", gamma, draws["gamma"]), ("theta", theta, draws["theta"]),
                                ("u", u, players["draws"])]:
        flat = sample.reshape(-1, len(truth))
        shares[name] = inside(flat, truth, 0.9).mean()
    return table, shares, acceptance


def one_repeat(args):
    data, seed = args
    rng = np.random.default_rng(seed)
    simulated, _, _, _, realized = simulate_chosen(data, rng)
    draws, _, _ = sampler_full.fit(simulated, Priors(), int(rng.integers(2**32)), n_chains=2, n_burn=1500,
                                   n_keep=3000, n_pilot=1500, u_every=3000)
    result = {}
    for k, truth in enumerate(TRUE_BETA):
        sample = draws["beta"][..., k].ravel()
        result[f"beta[{k}]_inside"], result[f"beta[{k}]_error"] = inside(sample, truth, 0.9), sample.mean() - truth
    for name, truth in TRUTHS.items():
        sample = draws[name].ravel()
        result[f"{name}_inside"], result[f"{name}_error"] = inside(sample, truth, 0.9), sample.mean() - truth
    return result


def repeated_recovery(data):
    seeds = np.random.SeedSequence(SEED + 3).spawn(N_REPEATS)
    with ProcessPoolExecutor(max_workers=max(1, min(12, (os.cpu_count() or 1) - 2))) as pool:
        results = pd.DataFrame(pool.map(one_repeat, [(data, s) for s in seeds], chunksize=1)).astype(float)
    names = [(f"beta[{k}]", f"beta, {name}", truth)
             for k, (name, truth) in enumerate(zip(data.slope_names, TRUE_BETA, strict=True))]
    names += [(name, name, truth) for name, truth in TRUTHS.items()]
    rows = []
    for key, name, truth in names:
        error = results[f"{key}_error"]
        rows.append({"quantity": name, "truth": truth, "share_inside_90": results[f"{key}_inside"].mean(),
                     "mean_error": error.mean(), "se_mean_error": error.std(ddof=1) / np.sqrt(len(error))})
    return pd.DataFrame(rows)


def one_dataset(args):
    data, seed = args
    rng = np.random.default_rng(seed)
    priors = Priors()
    beta, mu, tau_move, tau_reply, tau_player, gamma, theta, u = sampler_full.draw_from_prior(data, priors, rng)
    simulated = sampler_full.simulate_outcomes(data, beta, theta, u, rng)
    draws, players, _ = sampler_full.fit(simulated, priors, int(rng.integers(2**32)), n_chains=2, n_burn=1500,
                                         n_keep=3000, n_pilot=1500, u_every=10)
    truths = {"mu": mu, "sigma_move": 1 / np.sqrt(tau_move), "sigma_reply": 1 / np.sqrt(tau_reply),
              "sigma_player": 1 / np.sqrt(tau_player)}
    result = {}
    for level in LEVELS:
        for k in range(data.n_slopes):
            result[f"beta[{k}]_{level}"] = inside(draws["beta"][..., k].ravel(), beta[k], level)
        for name, truth in truths.items():
            result[f"{name}_{level}"] = inside(draws[name].ravel(), truth, level)
        for name, truth, sample in [("gamma", gamma, draws["gamma"]), ("theta", theta, draws["theta"]),
                                    ("u", u, players["draws"])]:
            result[f"{name}_{level}"] = inside(sample.reshape(-1, len(truth)), truth, level).mean()
    return result


def coverage(data):
    seeds = np.random.SeedSequence(SEED + 2).spawn(N_DATASETS)
    with ProcessPoolExecutor(max_workers=max(1, min(12, (os.cpu_count() or 1) - 2))) as pool:
        results = pd.DataFrame(pool.map(one_dataset, [(data, s) for s in seeds], chunksize=1)).astype(float)
    names = {f"beta[{k}]": f"beta, {name}" for k, name in enumerate(data.slope_names)}
    names.update({name: name for name in (*SCALARS, "gamma", "theta", "u")})
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
    table.to_csv(TABLES / "validation_full_recovery.csv", index=False)
    print(table.round(4).to_string(index=False))
    print("share of true values inside their 90% intervals: "
          + ", ".join(f"{name} {share:.3f}" for name, share in shares.items()))
    print(f"acceptance: beta {np.round(acceptance['beta'], 2)}, theta median {np.median(acceptance['theta']):.2f}, "
          f"u median {np.median(acceptance['u']):.2f}, rescaling {acceptance['scale_reply']:.2f} (replies) "
          f"{acceptance['scale_player']:.2f} (players)", flush=True)

    repeated = repeated_recovery(data)
    repeated.to_csv(TABLES / "validation_full_repeated.csv", index=False)
    print(repeated.round(4).to_string(index=False), flush=True)

    result = coverage(data)
    result.to_csv(TABLES / "validation_full_coverage.csv", index=False)
    print(result.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
