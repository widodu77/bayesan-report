"""Optional cross-check of our sampler with PyMC (NUTS) on the same model, priors and data.

PyMC is used only to check our results, never to produce them, so this script and its package are
kept apart: install them with `pip install -r requirements-crosscheck.txt`. The opening effects are
written as theta_o = mu + z_o / sqrt(tau) with z_o ~ N(0, 1), which is the same model and is easier
for NUTS.

Input:  tables/posterior_shared.csv, tables/posterior_openings.csv (from fit_real.py)
Output: tables/crosscheck_pymc.csv
"""
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from sampler import Priors, load_games

TABLES = Path(__file__).resolve().parent / "tables"
SEED = 20261113
OURS = {"beta": "beta", "mu": "mu", "sigma": "sigma"}


def fit_pymc(data, priors):
    with pm.Model():
        mu = pm.Normal("mu", priors.mu_mean, priors.mu_sd)
        tau = pm.Gamma("tau", alpha=priors.tau_shape, beta=priors.tau_rate)
        beta = pm.Normal("beta", priors.beta_mean, priors.beta_sd)
        z = pm.Normal("z", 0, 1, shape=data.n_openings)
        theta = pm.Deterministic("theta", mu + z / pm.math.sqrt(tau))
        pm.Deterministic("sigma", 1 / pm.math.sqrt(tau))
        pm.Bernoulli("y", logit_p=theta[data.opening] + beta * data.delta, observed=data.y)
        return pm.sample(draws=4000, tune=2000, chains=4, cores=4, target_accept=0.9, random_seed=SEED,
                         progressbar=False)


def main():
    data, codes = load_games()
    trace = fit_pymc(data, Priors())
    ours = pd.read_csv(TABLES / "posterior_shared.csv").set_index("quantity")
    summary = az.summary(trace, var_names=["beta", "mu", "sigma"], stat_focus="mean", round_to="none")
    divergences = int(trace.sample_stats["diverging"].sum())

    rows = []
    for name in ("beta", "mu", "sigma"):
        ref, new = ours.loc[OURS[name]], summary.loc[name]
        rows.append({
            "quantity": name,
            "our_mean": ref["mean"], "our_sd": ref["sd"], "our_mcse": ref["mcse"],
            "pymc_mean": new["mean"], "pymc_sd": new["sd"], "pymc_mcse": new["mcse_mean"],
            "z_mean": (ref["mean"] - new["mean"]) / np.hypot(ref["mcse"], new["mcse_mean"]),
            "relative_difference_sd": new["sd"] / ref["sd"] - 1,
            "pymc_rhat": new["r_hat"], "pymc_ess": new["ess_bulk"],
        })

    openings = pd.read_csv(TABLES / "posterior_openings.csv").set_index("opening_eco").loc[codes]
    theta = trace.posterior["theta"].to_numpy().reshape(-1, data.n_openings)
    gap = np.abs(theta.mean(axis=0) - openings["theta_mean"].to_numpy()) / theta.std(axis=0)
    rows.append({"quantity": "theta, largest |difference of means| / posterior sd over the openings",
                 "value": gap.max()})
    table = pd.DataFrame(rows)
    table["pymc_divergences"] = divergences

    TABLES.mkdir(exist_ok=True)
    table.to_csv(TABLES / "crosscheck_pymc.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
