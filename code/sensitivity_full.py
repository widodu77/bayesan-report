"""Prior and data sensitivity of the final model (sampler_full.py).

Each variant refits the final model with the same seed and settings, so differences come from the
variant. For each one the table reports the shared parameters and the headline claims: White's
advantage after 1.e4, how much the first move and the reply matter, the Sicilian against 1...e5, and 1.e3.

The diffuse prior on the precisions lets the spread between first moves, learned from only 20
first moves, wander towards zero, where the chains move slowly; that variant gets chains four times
longer (LONG_KEEP) to reach the same convergence as the others.

Usage: python sensitivity_full.py [--only VARIANT], where --only refits one variant and replaces its row.
Output: tables/sensitivity_full.csv
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import sampler_full
from diagnostics import effective_sample_size, split_rhat
from sampler import Priors, load_games

TABLES = Path(__file__).resolve().parent / "tables"
SEED = 20261121
N_BURN, N_KEEP, LONG_KEEP = 2000, 20_000, 80_000
LONG_CHAINS = {"tau Gamma(0.5, 0.005)"}


def logistic(x):
    return 1 / (1 + np.exp(-x))


def variants():
    base = Priors()
    full = {"openings": "moves", "slope_by": "rated"}
    return [
        ("baseline", "final model and priors", full, base),
        ("tau Gamma(2, 0.1)", "the three spreads: same prior mean of tau, half the variance", full,
         replace(base, tau_shape=2.0, tau_rate=0.1)),
        ("tau Gamma(0.5, 0.005)", "the three spreads: diffuse, prior mean 100", full,
         replace(base, tau_shape=0.5, tau_rate=0.005)),
        ("mu sd 1", "mu ~ N(0, 1)", full, replace(base, mu_sd=1.0)),
        ("beta N(0, 1)", "slopes centred on 0 instead of log 10", full, replace(base, beta_mean=0.0)),
        ("rated games only", "unrated games removed, one slope", {"openings": "moves", "rated_only": True}, base),
        ("no player rated 1500", "games with a player at the possible default rating removed",
         {**full, "drop_rating": 1500}, base),
        ("draws as half wins", "draws kept with y = 1/2", {**full, "draws_as_half": True}, base),
        ("at least 20 plies", "games shorter than 10 moves per side removed", {**full, "min_plies": 20}, base),
    ]


def summary(name, description, data, codes, draws):
    beta = draws["beta"] if draws["beta"].ndim == 3 else draws["beta"][..., None]
    gamma, theta = draws["gamma"], draws["theta"]
    moves, pairs = list(data.family_names), list(codes)
    e4, e3 = moves.index("e4"), moves.index("e3")
    sicilian, open_game = pairs.index("e4 c5"), pairs.index("e4 e5")
    row = {"variant": name, "description": description, "games": len(data.y)}
    quantities = {
        "beta_rated": beta[..., 0],
        "sigma_player": draws["sigma_player"],
        "sigma_move": draws["sigma_move"],
        "sigma_reply": draws["sigma_reply"],
        "advantage_e4_points": 400 * gamma[..., e4] / beta[..., 0],
        "p_white_e4": logistic(gamma[..., e4]),
        "p_white_e3": logistic(gamma[..., e3]),
        "p_white_e4_c5": logistic(theta[..., sicilian]),
        "p_white_e4_e5": logistic(theta[..., open_game]),
    }
    for key, q in quantities.items():
        low, high = np.percentile(q, [5, 95])
        row.update({f"{key}_mean": q.mean(), f"{key}_q05": low, f"{key}_q95": high})
    row["prob_move_matters_more"] = np.mean(draws["sigma_move"] > draws["sigma_reply"])
    row["prob_sicilian_better_for_black"] = np.mean(theta[..., sicilian] < theta[..., open_game])
    row["prob_e3_below_e4"] = np.mean(gamma[..., e3] < gamma[..., e4])
    shared = [draws[n] for n in ("mu", "sigma_move", "sigma_reply", "sigma_player")] + \
             [beta[..., k] for k in range(beta.shape[-1])]
    row["largest_rhat"] = max(split_rhat(s) for s in shared)
    row["smallest_ess"] = min(effective_sample_size(s) for s in shared)
    return row


def fit_variant(name, description, options, priors):
    data, codes = load_games(**options)
    n_keep = LONG_KEEP if name in LONG_CHAINS else N_KEEP
    draws, _, _ = sampler_full.fit(data, priors, SEED, n_burn=N_BURN, n_keep=n_keep, n_pilot=3000,
                                   parallel=True, u_every=n_keep)
    row = summary(name, description, data, codes, draws)
    row["kept_draws_per_chain"] = n_keep
    print(f"{name}: beta {row['beta_rated_mean']:.3f}, sigma_player {row['sigma_player_mean']:.3f}, "
          f"e4 {100 * row['p_white_e4_mean']:.1f}%, c5 {100 * row['p_white_e4_c5_mean']:.1f}% vs e5 "
          f"{100 * row['p_white_e4_e5_mean']:.1f}% (P {row['prob_sicilian_better_for_black']:.3f}), "
          f"P(move > reply) {row['prob_move_matters_more']:.2f}, Rhat {row['largest_rhat']:.3f}, "
          f"ESS {row['smallest_ess']:.0f}", flush=True)
    return row


def main():
    path = TABLES / "sensitivity_full.csv"
    if "--only" in sys.argv:
        name = sys.argv[sys.argv.index("--only") + 1]
        variant = next(v for v in variants() if v[0] == name)
        table = pd.read_csv(path)
        row = fit_variant(*variant)
        for column, value in row.items():
            table.loc[table["variant"] == name, column] = value
        table.to_csv(path, index=False)
        return
    rows = []
    for variant in variants():
        rows.append(fit_variant(*variant))
        pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()
