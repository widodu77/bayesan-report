"""Fit M2, the final model without player effects (sampler_moves.py), to the real games.

Its draws are used only by posterior_predictive_full.py, which compares M2 with the final model:
without player effects, the games of a player are not independent (the player dispersion check).

Output: results/draws_moves.npz
"""
from pathlib import Path

import numpy as np

import sampler_moves
from fit_real import N_BURN, N_KEEP, THETA_THIN
from sampler import Priors, fit, load_games

RESULTS = Path(__file__).resolve().parent / "results"
SEED = 20261117


def main():
    data, codes = load_games(openings="moves", slope_by="rated")
    draws, _, _ = fit(data, Priors(), SEED, n_burn=N_BURN, n_keep=N_KEEP, n_pilot=3000, parallel=True,
                      run=sampler_moves.run_chain)
    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(RESULTS / "draws_moves.npz", beta=draws["beta"], mu=draws["mu"],
                        sigma_move=draws["sigma_move"], sigma_reply=draws["sigma_reply"],
                        gamma=draws["gamma"][:, ::THETA_THIN].astype(np.float32),
                        theta=draws["theta"][:, ::THETA_THIN].astype(np.float32), codes=codes,
                        family_names=np.array(data.family_names), slope_names=np.array(data.slope_names))
    print(f"saved {RESULTS / 'draws_moves.npz'}")


if __name__ == "__main__":
    main()
