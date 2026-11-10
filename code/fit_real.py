"""Fit the model to the real games and write posterior summaries and diagnostics.

Output: results/draws.npz, tables/posterior_shared.csv, tables/posterior_openings.csv,
tables/sampler_diagnostics.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from diagnostics import effective_sample_size, split_rhat, summarize
from sampler import PROCESSED, Priors, fit, load_games

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"
RESULTS = ROOT / "results"
SEED = 20261110
THETA_THIN = 10
N_BURN = 2000
N_KEEP = 40_000  # per chain, for an ESS of sigma above 1,000


def logistic(x):
    return 1 / (1 + np.exp(-x))


def opening_labels(codes):
    """The most frequent opening name of each ECO code."""
    games = pd.read_csv(PROCESSED).query("winner != 'draw'")
    names = games.groupby("opening_eco")["opening_name"].agg(lambda s: s.value_counts().index[0])
    return names.loc[codes].to_numpy()


def shared_table(draws):
    quantities = {
        "beta": draws["beta"],
        "mu": draws["mu"],
        "sigma": draws["sigma"],
        "white advantage in rating points (400 mu / beta)": 400 * draws["mu"] / draws["beta"],
        "P(white wins) at equal ratings, average opening": logistic(draws["mu"]),
    }
    table = pd.DataFrame([{"quantity": name, **summarize(q)} for name, q in quantities.items()])
    table["prob_positive"] = [np.mean(q > 0) for q in quantities.values()]
    return table


def opening_table(data, codes, draws):
    theta = draws["theta"]
    above_average = theta - draws["mu"][:, :, None]
    low, median, high = np.percentile(theta, [5, 50, 95], axis=(0, 1))
    wins = np.bincount(data.opening, data.y, data.n_openings)
    games = data.games_per_opening
    return pd.DataFrame({
        "opening_eco": codes,
        "opening_name": opening_labels(codes),
        "games": games,
        "white_win_share": wins / games,
        "theta_mean": theta.mean(axis=(0, 1)),
        "theta_q05": low,
        "theta_median": median,
        "theta_q95": high,
        "prob_better_than_average": np.mean(above_average > 0, axis=(0, 1)),
        "rhat": [split_rhat(theta[:, :, j]) for j in range(data.n_openings)],
        "ess": [effective_sample_size(theta[:, :, j]) for j in range(data.n_openings)],
    })


def diagnostics_table(openings, shared, acceptance):
    rows = {
        "acceptance rate of beta": acceptance["beta"],
        "median acceptance rate of theta": np.median(acceptance["theta"]),
        "lowest acceptance rate of theta": acceptance["theta"].min(),
        "highest acceptance rate of theta": acceptance["theta"].max(),
        "largest split-Rhat, shared parameters": shared["rhat"].max(),
        "smallest ESS, shared parameters": shared["ess"].min(),
        "largest split-Rhat, theta": openings["rhat"].max(),
        "smallest ESS, theta": openings["ess"].min(),
    }
    return pd.DataFrame({"statistic": list(rows), "value": list(rows.values())})


def main():
    data, codes = load_games()
    chains, acceptance, _ = fit(data, Priors(), SEED, n_burn=N_BURN, n_keep=N_KEEP, n_pilot=3000, parallel=True,
                                keep_burn_in=True)
    draws = {name: chain[:, N_BURN:] for name, chain in chains.items()}

    TABLES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    # The burn-in of the shared parameters is kept only for the trace plots.
    np.savez_compressed(RESULTS / "draws.npz", beta=draws["beta"], mu=draws["mu"], sigma=draws["sigma"],
                        theta=draws["theta"][:, ::THETA_THIN].astype(np.float32), codes=codes,
                        **{f"burn_in_{name}": chains[name][:, :N_BURN] for name in ("beta", "mu", "sigma")})

    shared = shared_table(draws)
    openings = opening_table(data, codes, draws)
    diagnostics = diagnostics_table(openings, shared, acceptance)
    shared.to_csv(TABLES / "posterior_shared.csv", index=False)
    openings.to_csv(TABLES / "posterior_openings.csv", index=False)
    diagnostics.to_csv(TABLES / "sampler_diagnostics.csv", index=False)

    print(shared.round(4).to_string(index=False))
    print()
    print(diagnostics.round(4).to_string(index=False))
    print()
    ordered = openings[openings["games"] >= 30].sort_values("theta_mean", ascending=False)
    columns = ["opening_eco", "opening_name", "games", "white_win_share", "theta_mean", "theta_q05",
               "theta_q95", "prob_better_than_average"]
    print(pd.concat([ordered.head(5), ordered.tail(5)])[columns].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
