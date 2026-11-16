"""Fit the model of White's first move and Black's reply (sampler_moves.py) to the real games.

The opening is split into the two choices that each player makes alone: White's first move, then
Black's reply. The model keeps one rating slope for rated and one for unrated games
(fit_rating_slopes.py), which removed the misfit of the posterior predictive check.

Output: results/draws_moves.npz, tables/posterior_moves_shared.csv, tables/posterior_first_moves.csv,
tables/posterior_replies.csv, tables/sampler_diagnostics_moves.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import sampler_moves
from diagnostics import effective_sample_size, split_rhat, summarize
from fit_real import N_BURN, N_KEEP, THETA_THIN, logistic
from sampler import Priors, fit, load_games

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"
RESULTS = ROOT / "results"
SEED = 20261117
MIN_GAMES_CONTRAST = 300  # first moves whose replies are compared with the most played reply


def shared_table(data, draws):
    beta = draws["beta"]
    quantities = {f"beta, {name}": beta[..., k] for k, name in enumerate(data.slope_names)}
    quantities.update({
        "mu": draws["mu"],
        "sigma_move (spread between White's first moves)": draws["sigma_move"],
        "sigma_reply (spread between Black's replies to one first move)": draws["sigma_reply"],
        "sigma_move minus sigma_reply": draws["sigma_move"] - draws["sigma_reply"],
        "white advantage in rating points, rated games (400 mu / beta)": 400 * draws["mu"] / beta[..., 0],
    })
    table = pd.DataFrame([{"quantity": name, **summarize(q)} for name, q in quantities.items()])
    table["prob_positive"] = [np.mean(q > 0) for q in quantities.values()]
    return table


def first_move_table(data, draws):
    gamma = draws["gamma"].reshape(-1, len(data.family_names))
    games = np.bincount(data.family[data.opening], minlength=len(data.family_names))
    wins = np.bincount(data.family[data.opening], data.y, len(data.family_names))
    best = np.bincount(gamma.argmax(axis=1), minlength=len(data.family_names)) / len(gamma)
    probability = logistic(gamma)
    low, high = np.percentile(probability, [5, 95], axis=0)
    return pd.DataFrame({
        "first_move": data.family_names,
        "games": games,
        "white_win_share": wins / np.maximum(games, 1),
        "gamma_mean": gamma.mean(axis=0),
        "p_white_equal_ratings_mean": probability.mean(axis=0),
        "p_white_equal_ratings_q05": low,
        "p_white_equal_ratings_q95": high,
        "prob_above_average": np.mean(gamma > draws["mu"].reshape(-1, 1), axis=0),
        "prob_best_first_move": best,
    }).sort_values("games", ascending=False)


def reply_table(data, codes, draws):
    theta = draws["theta"].reshape(-1, data.n_openings)
    gamma = draws["gamma"].reshape(-1, len(data.family_names))
    games = data.games_per_opening
    probability = logistic(theta)
    low, high = np.percentile(probability, [5, 95], axis=0)
    table = pd.DataFrame({
        "first_move": [code.split()[0] for code in codes],
        "reply": [code.split()[1] for code in codes],
        "games": games,
        "white_win_share": np.bincount(data.opening, data.y, data.n_openings) / games,
        "theta_mean": theta.mean(axis=0),
        "p_white_equal_ratings_mean": probability.mean(axis=0),
        "p_white_equal_ratings_q05": low,
        "p_white_equal_ratings_q95": high,
        # Does this reply leave White less than the average of its first move?
        "prob_below_first_move": np.mean(theta < gamma[:, data.family], axis=0),
        "compared_with": "",
        "prob_below_compared": np.nan,
    })
    for move, rows in table.groupby("first_move"):
        if rows["games"].sum() < MIN_GAMES_CONTRAST:
            continue
        reference = rows["games"].idxmax()
        table.loc[rows.index, "compared_with"] = f"{move} {table.loc[reference, 'reply']}"
        table.loc[rows.index, "prob_below_compared"] = np.mean(theta[:, rows.index] < theta[:, [reference]], axis=0)
    return table.sort_values(["first_move", "games"], ascending=[True, False])


def diagnostics_table(draws, acceptance):
    shared = ["mu", "sigma_move", "sigma_reply"]
    series = [draws[name] for name in shared] + [draws["beta"][..., k] for k in range(draws["beta"].shape[-1])]
    gamma, theta = draws["gamma"], draws["theta"]
    rows = {
        "acceptance rate of beta (rated, unrated)": np.round(acceptance["beta"], 3).tolist(),
        "median acceptance rate of theta": np.median(acceptance["theta"]),
        "largest split-Rhat, shared parameters": max(split_rhat(s) for s in series),
        "smallest ESS, shared parameters": min(effective_sample_size(s) for s in series),
        "largest split-Rhat, first moves": max(split_rhat(gamma[..., m]) for m in range(gamma.shape[-1])),
        "smallest ESS, first moves": min(effective_sample_size(gamma[..., m]) for m in range(gamma.shape[-1])),
        "largest split-Rhat, pairs": max(split_rhat(theta[..., c]) for c in range(theta.shape[-1])),
        "smallest ESS, pairs": min(effective_sample_size(theta[..., c]) for c in range(theta.shape[-1])),
    }
    return pd.DataFrame({"statistic": list(rows), "value": list(rows.values())})


def main():
    data, codes = load_games(openings="moves", slope_by="rated")
    draws, acceptance, _ = fit(data, Priors(), SEED, n_burn=N_BURN, n_keep=N_KEEP, n_pilot=3000, parallel=True,
                               run=sampler_moves.run_chain)

    RESULTS.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    np.savez_compressed(RESULTS / "draws_moves.npz", beta=draws["beta"], mu=draws["mu"],
                        sigma_move=draws["sigma_move"], sigma_reply=draws["sigma_reply"],
                        gamma=draws["gamma"][:, ::THETA_THIN].astype(np.float32),
                        theta=draws["theta"][:, ::THETA_THIN].astype(np.float32), codes=codes,
                        family_names=np.array(data.family_names), slope_names=np.array(data.slope_names))

    tables = {"posterior_moves_shared": shared_table(data, draws),
              "posterior_first_moves": first_move_table(data, draws),
              "posterior_replies": reply_table(data, codes, draws),
              "sampler_diagnostics_moves": diagnostics_table(draws, acceptance)}
    for name, table in tables.items():
        table.to_csv(TABLES / f"{name}.csv", index=False)

    print(tables["posterior_moves_shared"].round(4).to_string(index=False))
    print()
    print(tables["sampler_diagnostics_moves"].to_string(index=False))
    print()
    print(tables["posterior_first_moves"].head(10).round(3).to_string(index=False))
    print()
    replies = tables["posterior_replies"]
    print(replies[(replies["games"] >= 30) & replies["first_move"].isin(["e4", "d4"])].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
