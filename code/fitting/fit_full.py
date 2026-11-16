"""Fit the final model (sampler_full.py) to the real games: White's first move and Black's reply,
one rating slope for rated and one for unrated games, and one effect per player.

Output: results/draws_full.npz, tables/posterior_full_shared.csv, tables/posterior_full_first_moves.csv,
tables/posterior_full_replies.csv, tables/posterior_full_players.csv, tables/sampler_diagnostics_full.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from core import sampler_full
from core.diagnostics import effective_sample_size, split_rhat, summarize
from core.sampler import PROCESSED, Priors, load_games
from fitting.fit_real import N_BURN, N_KEEP, THETA_THIN, logistic

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
RESULTS = ROOT / "results"
SEED = 20261119
U_EVERY = 40  # kept draws of the 14,832 player effects: one every 40 iterations
MIN_GAMES_PLAYER = 10  # players whose effects get convergence diagnostics
SHARED = ("mu", "sigma_move", "sigma_reply", "sigma_player")
MIN_GAMES_CONTRAST = 300  # first moves whose replies are compared with the most played reply


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


def player_names():
    """Player ids in the order of the player indices of load_games (decisive games, sorted ids)."""
    games = pd.read_csv(PROCESSED).query("winner != 'draw'")
    return np.array(sorted(set(games["white_id"]) | set(games["black_id"])))


def shared_table(data, draws):
    beta, gamma = draws["beta"], draws["gamma"]
    rated = beta[..., 0]
    e4, d4 = data.family_names.index("e4"), data.family_names.index("d4")
    quantities = {f"beta, {name}": beta[..., k] for k, name in enumerate(data.slope_names)}
    quantities.update({
        "beta, rated minus unrated": beta[..., 0] - beta[..., 1],
        "mu (average over the 20 first moves)": draws["mu"],
        "sigma_move (spread between White's first moves)": draws["sigma_move"],
        "sigma_reply (spread between Black's replies to one first move)": draws["sigma_reply"],
        "sigma_player (spread between players at equal rating)": draws["sigma_player"],
        "sigma_move minus sigma_reply": draws["sigma_move"] - draws["sigma_reply"],
        "sigma_player in rating points, rated games (400 sigma_player / beta)": 400 * draws["sigma_player"] / rated,
        "white advantage in rating points after 1.e4, rated games (400 gamma / beta)": 400 * gamma[..., e4] / rated,
        "white advantage in rating points after 1.d4, rated games (400 gamma / beta)": 400 * gamma[..., d4] / rated,
        "P(white wins) at equal ratings after 1.e4": logistic(gamma[..., e4]),
        "P(white wins) at equal ratings after 1.d4": logistic(gamma[..., d4]),
        "slope of rated games relative to the Elo formula (beta / log 10)": rated / np.log(10),
    })
    table = pd.DataFrame([{"quantity": name, **summarize(q)} for name, q in quantities.items()])
    table["prob_positive"] = [np.mean(q > 0) for q in quantities.values()]
    return table


def player_table(data, players):
    games = sampler_full.games_per_player(data)
    draws = players["draws"].reshape(-1, data.n_players)
    low, high = np.percentile(draws, [5, 95], axis=0)
    return pd.DataFrame({"player": player_names(), "games": games, "u_mean": players["mean"], "u_sd": players["sd"],
                         "u_q05": low, "u_q95": high,
                         "prob_stronger_than_rating": np.mean(draws > 0, axis=0)}).sort_values("games", ascending=False)


def diagnostics_table(data, draws, players, acceptance):
    series = [draws[name] for name in SHARED] + [draws["beta"][..., k] for k in range(data.n_slopes)]
    active = np.flatnonzero(sampler_full.games_per_player(data) >= MIN_GAMES_PLAYER)
    u = players["draws"]  # thinned, so its ESS counts thinned draws
    rows = {
        "acceptance rate of beta (rated, unrated)": np.round(acceptance["beta"], 3).tolist(),
        "median acceptance rate of theta": np.median(acceptance["theta"]),
        "median acceptance rate of the player effects": np.median(acceptance["u"]),
        "lowest and highest acceptance rate of the player effects":
            np.round([acceptance["u"].min(), acceptance["u"].max()], 3).tolist(),
        "largest split-Rhat, shared parameters": max(split_rhat(s) for s in series),
        "smallest ESS, shared parameters": min(effective_sample_size(s) for s in series),
        "largest split-Rhat, first moves": max(split_rhat(draws["gamma"][..., m]) for m in
                                              range(draws["gamma"].shape[-1])),
        "smallest ESS, first moves": min(effective_sample_size(draws["gamma"][..., m]) for m in
                                         range(draws["gamma"].shape[-1])),
        "largest split-Rhat, pairs": max(split_rhat(draws["theta"][..., c]) for c in range(data.n_openings)),
        "smallest ESS, pairs": min(effective_sample_size(draws["theta"][..., c]) for c in range(data.n_openings)),
        f"largest split-Rhat, players with at least {MIN_GAMES_PLAYER} games (draws kept every {U_EVERY})":
            max(split_rhat(u[..., p]) for p in active),
        f"smallest ESS, players with at least {MIN_GAMES_PLAYER} games (draws kept every {U_EVERY})":
            min(effective_sample_size(u[..., p]) for p in active),
    }
    return pd.DataFrame({"statistic": list(rows), "value": list(rows.values())})


def main():
    data, codes = load_games(openings="moves", slope_by="rated")
    chains, players, acceptance = sampler_full.fit(data, Priors(), SEED, n_burn=N_BURN, n_keep=N_KEEP,
                                                   n_pilot=3000, parallel=True, u_every=U_EVERY, keep_burn_in=True)
    draws = {name: chain[:, N_BURN:] for name, chain in chains.items()}

    RESULTS.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    # The burn-in of the shared parameters is kept only for the trace plots.
    np.savez_compressed(
        RESULTS / "draws_full.npz", beta=draws["beta"], **{name: draws[name] for name in SHARED},
        gamma=draws["gamma"][:, ::THETA_THIN].astype(np.float32),
        theta=draws["theta"][:, ::THETA_THIN].astype(np.float32), u=players["draws"],
        u_mean=players["mean"], u_sd=players["sd"], codes=codes, family_names=np.array(data.family_names),
        slope_names=np.array(data.slope_names), u_every=U_EVERY, theta_thin=THETA_THIN,
        **{f"burn_in_{name}": chains[name][:, :N_BURN] for name in (*SHARED, "beta")})

    tables = {"posterior_full_shared": shared_table(data, draws),
              "posterior_full_first_moves": first_move_table(data, draws),
              "posterior_full_replies": reply_table(data, codes, draws),
              "posterior_full_players": player_table(data, players),
              "sampler_diagnostics_full": diagnostics_table(data, draws, players, acceptance)}
    for name, table in tables.items():
        table.to_csv(TABLES / f"{name}.csv", index=False)

    print(tables["posterior_full_shared"].round(4).to_string(index=False))
    print()
    print(tables["sampler_diagnostics_full"].to_string(index=False))
    print()
    print(tables["posterior_full_first_moves"].head(10).round(3).to_string(index=False))
    print()
    replies = tables["posterior_full_replies"]
    print(replies[(replies["games"] >= 100) & replies["first_move"].isin(["e4", "d4"])].round(3).to_string(index=False))
    print()
    print(tables["posterior_full_players"].head(8).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
