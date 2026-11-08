"""Clean the raw Lichess games and write the analysis data set.

Input:  data/raw/games.csv
Output: data/processed/games_clean.csv, tables/data_cleaning.csv, tables/data_summary.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "games.csv"
PROCESSED = ROOT / "data" / "processed" / "games_clean.csv"
TABLES = ROOT / "tables"

MIN_PLIES = 10  # under five moves per side the game was abandoned, not played
MIN_GAMES_WELL_OBSERVED = 30

COLUMNS = ["id", "rated", "turns", "victory_status", "winner", "white_id", "white_rating",
           "black_id", "black_rating", "delta", "opening_eco", "opening_name"]


def clean(raw):
    """Apply the cleaning steps in order and return the games and a log of each step."""
    steps = [("raw file", raw)]
    steps.append(("one row per game id", steps[-1][1].drop_duplicates("id")))
    steps.append((f"at least {MIN_PLIES} plies", steps[-1][1].query("turns >= @MIN_PLIES")))
    steps.append(("decisive result (no draw)", steps[-1][1].query("winner != 'draw'")))

    log = pd.DataFrame({"step": [name for name, _ in steps],
                        "games": [len(games) for _, games in steps]})
    log["removed"] = (-log["games"].diff()).fillna(0).astype(int)
    # Draws stay in the output file for the sensitivity analysis that scores them as half a win.
    return steps[-2][1].copy(), log


def summarize(games):
    """Descriptive statistics of the games with a decisive result."""
    decisive = games[games["winner"] != "draw"]
    per_opening = decisive["opening_eco"].value_counts()
    well_observed = per_opening[per_opening >= MIN_GAMES_WELL_OBSERVED]
    players = pd.concat([decisive["white_id"], decisive["black_id"]])
    default_rating = (decisive["white_rating"] == 1500) | (decisive["black_rating"] == 1500)
    rows = {
        "games, all results": len(games),
        "white wins, share of all results": (games["winner"] == "white").mean(),
        "black wins, share of all results": (games["winner"] == "black").mean(),
        "draws, share of all results": (games["winner"] == "draw").mean(),
        "games, decisive": len(decisive),
        "white wins, share of decisive": (decisive["winner"] == "white").mean(),
        "openings": per_opening.size,
        "median games per opening": per_opening.median(),
        "openings with one game": int((per_opening == 1).sum()),
        f"openings with at least {MIN_GAMES_WELL_OBSERVED} games": well_observed.size,
        f"games in openings with at least {MIN_GAMES_WELL_OBSERVED} games": int(well_observed.sum()),
        "distinct players": players.nunique(),
        "share of games with a player rated exactly 1500": default_rating.mean(),
        "share of games that are rated": decisive["rated"].mean(),
        "standard deviation of delta": decisive["delta"].std(),
    }
    return pd.DataFrame({"statistic": list(rows), "value": list(rows.values())})


def main():
    raw = pd.read_csv(RAW)
    games, log = clean(raw)
    games["delta"] = (games["white_rating"] - games["black_rating"]) / 400

    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    games[COLUMNS].to_csv(PROCESSED, index=False)
    log.to_csv(TABLES / "data_cleaning.csv", index=False)
    summary = summarize(games)
    summary.to_csv(TABLES / "data_summary.csv", index=False)

    print(log.to_string(index=False))
    print()
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
