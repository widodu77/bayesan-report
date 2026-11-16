"""Write every number and table of the report from the result tables, so that the report quotes
only numbers produced by the code.

Input:  tables/*.csv (from the analysis scripts), results/draws_full.npz (from fit_full.py)
Output: ../report/generated/numbers.tex (one LaTeX macro per number) and ../report/generated/table_*.tex
"""
from pathlib import Path

import numpy as np
import pandas as pd

from validate_full import N_REPEATS

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"
RESULTS = ROOT / "results"
OUT = ROOT.parent / "report" / "generated"


def read(name, index=None):
    table = pd.read_csv(TABLES / f"{name}.csv")
    return table.set_index(index) if index else table


def num(x, digits=2):
    """A number with thousands separators that LaTeX typesets correctly."""
    return f"{x:,.{digits}f}".replace(",", "{,}").replace("-", "$-$", 1) if x < 0 else \
        f"{x:,.{digits}f}".replace(",", "{,}")


def pct(x, digits=1):
    return num(100 * x, digits)


def logistic(x):
    return 1 / (1 + np.exp(-x))


class Macros:
    def __init__(self):
        self.lines = []

    def add(self, name, value):
        if not name.isalpha():
            raise ValueError(f"LaTeX macro names must be letters only: {name}")
        self.lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    def interval(self, name, row, digits=2, scale=1.0):
        """Mean, and the two ends of the 90% interval, as three macros."""
        for suffix, column in (("", "mean"), ("Low", "q05"), ("High", "q95")):
            self.add(name + suffix, num(scale * row[column], digits))


def data_numbers(m):
    cleaning = read("data_cleaning", "step")
    summary = read("data_summary", "statistic")["value"]
    m.add("nRaw", num(cleaning.loc["raw file", "games"], 0))
    m.add("nDuplicates", num(cleaning.loc["one row per game id", "removed"], 0))
    m.add("nShort", num(cleaning.loc["at least 10 plies", "removed"], 0))
    m.add("nDraws", num(cleaning.loc["decisive result (no draw)", "removed"], 0))
    m.add("nDecisive", num(cleaning.loc["decisive result (no draw)", "games"], 0))
    m.add("shareDraws", pct(summary["draws, share of all results"]))
    m.add("shareWhiteDecisive", pct(summary["white wins, share of decisive"]))
    m.add("nOpenings", num(summary["openings"], 0))
    m.add("medianGamesOpening", num(summary["median games per opening"], 0))
    m.add("nOpeningsOne", num(summary["openings with one game"], 0))
    m.add("nPlayers", num(summary["distinct players"], 0))
    m.add("shareRated", pct(summary["share of games that are rated"]))
    m.add("shareFifteen", pct(summary["share of games with a player rated exactly 1500"]))

    players = read("posterior_full_players")
    active = players[players["games"] >= 10]
    m.add("nActivePlayers", num(len(active), 0))
    replies = read("posterior_full_replies")
    m.add("nPairs", num(len(replies), 0))
    m.add("nFirstMoves", num(replies["first_move"].nunique(), 0))


def first_model_numbers(m):
    shared = read("posterior_shared", "quantity")
    m.interval("betaEco", shared.loc["beta"])
    m.interval("muEco", shared.loc["mu"], 3)
    m.interval("sigmaEco", shared.loc["sigma"])
    m.interval("advantageEco", shared.loc["white advantage in rating points (400 mu / beta)"], 1)
    m.interval("pWhiteEco", shared.loc["P(white wins) at equal ratings, average opening"], 1, 100)

    check = read("posterior_predictive", "statistic")
    chi2 = check.loc["chi-square discrepancy over the bins of rating gap"]
    m.add("pShapeEco", num(chi2["p_value"], 3))
    gap = check[check.index.str.startswith("white win share, rating gap")]
    worst = gap["p_value_two_sided"].idxmin()
    m.add("worstGapBin", worst.removeprefix("white win share, rating gap ").replace("-", "$-$"))
    m.add("worstGapObserved", pct(gap.loc[worst, "observed"]))
    m.add("worstGapReplicated", pct(gap.loc[worst, "replicated_mean"]))

    grid = read("check_quadrature")
    m.add("gridMaxZ", num(grid["z"].abs().max(), 2))
    importance = read("check_importance_sampling")
    m.add("isWeightEss", num(importance["is_weight_ess"].iloc[0], 0))
    m.add("isDraws", num(importance["is_draws"].iloc[0], 0))
    m.add("isMaxZ", num(importance[["z_mean", "z_sd"]].abs().to_numpy().max(), 2))
    pymc = read("crosscheck_pymc")
    m.add("pymcMaxZ", num(pymc["z_mean"].abs().max(), 2))
    m.add("pymcThetaGap", pct(pymc["value"].dropna().iloc[0], 1))


def rating_slope_numbers(m):
    shared = read("posterior_rating_slopes", "quantity")
    m.interval("betaRatedSlopes", shared.loc["beta, rated games"])
    m.interval("betaUnratedSlopes", shared.loc["beta, unrated games"])
    checks = read("posterior_predictive_rating_slopes")
    for model, tag in (("one slope", "One"), ("rated and unrated slopes", "Two")):
        for subset, name in (("all games", "All"), ("rated games", "Rated"), ("unrated games", "Unrated")):
            row = checks[(checks["model"] == model)
                         & (checks["statistic"] == f"chi-square over deciles of rating gap, {subset}")].iloc[0]
            p = row["p_value"]
            m.add(f"pShape{tag}{name}", "<0.001" if p < 0.001 else num(p, 3))


def final_numbers(m):
    shared = read("posterior_full_shared", "quantity")
    m.interval("betaRated", shared.loc["beta, rated games"])
    m.interval("betaUnrated", shared.loc["beta, unrated games"])
    m.interval("betaDiff", shared.loc["beta, rated minus unrated"])
    m.interval("sigmaMove", shared.loc["sigma_move (spread between White's first moves)"])
    m.interval("sigmaReply", shared.loc["sigma_reply (spread between Black's replies to one first move)"])
    m.interval("sigmaPlayer", shared.loc["sigma_player (spread between players at equal rating)"])
    m.interval("sigmaPlayerPoints",
               shared.loc["sigma_player in rating points, rated games (400 sigma_player / beta)"], 0)
    m.interval("advantageEfour",
               shared.loc["white advantage in rating points after 1.e4, rated games (400 gamma / beta)"], 1)
    m.interval("pWhiteEfour", shared.loc["P(white wins) at equal ratings after 1.e4"], 1, 100)
    m.interval("pWhiteDfour", shared.loc["P(white wins) at equal ratings after 1.d4"], 1, 100)
    m.interval("eloRatio", shared.loc["slope of rated games relative to the Elo formula (beta / log 10)"])
    m.add("probAdvantageEfour", num(shared.loc["white advantage in rating points after 1.e4, rated games "
                                               "(400 gamma / beta)", "prob_positive"], 3))
    m.add("probMoveMatters", num(shared.loc["sigma_move minus sigma_reply", "prob_positive"], 2))

    first = read("posterior_full_first_moves", "first_move")
    for move, name in (("e3", "Ethree"), ("c4", "Cfour")):
        row = first.loc[move]
        m.add(f"pWhite{name}", pct(row["p_white_equal_ratings_mean"]))
        m.add(f"pWhite{name}Low", pct(row["p_white_equal_ratings_q05"]))
        m.add(f"pWhite{name}High", pct(row["p_white_equal_ratings_q95"]))
        m.add(f"games{name}", num(row["games"], 0))
    m.add("probEthreeAbove", num(first.loc["e3", "prob_above_average"], 3))

    replies = read("posterior_full_replies")
    e4 = replies[replies["first_move"] == "e4"].set_index("reply")
    for reply, name in (("c5", "Sicilian"), ("e5", "OpenGame"), ("e6", "French")):
        row = e4.loc[reply]
        m.add(f"pWhite{name}", pct(row["p_white_equal_ratings_mean"]))
        m.add(f"pWhite{name}Low", pct(row["p_white_equal_ratings_q05"]))
        m.add(f"pWhite{name}High", pct(row["p_white_equal_ratings_q95"]))
        m.add(f"games{name}", num(row["games"], 0))
    m.add("probSicilian", num(e4.loc["c5", "prob_below_compared"], 3))
    m.add("probFrench", num(e4.loc["e6", "prob_below_compared"], 2))

    # Chance that the higher-rated player wins a rated game after 1.e4, averaged over the two colours.
    draws = np.load(RESULTS / "draws_full.npz", allow_pickle=True)
    beta = draws["beta"][:, ::int(draws["theta_thin"]), 0].ravel()
    gamma = draws["gamma"][..., list(draws["family_names"]).index("e4")].ravel().astype(float)
    for gap in (100, 200):
        x = gap / 400
        score = 0.5 * (logistic(gamma + beta * x) + logistic(beta * x - gamma))
        tag = "Hundred" if gap == 100 else "TwoHundred"
        m.add(f"favourite{tag}", pct(score.mean()))
        m.add(f"favourite{tag}Low", pct(np.percentile(score, 5)))
        m.add(f"favourite{tag}High", pct(np.percentile(score, 95)))
        m.add(f"elo{tag}", pct(1 / (1 + 10 ** (-gap / 400))))

    diagnostics = read("sampler_diagnostics_full", "statistic")["value"]
    m.add("rhatFinal", num(max(float(diagnostics[k]) for k in diagnostics.index if "Rhat" in k), 3))
    m.add("essFinalShared", num(float(diagnostics["smallest ESS, shared parameters"]), 0))
    m.add("essFinalPairs", num(float(diagnostics["smallest ESS, pairs"]), 0))
    m.add("essFinalPlayers", num(float(diagnostics[[k for k in diagnostics.index
                                                    if k.startswith("smallest ESS, players")][0]]), 0))
    m.add("acceptPlayers", num(float(diagnostics["median acceptance rate of the player effects"]), 2))
    m.add("acceptTheta", num(float(diagnostics["median acceptance rate of theta"]), 2))

    checks = read("posterior_predictive_full")
    for model, tag in (("without player effects", "Without"), ("with player effects", "With")):
        rows = checks[checks["model"] == model].set_index("statistic")
        dispersion = rows[rows.index.str.startswith("player dispersion")].iloc[0]
        m.add(f"dispersion{tag}", num(dispersion["observed"], 2))
        m.add(f"dispersion{tag}Low", num(dispersion["replicated_q05"], 2))
        m.add(f"dispersion{tag}High", num(dispersion["replicated_q95"], 2))
        p = dispersion["p_value"]
        m.add(f"pDispersion{tag}", "<0.001" if p < 0.001 else num(p, 2))
        if tag == "With":
            others = rows[~rows.index.str.startswith("player dispersion")]["p_value"]
            m.add("pChecksFinalLow", num(others.min(), 2))
            m.add("pChecksFinalHigh", num(others.max(), 2))


def validation_numbers(m):
    repeated = read("validation_full_repeated", "quantity")
    for name, tag in (("sigma_reply", "Reply"), ("sigma_move", "Move")):
        m.add(f"biasSigma{tag}", num(repeated.loc[name, "mean_error"], 3))
        m.add(f"biasSigma{tag}Se", num(repeated.loc[name, "se_mean_error"], 3))
    m.add("insideSigmaReply", pct(repeated.loc["sigma_reply", "share_inside_90"], 0))
    m.add("nRepeatsFull", num(N_REPEATS, 0))


def validation_table():
    models = [("First model (ECO openings)", "validation_recovery", "validation_coverage"),
              ("+ rated and unrated slopes", "validation_rating_slopes_recovery", "validation_rating_slopes_coverage"),
              ("First move and reply", "validation_moves_recovery", "validation_moves_coverage"),
              ("Final model (+ player effects)", "validation_full_recovery", "validation_full_coverage")]
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"Model & True values inside & \multicolumn{2}{c}{Coverage over 100 data sets} \\",
             r" & 90\% intervals (one data set) & 50\% intervals & 90\% intervals \\", r"\midrule"]
    for label, recovery_name, coverage_name in models:
        recovery = read(recovery_name)
        recovery = recovery[~recovery["quantity"].str.contains("minus")]
        coverage = read(coverage_name)
        lines.append(f"{label} & {int(recovery['inside_90'].sum())} of {len(recovery)} & "
                     f"{pct(coverage['coverage_50'].min(), 0)}--{pct(coverage['coverage_50'].max(), 0)}\\% & "
                     f"{pct(coverage['coverage_90'].min(), 0)}--{pct(coverage['coverage_90'].max(), 0)}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def cleaning_table():
    cleaning = read("data_cleaning")
    lines = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule", r"Step & Games & Removed \\", r"\midrule"]
    for _, row in cleaning.iterrows():
        removed = "" if row["removed"] == 0 else num(row["removed"], 0)
        lines.append(f"{row['step'].capitalize()} & {num(row['games'], 0)} & {removed} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def final_table():
    shared = read("posterior_full_shared", "quantity")
    first = read("posterior_full_first_moves", "first_move")
    replies = read("posterior_full_replies")
    e4 = replies[replies["first_move"] == "e4"].set_index("reply")
    rows = [
        (r"Slope $\beta_g$, rated games", shared.loc["beta, rated games"], 2, 1),
        (r"Slope $\beta_g$, unrated games", shared.loc["beta, unrated games"], 2, 1),
        (r"Spread between first moves, $\sigma_{\mathrm{m}}$", shared.loc[
            "sigma_move (spread between White's first moves)"], 2, 1),
        (r"Spread between replies, $\sigma_{\mathrm{r}}$", shared.loc[
            "sigma_reply (spread between Black's replies to one first move)"], 2, 1),
        (r"Spread between players, $\sigma_{\mathrm{p}}$", shared.loc[
            "sigma_player (spread between players at equal rating)"], 2, 1),
        (r"\quad in rating points, $400\,\sigma_{\mathrm{p}}/\beta_{\mathrm{rated}}$", shared.loc[
            "sigma_player in rating points, rated games (400 sigma_player / beta)"], 0, 1),
        (r"White's edge after 1.e4, rating points", shared.loc[
            "white advantage in rating points after 1.e4, rated games (400 gamma / beta)"], 1, 1),
    ]
    lines = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule", r"Quantity & Mean & 90\% interval \\", r"\midrule"]
    for label, row, digits, scale in rows:
        lines.append(f"{label} & {num(scale * row['mean'], digits)} & "
                     f"{num(scale * row['q05'], digits)} to {num(scale * row['q95'], digits)} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{3}{@{}l}{$\Pr(\text{White wins})$ at equal ratings, \%} \\")
    for label, row in [("after 1.e4", first.loc["e4"]), ("after 1.d4", first.loc["d4"]),
                       ("after 1.c4", first.loc["c4"]), ("after 1.e3", first.loc["e3"]),
                       ("after 1.e4 e5", e4.loc["e5"]), ("after 1.e4 c5 (Sicilian)", e4.loc["c5"]),
                       ("after 1.e4 e6 (French)", e4.loc["e6"])]:
        lines.append(f"\\quad {label} & {pct(row['p_white_equal_ratings_mean'])} & "
                     f"{pct(row['p_white_equal_ratings_q05'])} to {pct(row['p_white_equal_ratings_q95'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def sensitivity_table():
    table = read("sensitivity_full")
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Variant & Games & $\beta_{\mathrm{rated}}$ & $\sigma_{\mathrm{p}}$ & 1.e4, \% & "
             r"1.e4 c5 / e5, \% & $\Pr$(c5 better) \\", r"\midrule"]
    for _, row in table.iterrows():
        lines.append(f"{row['variant']} & {num(row['games'], 0)} & {num(row['beta_rated_mean'], 2)} & "
                     f"{num(row['sigma_player_mean'], 2)} & {pct(row['p_white_e4_mean'])} & "
                     f"{pct(row['p_white_e4_c5_mean'])} / {pct(row['p_white_e4_e5_mean'])} & "
                     f"{num(row['prob_sicilian_better_for_black'], 3)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def sensitivity_numbers(m):
    table = read("sensitivity_full", "variant")
    m.add("sensSicilianLow", num(table["prob_sicilian_better_for_black"].min(), 2))
    m.add("sensSicilianHigh", num(table["prob_sicilian_better_for_black"].max(), 2))
    m.add("sensEfourLow", pct(table["p_white_e4_mean"].min()))
    m.add("sensEfourHigh", pct(table["p_white_e4_mean"].max()))
    m.add("sensMoveLow", num(table["prob_move_matters_more"].min(), 2))
    m.add("sensMoveHigh", num(table["prob_move_matters_more"].max(), 2))
    m.add("sensEthreeLow", num(table["prob_e3_below_e4"].min(), 3))
    m.add("sensPlayerLow", num(table["sigma_player_mean"].min(), 2))
    m.add("sensPlayerHigh", num(table["sigma_player_mean"].max(), 2))
    m.add("sensPlayerBaseline", num(table.loc["baseline", "sigma_player_mean"], 2))
    m.add("sensPlayerRatedOnly", num(table.loc["rated games only", "sigma_player_mean"], 2))
    m.add("sensPlayerNoFifteen", num(table.loc["no player rated 1500", "sigma_player_mean"], 2))
    m.add("sensRhatMax", num(table["largest_rhat"].max(), 3))
    m.add("nVariants", num(len(table), 0))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = Macros()
    data_numbers(m)
    first_model_numbers(m)
    rating_slope_numbers(m)
    final_numbers(m)
    validation_numbers(m)
    sensitivity_numbers(m)
    (OUT / "numbers.tex").write_text("% Written by code/make_report_numbers.py; do not edit.\n"
                                     + "\n".join(m.lines) + "\n", encoding="utf-8")
    for name, content in (("cleaning", cleaning_table()), ("validation", validation_table()),
                          ("final", final_table()), ("sensitivity", sensitivity_table())):
        (OUT / f"table_{name}.tex").write_text("% Written by code/make_report_numbers.py; do not edit.\n"
                                               + content + "\n", encoding="utf-8")
    print(f"{len(m.lines)} numbers and 4 tables written to {OUT}")


if __name__ == "__main__":
    main()
