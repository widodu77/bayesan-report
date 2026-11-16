"""Figure: White's chance of winning by first move, and by Black's reply to 1.e4.

Input:  tables/posterior_first_moves.csv, tables/posterior_replies.csv, tables/posterior_moves_shared.csv
        (from fit_moves.py)
Output: figures/moves.pdf, .png and .tex
"""
from pathlib import Path

import numpy as np
import pandas as pd

from plotting import COLORS, INK, MUTED, TEXT_WIDTH, plt, save

TABLES = Path(__file__).resolve().parent / "tables"
MIN_GAMES_FIRST_MOVE = 90
MIN_GAMES_REPLY = 140


def interval_panel(ax, rows, labels, reference, reference_label, title, highlight=()):
    rows = rows.sort_values("p_white_equal_ratings_mean")
    y = np.arange(len(rows))
    colors = [COLORS[1] if label in highlight else COLORS[0] for label in rows[labels]]
    ax.axvline(50, color=MUTED, linewidth=0.6)
    if reference is not None:
        ax.axvline(100 * reference, color=MUTED, linewidth=0.8, linestyle="--")
        ax.text(100 * reference + 0.2, -0.55, reference_label, ha="left", va="bottom", fontsize=7, color=MUTED)
    ax.hlines(y, 100 * rows["p_white_equal_ratings_q05"], 100 * rows["p_white_equal_ratings_q95"], colors=colors,
              linewidth=2.5, alpha=0.7)
    ax.scatter(100 * rows["p_white_equal_ratings_mean"], y, s=14, color=INK, zorder=3)
    ax.set_yticks(y, [f"{label}  ({games:,})" for label, games in zip(rows[labels], rows["games"], strict=True)])
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("P(White wins) at equal ratings (%)")
    ax.set_title(title, loc="left")


def main():
    first = pd.read_csv(TABLES / "posterior_first_moves.csv")
    replies = pd.read_csv(TABLES / "posterior_replies.csv")
    shared = pd.read_csv(TABLES / "posterior_moves_shared.csv").set_index("quantity")

    first = first[first["games"] >= MIN_GAMES_FIRST_MOVE].copy()
    first["label"] = "1." + first["first_move"]
    e4 = replies[(replies["first_move"] == "e4") & (replies["games"] >= MIN_GAMES_REPLY)].copy()
    e4["label"] = "1.e4 " + e4["reply"]
    e4_average = first.set_index("first_move").loc["e4"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.0), layout="constrained")
    interval_panel(left, first, "label", None, "", "(a) White's first move", highlight=("1.e3",))
    interval_panel(right, e4, "label", e4_average["p_white_equal_ratings_mean"], "1.e4 on average",
                   "(b) Black's reply to 1.e4", highlight=("1.e4 c5",))

    sicilian = e4.set_index("reply").loc["c5"]
    open_game = e4.set_index("reply").loc["e5"]
    e3 = first.set_index("first_move").loc["e3"]
    c4 = first.set_index("first_move").loc["c4"]
    spread_move = shared.loc["sigma_move (spread between White's first moves)"]
    spread_reply = shared.loc["sigma_reply (spread between Black's replies to one first move)"]
    difference = shared.loc["sigma_move minus sigma_reply"]
    caption = (
        "White's chance of winning a decisive game against an opponent of the same rating, from the model of "
        "White's first move and Black's reply, each chosen by one player alone. Dots are posterior means and "
        "bars 90\\% intervals, in orange for the two moves discussed below; the number of decisive games is in "
        "parentheses. "
        f"(a): first moves with at least {MIN_GAMES_FIRST_MOVE} games. 1.e3 is clearly worse for White "
        f"({100 * e3['p_white_equal_ratings_mean']:.1f}\\%, {100 * e3['p_white_equal_ratings_q05']:.1f} to "
        f"{100 * e3['p_white_equal_ratings_q95']:.1f}\\%), and 1.c4 the best of the common moves "
        f"({100 * c4['p_white_equal_ratings_mean']:.1f}\\%). "
        f"(b): replies to 1.e4 with at least {MIN_GAMES_REPLY} games; the dashed line is 1.e4 on average. "
        f"After the Sicilian, 1...c5, White wins {100 * sicilian['p_white_equal_ratings_mean']:.1f}\\% "
        f"({100 * sicilian['p_white_equal_ratings_q05']:.1f} to {100 * sicilian['p_white_equal_ratings_q95']:.1f}\\%), "
        f"against {100 * open_game['p_white_equal_ratings_mean']:.1f}\\% after 1...e5; the posterior "
        f"probability that the Sicilian is better for Black than 1...e5 is "
        f"{100 * sicilian['prob_below_compared']:.1f}\\%. "
        f"Overall, White's first move matters more than Black's reply: the spread between first moves, "
        f"$\\sigma_{{\\text{{move}}}} = {spread_move['mean']:.2f}$, exceeds the spread between replies to one "
        f"first move, $\\sigma_{{\\text{{reply}}}} = {spread_reply['mean']:.2f}$, with posterior probability "
        f"{difference['prob_positive']:.2f}."
    )
    save(fig, "moves", caption)


if __name__ == "__main__":
    main()
