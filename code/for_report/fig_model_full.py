"""Figure: the final model as a Bayesian network, with plates for the first moves, the pairs
(first move, reply), the players, the two groups of games for the rating slope, and the games.

Observed nodes are shaded and unknown nodes are white, as in the course.

Input:  data/processed/games_clean.csv (only for the numbers of games, pairs and players)
Output: figures/model_full.pdf, .png and .tex
"""
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from core.plotting import INK, MUTED, TEXT_WIDTH, plt, save
from core.sampler import Priors, load_games

RADIUS = 0.42
OBSERVED = "#c9c8c2"

NODES = {  # name: (x, y, label, observed)
    "mu": (0.9, 6.0, r"$\mu$", False),
    "tau_move": (2.5, 6.0, r"$\tau_{\mathrm{m}}$", False),
    "gamma": (1.5, 4.0, r"$\gamma_m$", False),
    "tau_reply": (4.3, 6.0, r"$\tau_{\mathrm{r}}$", False),
    "theta": (3.8, 3.0, r"$\theta_c$", False),
    "tau_player": (7.7, 6.0, r"$\tau_{\mathrm{p}}$", False),
    "u": (7.7, 4.35, r"$u_p$", False),
    "beta": (10.5, 4.35, r"$\beta_g$", False),
    "y": (8.2, 1.1, r"$y_i$", True),
    "delta": (10.5, 1.1, r"$\Delta_i$", True),
}
EDGES = [("mu", "gamma"), ("tau_move", "gamma"), ("gamma", "theta"), ("tau_reply", "theta"), ("theta", "y"),
         ("tau_player", "u"), ("u", "y"), ("beta", "y"), ("delta", "y")]
EDGE_LABELS = [  # (x, y, text, horizontal alignment), set beside the arrows they name
    (5.75, 2.45, r"$c = c(i)$", "center"),
    (7.6, 3.0, r"$p = w(i), b(i)$", "right"),
    (9.75, 2.95, r"$g = g(i)$", "left"),
]


def plate(ax, x0, y0, x1, y1, text, side="right"):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.15",
                                facecolor="none", edgecolor=MUTED, linewidth=1))
    x = x1 - 0.12 if side == "right" else x0 + 0.12
    ax.text(x, y0 + 0.1, text, ha=side, va="bottom", fontsize=8, color=MUTED)


def main():
    data, _ = load_games(openings="moves", slope_by="rated")
    priors = Priors()
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 3.6))
    ax.set_xlim(0.1, 12.0)
    ax.set_ylim(-0.15, 6.6)
    ax.set_aspect("equal")
    ax.axis("off")

    plate(ax, 0.4, 1.7, 5.2, 5.0, f"$m = 1, \\dots, {len(data.family_names)}$", side="left")
    plate(ax, 2.6, 2.05, 5.0, 3.9, f"$c = 1, \\dots, {data.n_openings}$")
    plate(ax, 6.6, 3.3, 8.9, 5.1, f"$p = 1, \\dots, {data.n_players:,}$")
    plate(ax, 9.4, 3.3, 11.8, 5.1, "$g = 1, 2$")
    plate(ax, 6.2, 0.0, 11.8, 2.25, f"$i = 1, \\dots, {len(data.y):,}$")

    for x, y, label, observed in NODES.values():
        ax.add_patch(Circle((x, y), RADIUS, facecolor=OBSERVED if observed else "white", edgecolor=INK,
                            linewidth=1.2, zorder=2))
        ax.text(x, y, label, ha="center", va="center", fontsize=11, color=INK, zorder=3)
    for start, end in EDGES:
        (x0, y0), (x1, y1) = NODES[start][:2], NODES[end][:2]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, color=INK,
                                     linewidth=1, shrinkA=RADIUS * 72 * 0.42, shrinkB=RADIUS * 72 * 0.42,
                                     zorder=1))
    for x, y, text, side in EDGE_LABELS:
        ax.text(x, y, text, ha=side, va="center", fontsize=7, color=MUTED)

    caption = (
        "The final model as a Bayesian network. Shaded nodes are observed and white nodes are unknown; plates "
        f"repeat their content for each of the {len(data.family_names)} first moves $m$ of White, the "
        f"{data.n_openings} pairs $c$ of a first move and Black's reply (nested in their first move), the "
        f"{data.n_players:,} players $p$, the two groups $g$ of rated and unrated games, and the "
        f"{len(data.y):,} decisive games $i$. Game $i$ is played by White $w(i)$ against Black $b(i)$ in pair "
        "$c(i)$, in group $g(i)$; $y_i = 1$ if White wins and $0$ if Black wins, and $\\Delta_i$ is the rating "
        "of White minus the rating of Black, divided by 400. "
        "$y_i \\sim \\text{Bernoulli}(\\operatorname{logit}^{-1}(\\theta_{c(i)} + \\beta_{g(i)} \\Delta_i "
        "+ u_{w(i)} - u_{b(i)}))$, "
        "$\\theta_c \\sim \\mathcal{N}(\\gamma_{m(c)}, 1/\\tau_{\\mathrm{r}})$, "
        "$\\gamma_m \\sim \\mathcal{N}(\\mu, 1/\\tau_{\\mathrm{m}})$, "
        "$u_p \\sim \\mathcal{N}(0, 1/\\tau_{\\mathrm{p}})$, "
        f"$\\mu \\sim \\mathcal{{N}}({priors.mu_mean:g}, {priors.mu_sd:g}^2)$, "
        f"$\\beta_g \\sim \\mathcal{{N}}(\\log 10, {priors.beta_sd:g}^2)$ and "
        f"$\\tau_{{\\mathrm{{m}}}}, \\tau_{{\\mathrm{{r}}}}, \\tau_{{\\mathrm{{p}}}} \\sim "
        f"\\text{{Gamma}}({priors.tau_shape:g}, {priors.tau_rate:g})$ (shape and rate); we report the spreads "
        "$\\sigma = 1/\\sqrt{\\tau}$."
    )
    save(fig, "model_full", caption)


if __name__ == "__main__":
    main()
