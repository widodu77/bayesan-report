"""Figure: the model as a Bayesian network, with plates for the openings and the games.

Observed nodes are shaded and unknown nodes are white, as in the course.

Input:  data/processed/games_clean.csv (only for the numbers of games and openings)
Output: figures/model.pdf, .png and .tex
"""
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from plotting import INK, MUTED, plt, save
from sampler import Priors, load_games

RADIUS = 0.42
OBSERVED = "#c9c8c2"
NODES = {  # name: (x, y, label, observed)
    "mu": (1.6, 4.6, r"$\mu$", False),
    "tau": (3.4, 4.6, r"$\tau$", False),
    "theta": (2.5, 2.6, r"$\theta_o$", False),
    "beta": (6.6, 4.6, r"$\beta$", False),
    "y": (6.6, 1.0, r"$y_i$", True),
    "delta": (8.6, 1.0, r"$\Delta_i$", True),
}
EDGES = [("mu", "theta"), ("tau", "theta"), ("theta", "y"), ("beta", "y"), ("delta", "y")]


def plate(ax, x0, y0, x1, y1, text):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.15",
                                facecolor="none", edgecolor=MUTED, linewidth=1))
    ax.text(x1 - 0.12, y0 + 0.12, text, ha="right", va="bottom", fontsize=8, color=MUTED)


def main():
    data, _ = load_games()
    priors = Priors()
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.set_xlim(0.4, 9.8)
    ax.set_ylim(-0.1, 5.4)
    ax.set_aspect("equal")
    ax.axis("off")

    plate(ax, 1.4, 1.6, 3.6, 3.5, f"$o = 1, \\dots, {data.n_openings}$")
    plate(ax, 5.6, 0.0, 9.6, 1.95, f"$i = 1, \\dots, {len(data.y)}$")

    for x, y, label, observed in NODES.values():
        ax.add_patch(Circle((x, y), RADIUS, facecolor=OBSERVED if observed else "white", edgecolor=INK,
                            linewidth=1.2, zorder=2))
        ax.text(x, y, label, ha="center", va="center", fontsize=12, color=INK, zorder=3)
    for start, end in EDGES:
        (x0, y0), (x1, y1) = NODES[start][:2], NODES[end][:2]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color=INK,
                                     linewidth=1, shrinkA=RADIUS * 72 * 0.49, shrinkB=RADIUS * 72 * 0.49,
                                     zorder=1))
    ax.text(4.75, 1.95, r"$o = o(i)$", ha="center", va="center", fontsize=8, color=MUTED, rotation=-28)

    caption = (
        "The model as a Bayesian network. Shaded nodes are observed and white nodes are unknown; plates repeat "
        f"their content for each of the {data.n_openings} openings and the {len(data.y):,} decisive games. "
        "Game $i$ is played in opening $o(i)$; its result is $y_i = 1$ if White wins and $0$ if Black wins, "
        "and $\\Delta_i$ is the rating of White minus the rating of Black, divided by 400. "
        "$y_i \\sim \\text{Bernoulli}(\\operatorname{logit}^{-1}(\\theta_{o(i)} + \\beta \\Delta_i))$, "
        "$\\theta_o \\sim \\mathcal{N}(\\mu, 1/\\tau)$, "
        f"$\\mu \\sim \\mathcal{{N}}({priors.mu_mean:g}, {priors.mu_sd:g}^2)$, "
        f"$\\beta \\sim \\mathcal{{N}}(\\log 10, {priors.beta_sd:g}^2)$ and "
        f"$\\tau \\sim \\text{{Gamma}}({priors.tau_shape:g}, {priors.tau_rate:g})$ (shape and rate); we report $\\sigma = 1/\\sqrt{{\\tau}}$."
    )
    save(fig, "model", caption)


if __name__ == "__main__":
    main()
