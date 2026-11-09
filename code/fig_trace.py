"""Figure: trace plots and autocorrelations of beta, mu and sigma for the four chains.

Input:  results/draws.npz (from fit_real.py)
Output: figures/trace.pdf, .png and .tex
"""
from pathlib import Path

import numpy as np

from diagnostics import autocovariance, effective_sample_size, split_rhat
from plotting import TEXT_WIDTH, plt, save

RESULTS = Path(__file__).resolve().parent / "results"
PARAMETERS = [("beta", r"$\beta$"), ("mu", r"$\mu$"), ("sigma", r"$\sigma$")]
N_START = 300  # first iterations shown, enough to see the chains leave their dispersed starts
TRACE_THIN = 20  # plotting every draw of the kept part would only draw a solid band
MAX_LAG = 150


def autocorrelation(chain):
    covariance = autocovariance(chain)
    return covariance / covariance[0]


def main():
    saved = np.load(RESULTS / "draws.npz", allow_pickle=True)
    n_burn = saved["burn_in_beta"].shape[1]
    n_chains, n_keep = saved["beta"].shape

    fig, axes = plt.subplots(3, 3, figsize=(TEXT_WIDTH, 5.6), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.6, 1]})
    summary = {}
    for row, (name, symbol) in enumerate(PARAMETERS):
        start, kept = saved[f"burn_in_{name}"], saved[name]
        summary[name] = (split_rhat(kept), effective_sample_size(kept))
        iteration = np.arange(n_burn + n_keep)
        for c in range(n_chains):
            axes[row, 0].plot(iteration[:N_START], start[c, :N_START], linewidth=0.9, label=f"chain {c + 1}")
            axes[row, 1].plot(iteration[n_burn::TRACE_THIN], kept[c, ::TRACE_THIN], linewidth=0.4, alpha=0.8)
            axes[row, 2].plot(np.arange(MAX_LAG + 1), autocorrelation(kept[c])[:MAX_LAG + 1], linewidth=0.9)
        axes[row, 0].set_ylabel(symbol, rotation=0, labelpad=10, fontsize=11)
        axes[row, 2].axhline(0, color="#52514e", linewidth=0.6)
        axes[row, 2].set_ylim(-0.1, 1.02)
        rhat, ess = summary[name]
        axes[row, 1].text(0.99, 0.96, f"$\\hat R$ = {rhat:.3f}, ESS = {ess:,.0f}",
                          transform=axes[row, 1].transAxes, ha="right", va="top", fontsize=7,
                          bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})

    axes[0, 0].set_title(f"(a) first {N_START} iterations", loc="left")
    axes[0, 1].set_title(f"(b) after burn-in (every {TRACE_THIN}th draw)", loc="left")
    axes[0, 2].set_title("(c) autocorrelation", loc="left")
    axes[2, 0].set_xlabel("iteration")
    axes[2, 1].set_xlabel("iteration")
    axes[2, 2].set_xlabel("lag")
    for ax in axes[:2].ravel():
        ax.tick_params(labelbottom=False)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncols=n_chains)

    caption = (
        f"Convergence of the sampler on the real data: {n_chains} chains started from points drawn from the "
        f"prior. (a): the first {N_START} iterations, where the chains leave their dispersed starting points "
        f"and meet long before the end of the burn-in of {n_burn:,} iterations. "
        f"(b): the {n_keep:,} iterations kept per chain after burn-in, which overlap, with the split-$\\hat R$ "
        "and the effective sample size (ESS) of the four chains together. "
        "(c): autocorrelation of each chain after burn-in. $\\beta$ mixes fast; $\\mu$ and especially "
        "$\\sigma$ are more autocorrelated, because they move only through the "
        f"{saved['theta'].shape[2]} opening effects, but "
        f"their ESS of {summary['mu'][1]:,.0f} and {summary['sigma'][1]:,.0f} still make the Monte Carlo "
        "errors small compared with the posterior standard deviations."
    )
    save(fig, "trace", caption)


if __name__ == "__main__":
    main()
