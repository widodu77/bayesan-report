"""Figure: trace plots and autocorrelations of the shared parameters of the final model.

Input:  results/draws_full.npz (from fit_full.py)
Output: figures/trace_full.pdf, .png and .tex
"""
from pathlib import Path

import numpy as np

from core.diagnostics import autocovariance, effective_sample_size, split_rhat
from core.plotting import TEXT_WIDTH, plt, save
from core.rounding import fmt

RESULTS = Path(__file__).resolve().parents[1] / "results"
PARAMETERS = [("beta", r"$\beta_{\mathrm{rated}}$"), ("mu", r"$\mu$"), ("sigma_move", r"$\sigma_{\mathrm{m}}$"),
              ("sigma_reply", r"$\sigma_{\mathrm{r}}$"), ("sigma_player", r"$\sigma_{\mathrm{p}}$")]
N_START = 300
TRACE_THIN = 20
MAX_LAG = 300


def series(saved, name):
    """Burn-in and kept draws of one scalar parameter; beta is the rated-games slope."""
    start, kept = saved[f"burn_in_{name}"], saved[name]
    if name == "beta":
        start, kept = start[..., 0], kept[..., 0]
    return start, kept


def main():
    saved = np.load(RESULTS / "draws_full.npz", allow_pickle=True)
    n_chains, n_keep = saved["mu"].shape
    n_burn = saved["burn_in_mu"].shape[1]

    fig, axes = plt.subplots(len(PARAMETERS), 3, figsize=(TEXT_WIDTH, 7.2), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.6, 1]})
    summary = {}
    for row, (name, symbol) in enumerate(PARAMETERS):
        start, kept = series(saved, name)
        summary[name] = (split_rhat(kept), effective_sample_size(kept))
        iteration = np.arange(n_burn + n_keep)
        for c in range(n_chains):
            axes[row, 0].plot(iteration[:N_START], start[c, :N_START], linewidth=0.9, label=f"chain {c + 1}")
            axes[row, 1].plot(iteration[n_burn::TRACE_THIN], kept[c, ::TRACE_THIN], linewidth=0.4, alpha=0.8)
            covariance = autocovariance(kept[c])
            axes[row, 2].plot(np.arange(MAX_LAG + 1), (covariance / covariance[0])[:MAX_LAG + 1], linewidth=0.9)
        axes[row, 0].set_ylabel(symbol, rotation=0, labelpad=14, fontsize=10)
        axes[row, 2].axhline(0, color="#52514e", linewidth=0.6)
        axes[row, 2].set_ylim(-0.1, 1.02)
        rhat, ess = summary[name]
        axes[row, 1].text(0.99, 0.96, f"$\\hat R$ = {fmt(rhat, 3)}, ESS = {fmt(ess, 0)}", transform=axes[row, 1].transAxes,
                          ha="right", va="top", fontsize=7,
                          bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
    axes[0, 0].set_title(f"(a) first {N_START} iterations", loc="left")
    axes[0, 1].set_title(f"(b) after burn-in (every {TRACE_THIN}th draw)", loc="left")
    axes[0, 2].set_title("(c) autocorrelation", loc="left")
    for ax in axes[-1]:
        ax.set_xlabel("lag" if ax is axes[-1, 2] else "iteration")
    for ax in axes[:-1].ravel():
        ax.tick_params(labelbottom=False)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncols=n_chains)

    caption = (
        f"Convergence of the sampler of the final model on the real data: {n_chains} chains started from points "
        f"drawn from the prior. (a): the first {N_START} iterations; (b): the {n_keep:,} iterations kept per chain "
        f"after a burn-in of {n_burn:,}, with the split-$\\hat R$ and the effective sample size (ESS) of the four "
        "chains together; (c): autocorrelation of each chain after burn-in."
    )
    save(fig, "trace_full", caption)


if __name__ == "__main__":
    main()
