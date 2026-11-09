"""Refit the model under variants of the priors or of the data, and compare the key quantities.

Every variant is fitted with the settings and the seed of fit_real.py, so the differences between
variants come from the variant and not from different random numbers.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from diagnostics import effective_sample_size, split_rhat
from fit_real import N_KEEP, SEED
from sampler import fit

MIN_GAMES = 30  # openings compared in the ranking
N_TOP = 10


def key_quantities(draws):
    return {
        "beta": draws["beta"],
        "mu": draws["mu"],
        "sigma": draws["sigma"],
        "advantage": 400 * draws["mu"] / draws["beta"],  # White's advantage in rating points
    }


def fit_variant(data, codes, priors):
    """Posterior summaries of one variant, and the posterior mean of theta for each ECO code."""
    draws, _, _ = fit(data, priors, SEED, n_burn=2000, n_keep=N_KEEP, n_pilot=3000, parallel=True)
    row = {}
    for name, q in key_quantities(draws).items():
        low, high = np.percentile(q, [5, 95])
        row.update({f"{name}_mean": q.mean(), f"{name}_q05": low, f"{name}_q95": high})
    row["largest_rhat"] = max(split_rhat(draws[n]) for n in ("beta", "mu", "sigma"))
    row["smallest_ess"] = min(effective_sample_size(draws[n]) for n in ("beta", "mu", "sigma"))
    openings = pd.DataFrame({"theta": draws["theta"].mean(axis=(0, 1)), "games": data.games_per_opening},
                            index=codes)
    return row, openings


def ranking_agreement(openings, baseline):
    """Rank correlation of the opening effects with the baseline, on openings well observed in both."""
    both = openings.join(baseline, rsuffix="_baseline", how="inner")
    both = both[(both["games"] >= MIN_GAMES) & (both["games_baseline"] >= MIN_GAMES)]
    top = set(both.nlargest(N_TOP, "theta").index)
    top_baseline = set(both.nlargest(N_TOP, "theta_baseline").index)
    return {"openings_ranked": len(both),
            "rank_correlation_with_baseline": spearmanr(both["theta"], both["theta_baseline"]).statistic,
            f"top_{N_TOP}_shared_with_baseline": len(top & top_baseline)}


def compare(variants):
    """Fit each variant (name, description, data, codes, priors); the first one is the baseline."""
    rows, baseline = [], None
    for name, description, data, codes, priors in variants:
        row, openings = fit_variant(data, codes, priors)
        baseline = openings if baseline is None else baseline
        rows.append({"variant": name, "description": description, "games": len(data.y), **row,
                     **ranking_agreement(openings, baseline)})
        print(f"{name}: beta {row['beta_mean']:.3f}, mu {row['mu_mean']:.3f}, sigma {row['sigma_mean']:.3f}, "
              f"advantage {row['advantage_mean']:.1f}", flush=True)
    return pd.DataFrame(rows)
