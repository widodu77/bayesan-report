"""Export small tables of the final model's draws for the notebook, so that it does not need the
240 MB results/draws_full.npz: thinned traces, autocorrelations, and the chance that the
higher-rated player wins. Nothing is refitted; the tables are read off the saved draws.

Input:  results/draws_full.npz (from fit_full.py)
Output: tables/trace_full.csv, tables/autocorrelation_full.csv, tables/favourite_full.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from diagnostics import autocovariance
from fig_trace_full import MAX_LAG, N_START, PARAMETERS, TRACE_THIN, series
from make_report_numbers import logistic

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
GAPS = (100, 200)  # rating gaps, in points, for the chance that the higher-rated player wins


def trace_table(saved):
    """First N_START iterations of the burn-in, then every TRACE_THIN-th kept draw, per chain."""
    n_burn = saved["burn_in_mu"].shape[1]
    rows = []
    for name, _ in PARAMETERS:
        start, kept = series(saved, name)
        for c in range(kept.shape[0]):
            rows.append(pd.DataFrame({"parameter": name, "chain": c + 1, "phase": "burn-in",
                                      "iteration": np.arange(N_START), "value": start[c, :N_START]}))
            iteration = n_burn + np.arange(0, kept.shape[1], TRACE_THIN)
            rows.append(pd.DataFrame({"parameter": name, "chain": c + 1, "phase": "kept",
                                      "iteration": iteration, "value": kept[c, ::TRACE_THIN]}))
    return pd.concat(rows, ignore_index=True)


def autocorrelation_table(saved):
    rows = []
    for name, _ in PARAMETERS:
        _, kept = series(saved, name)
        for c in range(kept.shape[0]):
            covariance = autocovariance(kept[c])
            rows.append(pd.DataFrame({"parameter": name, "chain": c + 1, "lag": np.arange(MAX_LAG + 1),
                                      "autocorrelation": (covariance / covariance[0])[:MAX_LAG + 1]}))
    return pd.concat(rows, ignore_index=True)


def favourite_table(saved):
    """Chance that the higher-rated player wins a rated game after 1.e4, averaged over the two colours,
    as in make_report_numbers.py."""
    beta = saved["beta"][:, ::int(saved["theta_thin"]), 0].ravel()
    gamma = saved["gamma"][..., list(saved["family_names"]).index("e4")].ravel().astype(float)
    rows = []
    for gap in GAPS:
        x = gap / 400
        score = 0.5 * (logistic(gamma + beta * x) + logistic(beta * x - gamma))
        rows.append({"rating_gap": gap, "mean": score.mean(), "q05": np.percentile(score, 5),
                     "q95": np.percentile(score, 95), "elo_expected_score": 1 / (1 + 10 ** (-x))})
    return pd.DataFrame(rows)


def main():
    saved = np.load(RESULTS / "draws_full.npz", allow_pickle=True)
    TABLES.mkdir(exist_ok=True)
    trace_table(saved).to_csv(TABLES / "trace_full.csv", index=False)
    autocorrelation_table(saved).to_csv(TABLES / "autocorrelation_full.csv", index=False)
    favourite = favourite_table(saved)
    favourite.to_csv(TABLES / "favourite_full.csv", index=False)
    print(favourite.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
