"""Run the whole analysis from the raw data to the figures, in order, and stop at the first failure.

Usage: python run_all.py [--with-pymc]

--with-pymc also runs the optional PyMC cross-check, which needs requirements-crosscheck.txt.
The running time of each step is printed and written to tables/running_times.csv.
"""
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

STEPS = [
    # Data
    "download_data.py",
    "prepare_data.py",
    # First model: one effect per ECO code
    "check_sampler.py",
    "validate_simulated.py",
    "fit_real.py",
    "posterior_predictive.py",
    "check_importance_sampling.py",
    # Rated and unrated rating slopes
    "validate_rating_slopes.py",
    "fit_rating_slopes.py",
    # White's first move and Black's reply
    "validate_moves.py",
    # Final model, with player effects
    "validate_full.py",
    "fit_full.py",
    "posterior_predictive_full.py",
    "sensitivity_full.py",
    # Figures
    "fig_data.py",
    "fig_model_full.py",
    "fig_validation_full.py",
    "fig_trace_full.py",
    "fig_rating_slopes.py",
    "fig_moves.py",
    "fig_posterior_predictive_full.py",
    # Every number and table of the report
    "make_report_numbers.py",
]


def main():
    steps = list(STEPS)
    if "--with-pymc" in sys.argv[1:]:
        steps.insert(steps.index("fit_real.py") + 1, "crosscheck_pymc.py")
    times = []
    for script in steps:
        print(f"=== {script}", flush=True)
        start = time.time()
        subprocess.run([sys.executable, script], cwd=ROOT, check=True)
        times.append({"script": script, "minutes": (time.time() - start) / 60})
        print(f"=== {script}: {times[-1]['minutes']:.1f} min", flush=True)
    table = pd.DataFrame(times)
    table.to_csv(ROOT / "tables" / "running_times.csv", index=False)
    print(f"total: {table['minutes'].sum():.0f} min")


if __name__ == "__main__":
    main()
