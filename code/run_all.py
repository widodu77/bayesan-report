"""Run the whole analysis from the raw data to the figures, in order, and stop at the first failure.

Usage, from code/: python run_all.py [--with-pymc]

Each step is a script of one of the folders, run as a module (python -m folder.script) so that it can
import the shared code of core/. One step can be run alone the same way, for example
python -m fitting.fit_full.

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
    "data_prep.download_data",
    "data_prep.prepare_data",
    # First model: one effect per ECO code
    "validation.check_sampler",
    "validation.validate_simulated",
    "fitting.fit_real",
    "model_checks.posterior_predictive",
    "validation.check_importance_sampling",
    # Rated and unrated rating slopes
    "validation.validate_rating_slopes",
    "fitting.fit_rating_slopes",
    # White's first move and Black's reply
    "validation.validate_moves",
    "fitting.fit_moves",
    # Final model, with player effects
    "validation.validate_full",
    "fitting.fit_full",
    "fitting.export_traces",
    "model_checks.posterior_predictive_full",
    "model_checks.sensitivity_full",
    # Figures
    "for_report.fig_data",
    "for_report.fig_model_full",
    "for_report.fig_validation_full",
    "for_report.fig_trace_full",
    "for_report.fig_rating_slopes",
    "for_report.fig_moves",
    "for_report.fig_posterior_predictive_full",
    # Every number and table of the report
    "for_report.make_report_numbers",
]


def main():
    steps = list(STEPS)
    if "--with-pymc" in sys.argv[1:]:
        steps.insert(steps.index("fitting.fit_real") + 1, "validation.crosscheck_pymc")
    times = []
    for module in steps:
        print(f"=== {module}", flush=True)
        start = time.time()
        subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True)
        times.append({"step": module, "minutes": (time.time() - start) / 60})
        print(f"=== {module}: {times[-1]['minutes']:.1f} min", flush=True)
    table = pd.DataFrame(times)
    table.to_csv(ROOT / "tables" / "running_times.csv", index=False)
    print(f"total: {table['minutes'].sum():.0f} min")


if __name__ == "__main__":
    main()
