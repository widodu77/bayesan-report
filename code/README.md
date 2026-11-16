# Who controls White's first-move advantage?

Code for the project of Walid Ajbar and Millan Das, Bayesian Networks and Markov Models in Action (CentraleSupélec, 2026).

A Bayesian logistic regression of 17,682 decisive Lichess games on the rating gap, with hierarchical effects for White's first move and Black's reply, one rating slope for rated and one for unrated games, and one effect per player. All samplers and diagnostics are written by us with NumPy; PyMC is used only for an optional cross-check.

## How to run

With Python 3.12 and the packages of `requirements.txt` installed, from this folder:

```
python run_all.py
```

This runs every step below in order, from the raw data to the figures and the numbers of the report, and stops at the first failure. It takes about 3 hours on a 16-core machine; the validation scripts and the sensitivity analyses are the long steps, and they run several processes in parallel. Each script can also be run alone, in the order of `run_all.py`. Random seeds are fixed in every script, so a rerun reproduces every number (up to the last floating-point digits on a different machine).

`python run_all.py --with-pymc` also runs the optional PyMC cross-check, which needs `pip install -r requirements-crosscheck.txt`.

To build the report, from `../report`: `pdflatex report`, `bibtex report`, then `pdflatex report` twice. The report reads its figures from `figures/` and its numbers and tables from `../report/generated/`, written by `make_report_numbers.py`, so every number in it comes from this code.

The data, `data/raw/games.csv`, is included (CC0); `download_data.py` downloads it again only if it is missing.

## What produces what

The report has three models besides the final one: $M_0$ with one effect per opening code, $M_1$ with rated and unrated slopes, $M_2$ with first moves and replies, and the final model $M_3$ with player effects.

| Script | Outputs | In the report |
| --- | --- | --- |
| `download_data.py` | `data/raw/games.csv` | Section 2 |
| `prepare_data.py` | `data/processed/games_clean.csv`, `tables/data_cleaning.csv`, `tables/data_summary.csv` | Section 2 |
| `check_sampler.py` | `tables/check_quadrature.csv` | Section 5 (grid check of $M_0$) |
| `validate_simulated.py` | `tables/validation_recovery.csv`, `validation_recovery_theta.csv`, `validation_repeated.csv`, `validation_coverage.csv` | Table 1, Figure 9 |
| `fit_real.py` | `results/draws.npz`, `tables/posterior_shared.csv`, `posterior_openings.csv`, `sampler_diagnostics.csv` | Section 6 ($M_0$), Figures 10, 12 |
| `posterior_predictive.py` | `tables/posterior_predictive.csv`, `posterior_predictive_replicates.csv` | Section 6 ($M_0$), Figure 11 |
| `check_importance_sampling.py` | `tables/check_importance_sampling.csv` | Section 5 |
| `crosscheck_pymc.py` (optional) | `tables/crosscheck_pymc.csv` | Section 5 |
| `sensitivity_priors.py`, `sensitivity_data.py` | `tables/sensitivity_priors.csv`, `sensitivity_data.csv` | sensitivity of $M_0$ (not in the report) |
| `validate_rating_slopes.py` | `tables/validation_rating_slopes_recovery.csv`, `validation_rating_slopes_coverage.csv` | Table 1 |
| `fit_rating_slopes.py` | `results/draws_rating_slopes.npz`, `tables/posterior_rating_slopes.csv`, `favourite_win_probability.csv`, `posterior_predictive_rating_slopes.csv` | Section 6, Figure 3 |
| `validate_moves.py` | `tables/validation_moves_recovery.csv`, `validation_moves_coverage.csv` | Table 1 |
| `fit_moves.py` | `results/draws_moves.npz`, `tables/posterior_moves_shared.csv`, `posterior_first_moves.csv`, `posterior_replies.csv`, `sampler_diagnostics_moves.csv` | Section 6 ($M_2$) |
| `validate_full.py` | `tables/validation_full_recovery.csv`, `validation_full_repeated.csv`, `validation_full_coverage.csv` | Section 5, Table 1, Figure 6 |
| `fit_full.py` | `results/draws_full.npz`, `tables/posterior_full_shared.csv`, `posterior_full_first_moves.csv`, `posterior_full_replies.csv`, `posterior_full_players.csv`, `sampler_diagnostics_full.csv` | Section 6, Table 3, Figures 5, 7 |
| `posterior_predictive_full.py` | `tables/posterior_predictive_full.csv`, `posterior_predictive_full_replicates.csv` | Section 6, Figure 4 |
| `sensitivity_full.py` | `tables/sensitivity_full.csv` | Table 2 |
| `make_report_numbers.py` | `../report/generated/numbers.tex`, `table_validation.tex`, `table_sensitivity.tex`, `table_final.tex`, `table_cleaning.tex` | every number of the report, Tables 1 to 3 |

| Figure script | Output | In the report |
| --- | --- | --- |
| `fig_data.py` | `figures/data.*` | Figure 1 |
| `fig_model_full.py` | `figures/model_full.*` | Figure 2 |
| `fig_rating_slopes.py` | `figures/rating_slopes.*` | Figure 3 |
| `fig_posterior_predictive_full.py` | `figures/posterior_predictive_full.*` | Figure 4 |
| `fig_moves.py` | `figures/moves.*` | Figure 5 |
| `fig_validation_full.py` | `figures/validation_full.*` | Figure 6 |
| `fig_trace_full.py` | `figures/trace_full.*` | Figure 7 |
| `fig_model.py` | `figures/model.*` | Figure 8 |
| `fig_validation.py` | `figures/validation.*` | Figure 9 |
| `fig_trace.py` | `figures/trace.*` | Figure 10 |
| `fig_posterior_predictive.py` | `figures/posterior_predictive.*` | Figure 11 |
| `fig_openings.py` | `figures/openings.*` | Figure 12 |

Each figure script writes `figures/<name>.pdf` for the report, `figures/<name>.png` for a quick look, and `figures/<name>.tex`, a figure environment with a caption computed from the data (used for the figures in the appendices).

Modules: `sampler.py` holds the data loading and the sampler of $M_0$ to $M_1$ (one effect per opening, one rating slope per group), `sampler_moves.py` the sampler of $M_2$, `sampler_full.py` the sampler of the final model, `diagnostics.py` split-$\hat R$, effective sample sizes and Monte Carlo errors, `sensitivity.py` the refit-and-compare code of the sensitivity of $M_0$, and `plotting.py` the shared style of the figures.

## Data

Chess Game Dataset (Lichess), uploaded by datasnaek on Kaggle: https://www.kaggle.com/datasets/datasnaek/chess (CC0: Public Domain).
