# Who controls White's first-move advantage?

Code for the project of Walid Ajbar and Millan Das, Bayesian Networks and Markov Models in Action (CentraleSupélec, 2026).

A Bayesian logistic regression of 17,682 decisive Lichess games on the rating gap, with hierarchical effects for White's first move and Black's reply, one rating slope for rated and one for unrated games, and one effect per player. All samplers and diagnostics are written by us with NumPy; PyMC is used only for an optional cross-check.

**Where to start:** `chess_analysis.ipynb` walks through the whole project, from the raw data to the answer.

## What is where

```
code/
├── README.md                this file
├── chess_analysis.ipynb     the walkthrough of the project (start here)
├── run_all.py               runs every script below, in order
├── requirements.txt         the packages (requirements-crosscheck.txt adds PyMC for the optional cross-check)
│
├── core/                    shared code, imported by the scripts
│   ├── sampler.py             data loading, and the sampler of M0 and M1
│   ├── sampler_moves.py       the sampler of M2 (first move and reply)
│   ├── sampler_full.py        the sampler of the final model (M2 + player effects)
│   ├── diagnostics.py         split-Rhat, effective sample sizes, Monte Carlo errors
│   ├── plotting.py            the style of the figures
│   └── rounding.py            the one rounding rule of every number shown
│
├── data_prep/               1. get and clean the games
├── validation/              2. check the samplers on simulated games and against other methods
├── fitting/                 3. fit the models to the real games
├── model_checks/            4. posterior predictive checks and sensitivity analysis
├── for_report/              5. the figures of the report, and every number and table it quotes
│
├── data/                    the games: raw/ (included) and processed/ (written by data_prep)
├── tables/                  the results of the scripts, as small CSV tables (all included)
├── figures/                 the figures of the report (all included)
└── results/                 the posterior draws (not included: about 300 MB, rebuilt by fitting/)
```

The report is in `../report/`.

## How to run

With Python 3.12 and the packages of `requirements.txt` installed, from this folder:

```
python run_all.py
```

This runs every step below in order, from the raw data to the figures and the numbers of the report, and stops at the first failure. We estimate that it takes about 3 hours on a 16-core machine (we ran the scripts one at a time, never all at once); the validation scripts and the sensitivity analyses are the long steps, and they run several processes in parallel. Random seeds are fixed in every script, so a rerun reproduces every number (up to the last floating-point digits on a different machine).

Each script can also be run alone, from this folder, as a module so that it finds `core/`: for example `python -m fitting.fit_full`. Run them in the order of `run_all.py`.

`python run_all.py --with-pymc` also runs the optional PyMC cross-check, which needs `pip install -r requirements-crosscheck.txt`.

To build the report, from `../report`: `pdflatex report`, `bibtex report`, then `pdflatex report` twice. The report reads its figures from `figures/` and its numbers and tables from `../report/generated/`, written by `for_report/make_report_numbers.py`, so every number in it comes from this code.

The data, `data/raw/games.csv`, is included (CC0); `data_prep/download_data.py` downloads it again only if it is missing.

## The notebook

`chess_analysis.ipynb` does not fit any model: it only reads the tables that the scripts save in `tables/` (all of them are included), so it runs in a few seconds with the packages of `requirements.txt` and `jupyter`. Whenever a step needs a fit or simulated games, it names the script that did it. Since every script has a fixed seed, running `run_all.py` again rebuilds exactly the same tables. Its Part 12 recomputes every number of the report that the notebook quotes and checks that it matches `../report/generated/numbers.tex`.

## What produces what

The report has three models besides the final one: $M_0$ with one effect per opening code, $M_1$ with rated and unrated slopes, $M_2$ with first moves and replies, and the final model $M_3$ with player effects.

| Script | Outputs | In the report |
| --- | --- | --- |
| `data_prep/download_data.py` | `data/raw/games.csv` | Section 2 |
| `data_prep/prepare_data.py` | `data/processed/games_clean.csv`, `tables/data_cleaning.csv`, `tables/data_summary.csv` | Section 2 |
| `validation/check_sampler.py` | `tables/check_quadrature.csv` | Section 5 (grid check of $M_0$) |
| `validation/validate_simulated.py` | `tables/validation_recovery.csv`, `validation_coverage.csv` | Table 1 |
| `fitting/fit_real.py` | `results/draws.npz`, `tables/posterior_shared.csv`, `posterior_openings.csv` | Section 6 ($M_0$) |
| `model_checks/posterior_predictive.py` | `tables/posterior_predictive.csv` | Section 6 ($M_0$) |
| `validation/check_importance_sampling.py` | `tables/check_importance_sampling.csv` | Section 5 |
| `validation/crosscheck_pymc.py` (optional) | `tables/crosscheck_pymc.csv` | Section 5 |
| `validation/validate_rating_slopes.py` | `tables/validation_rating_slopes_recovery.csv`, `validation_rating_slopes_coverage.csv` | Table 1 |
| `fitting/fit_rating_slopes.py` | `results/draws_rating_slopes.npz`, `tables/posterior_rating_slopes.csv`, `favourite_win_probability.csv`, `posterior_predictive_rating_slopes.csv` | Section 6, Figure 3 |
| `validation/validate_moves.py` | `tables/validation_moves_recovery.csv`, `validation_moves_coverage.csv` | Table 1 |
| `fitting/fit_moves.py` | `results/draws_moves.npz` (draws of $M_2$, the final model without player effects) | used by `model_checks/posterior_predictive_full.py` |
| `validation/validate_full.py` | `tables/validation_full_recovery.csv`, `validation_full_repeated.csv`, `validation_full_coverage.csv` | Section 5, Table 1, Figure 6 |
| `fitting/fit_full.py` | `results/draws_full.npz`, `tables/posterior_full_shared.csv`, `posterior_full_first_moves.csv`, `posterior_full_replies.csv`, `posterior_full_players.csv`, `sampler_diagnostics_full.csv` | Section 6, Table 3, Figures 5, 7 |
| `fitting/export_traces.py` | `tables/trace_full.csv`, `autocorrelation_full.csv`, `favourite_full.csv` | Section 6, the notebook (Parts 8 and 9) |
| `model_checks/posterior_predictive_full.py` | `tables/posterior_predictive_full.csv`, `posterior_predictive_full_replicates.csv` | Section 6, Figure 4 |
| `model_checks/sensitivity_full.py` | `tables/sensitivity_full.csv` (its baseline row is the main fit of `fitting/fit_full.py`, draw for draw) | Table 2 |
| `for_report/make_report_numbers.py` | `../report/generated/numbers.tex`, `table_validation.tex`, `table_sensitivity.tex`, `table_final.tex` | every number of the report, Tables 1 to 3 |

| Figure script | Output | In the report |
| --- | --- | --- |
| `for_report/fig_data.py` | `figures/data.*` | Figure 1 |
| `for_report/fig_model_full.py` | `figures/model_full.*` | Figure 2 |
| `for_report/fig_rating_slopes.py` | `figures/rating_slopes.*` | Figure 3 |
| `for_report/fig_posterior_predictive_full.py` | `figures/posterior_predictive_full.*` | Figure 4 |
| `for_report/fig_moves.py` | `figures/moves.*` | Figure 5 |
| `for_report/fig_validation_full.py` | `figures/validation_full.*` | Figure 6 |
| `for_report/fig_trace_full.py` | `figures/trace_full.*` | Figure 7 |

Each figure script writes `figures/<name>.pdf` for the report, `figures/<name>.png` for a quick look, and `figures/<name>.tex`, a figure environment with a caption computed from the data (used for the figures in the appendices).

## Data

Chess Game Dataset (Lichess), uploaded by datasnaek on Kaggle: https://www.kaggle.com/datasets/datasnaek/chess (CC0: Public Domain).
