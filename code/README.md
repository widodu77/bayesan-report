# Which chess openings give White an edge?

Code for the project of Walid Ajbar and Millan Das, Bayesian Networks and Markov Models in Action.

## How to run

From this folder, with the packages of `requirements.txt` installed:

```
python download_data.py
python prepare_data.py
python check_sampler.py
python validate_simulated.py
python fit_real.py
```

`download_data.py` does nothing if `data/raw/games.csv` is already there.
`check_sampler.py` takes about 1 minute, `validate_simulated.py` over 8 minutes on 12 processes, and `fit_real.py` about 2.5 minutes.
Random seeds are fixed in each script.

## What produces what

| Script | Output |
| --- | --- |
| `download_data.py` | `data/raw/games.csv` |
| `prepare_data.py` | `data/processed/games_clean.csv`, `tables/data_cleaning.csv`, `tables/data_summary.csv` |
| `check_sampler.py` | `tables/check_quadrature.csv` |
| `validate_simulated.py` | `tables/validation_recovery.csv`, `tables/validation_repeated.csv`, `tables/validation_coverage.csv` |
| `fit_real.py` | `results/draws.npz`, `tables/posterior_shared.csv`, `tables/posterior_openings.csv`, `tables/sampler_diagnostics.csv` |

`sampler.py` holds the model and the sampler, and `diagnostics.py` the convergence measures.

## Data

Chess Game Dataset (Lichess), uploaded by datasnaek on Kaggle: https://www.kaggle.com/datasets/datasnaek/chess (CC0: Public Domain).

## Project checklist

Remove this section before submission. Each open item has an issue on GitHub.
Due: Monday 16 November 2026, 23:59 Paris time, on EduNao.

### Done

- [x] Topic chosen after T10 was taken, and approved by the professor
- [x] Pitch deck made
- [x] Project folder created, LaTeX template copied into `report/`
- [x] Data downloaded from Kaggle (`download_data.py`)
- [x] Data cleaned (`prepare_data.py`): 20,058 raw games to 17,682 decisive games, 359 openings
- [x] Cleaning and summary tables written
- [x] Sampler written (`sampler.py`)
- [x] Diagnostics written: split-Rhat, ESS, Monte Carlo error (`diagnostics.py`)
- [x] Sampler matches exact grid integration on a small problem (`check_sampler.py`)
- [x] Validation on simulated games (`validate_simulated.py`)
- [x] Real data fitted (`fit_real.py`), split-Rhat at most 1.005
- [x] Posterior summaries written

### To do: analysis

- [ ] Run longer chains for the final numbers ([#1](https://github.com/widodu77/bayesan-report/issues/1))
- [ ] Posterior predictive checks ([#2](https://github.com/widodu77/bayesan-report/issues/2))
- [ ] Prior sensitivity analysis ([#3](https://github.com/widodu77/bayesan-report/issues/3))
- [ ] Data sensitivity analysis ([#4](https://github.com/widodu77/bayesan-report/issues/4))
- [ ] Importance sampling on the model without openings ([#5](https://github.com/widodu77/bayesan-report/issues/5))
- [ ] Optional: PyMC cross-check of the hierarchical model ([#6](https://github.com/widodu77/bayesan-report/issues/6))

### To do: figures

- [ ] Figure: descriptive view of the data ([#7](https://github.com/widodu77/bayesan-report/issues/7))
- [ ] Figure: the model as a Bayesian network ([#8](https://github.com/widodu77/bayesan-report/issues/8))
- [ ] Figure: validation on simulated data ([#9](https://github.com/widodu77/bayesan-report/issues/9))
- [ ] Figure: trace plots and autocorrelations ([#10](https://github.com/widodu77/bayesan-report/issues/10))
- [ ] Figure: ranking of the openings with uncertainty ([#11](https://github.com/widodu77/bayesan-report/issues/11))
- [ ] Figure: posterior predictive checks ([#12](https://github.com/widodu77/bayesan-report/issues/12))

### To do: report

- [ ] Set up the report file ([#13](https://github.com/widodu77/bayesan-report/issues/13))
- [ ] Report: Introduction ([#14](https://github.com/widodu77/bayesan-report/issues/14))
- [ ] Report: Data ([#15](https://github.com/widodu77/bayesan-report/issues/15))
- [ ] Report: Model ([#16](https://github.com/widodu77/bayesan-report/issues/16))
- [ ] Report: Method ([#17](https://github.com/widodu77/bayesan-report/issues/17))
- [ ] Report: Validation ([#18](https://github.com/widodu77/bayesan-report/issues/18))
- [ ] Report: Results ([#19](https://github.com/widodu77/bayesan-report/issues/19))
- [ ] Report: Discussion and limitations ([#20](https://github.com/widodu77/bayesan-report/issues/20))
- [ ] Report: References ([#21](https://github.com/widodu77/bayesan-report/issues/21))
- [ ] Final report review ([#25](https://github.com/widodu77/bayesan-report/issues/25))

### To do: code and submission

- [ ] One script that runs everything ([#22](https://github.com/widodu77/bayesan-report/issues/22))
- [ ] Finish the README ([#23](https://github.com/widodu77/bayesan-report/issues/23))
- [ ] Final run from a clean state ([#24](https://github.com/widodu77/bayesan-report/issues/24))
- [ ] Make sure both of you can explain every part ([#26](https://github.com/widodu77/bayesan-report/issues/26))
- [ ] Build the zip and upload it to EduNao ([#27](https://github.com/widodu77/bayesan-report/issues/27))
