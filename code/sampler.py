"""Metropolis-within-Gibbs sampler for the hierarchical logistic model of chess openings.

For game i played in opening o(i):
    y_i ~ Bernoulli(logistic(theta_o(i) + beta * delta_i))
    theta_o ~ Normal(mu, 1 / tau)
    mu ~ Normal(mu_mean, mu_sd^2), beta ~ Normal(beta_mean, beta_sd^2), tau ~ Gamma(shape, rate)
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed" / "games_clean.csv"

ROBERTS_SCALE = 2.38  # one-dimensional random-walk scale, Section 6.5 of the notes


@dataclass(frozen=True)
class Priors:
    mu_mean: float = 0.0
    mu_sd: float = 0.5
    beta_mean: float = np.log(10)  # the Elo formula gives log(10) per 400 rating points
    beta_sd: float = 1.0
    tau_shape: float = 1.0
    tau_rate: float = 0.05


@dataclass(frozen=True)
class Games:
    y: np.ndarray
    delta: np.ndarray
    opening: np.ndarray
    n_openings: int

    @property
    def games_per_opening(self):
        return np.bincount(self.opening, minlength=self.n_openings)


@dataclass(frozen=True)
class Steps:
    theta: np.ndarray
    beta: float


def load_games(path=PROCESSED):
    """Load the decisive games and return them with the ECO code of each opening index."""
    games = pd.read_csv(path)
    games = games[games["winner"] != "draw"]
    opening, codes = pd.factorize(games["opening_eco"], sort=True)
    data = Games(y=(games["winner"] == "white").to_numpy(dtype=float),
                 delta=games["delta"].to_numpy(), opening=opening, n_openings=len(codes))
    return data, np.asarray(codes)


def simulate_outcomes(data, beta, theta, rng):
    """Return a copy of the games with results drawn from the model."""
    eta = theta[data.opening] + beta * data.delta
    y = (rng.uniform(size=len(eta)) < 1 / (1 + np.exp(-eta))).astype(float)
    return Games(y=y, delta=data.delta, opening=data.opening, n_openings=data.n_openings)


def draw_from_prior(n_openings, priors, rng):
    """Draw (beta, mu, tau, theta) from the prior."""
    beta = rng.normal(priors.beta_mean, priors.beta_sd)
    mu = rng.normal(priors.mu_mean, priors.mu_sd)
    tau = rng.gamma(priors.tau_shape, 1 / priors.tau_rate)
    theta = rng.normal(mu, 1 / np.sqrt(tau), size=n_openings)
    return beta, mu, tau, theta


def log_likelihood_terms(data, eta):
    return data.y * eta - np.logaddexp(0, eta)


def initial_steps(data, priors):
    """Proposal scales from the curvature of the likelihood, used only to start the pilot run."""
    prior_tau = priors.tau_shape / priors.tau_rate
    theta_sd = 1 / np.sqrt(0.25 * data.games_per_opening + prior_tau)
    beta_sd = 1 / np.sqrt(0.25 * np.sum(data.delta**2) + 1 / priors.beta_sd**2)
    return Steps(theta=ROBERTS_SCALE * theta_sd, beta=ROBERTS_SCALE * beta_sd)


def run_chain(data, priors, steps, n_iter, rng, init=None):
    """Run one chain; returns the draws of every iteration and the Metropolis acceptance counts."""
    J = data.n_openings
    beta, mu, tau, theta = draw_from_prior(J, priors, rng) if init is None else init

    draws = {"beta": np.empty(n_iter), "mu": np.empty(n_iter), "sigma": np.empty(n_iter),
             "theta": np.empty((n_iter, J))}
    accepted_theta = np.zeros((n_iter, J), dtype=bool)
    accepted_beta = np.zeros(n_iter, dtype=bool)

    mu_prior_precision = 1 / priors.mu_sd**2
    for t in range(n_iter):
        # Openings are independent given (beta, mu, tau), so all of them move in one sweep.
        slope = beta * data.delta
        current = np.bincount(data.opening, log_likelihood_terms(data, theta[data.opening] + slope), J)
        proposal = theta + steps.theta * rng.standard_normal(J)
        proposed = np.bincount(data.opening, log_likelihood_terms(data, proposal[data.opening] + slope), J)
        log_ratio = proposed - current - 0.5 * tau * ((proposal - mu)**2 - (theta - mu)**2)
        accept = np.log(rng.uniform(size=J)) < log_ratio
        theta = np.where(accept, proposal, theta)
        accepted_theta[t] = accept

        intercept = theta[data.opening]
        beta_proposal = beta + steps.beta * rng.standard_normal()
        log_ratio = (log_likelihood_terms(data, intercept + beta_proposal * data.delta).sum()
                     - log_likelihood_terms(data, intercept + beta * data.delta).sum()
                     - 0.5 * ((beta_proposal - priors.beta_mean)**2
                              - (beta - priors.beta_mean)**2) / priors.beta_sd**2)
        if np.log(rng.uniform()) < log_ratio:
            beta = beta_proposal
            accepted_beta[t] = True

        variance = 1 / (mu_prior_precision + J * tau)
        mean = variance * (mu_prior_precision * priors.mu_mean + tau * theta.sum())
        mu = rng.normal(mean, np.sqrt(variance))

        tau = rng.gamma(priors.tau_shape + J / 2,
                        1 / (priors.tau_rate + 0.5 * np.sum((theta - mu)**2)))

        draws["beta"][t], draws["mu"][t], draws["sigma"][t] = beta, mu, 1 / np.sqrt(tau)
        draws["theta"][t] = theta

    return draws, accepted_theta, accepted_beta


def tune_steps(data, priors, rng, n_pilot):
    """Set each proposal scale to 2.38 posterior standard deviations, measured on a pilot run."""
    draws, _, _ = run_chain(data, priors, initial_steps(data, priors), n_pilot, rng)
    kept = n_pilot // 2
    return Steps(theta=ROBERTS_SCALE * draws["theta"][kept:].std(axis=0),
                 beta=ROBERTS_SCALE * draws["beta"][kept:].std())


def _run_seeded_chain(args):
    data, priors, steps, n_iter, seed = args
    return run_chain(data, priors, steps, n_iter, np.random.default_rng(seed))


def fit(data, priors, seed, n_chains=4, n_burn=2000, n_keep=5000, n_pilot=2000, parallel=False):
    """Tune on a pilot run, then run the chains from dispersed starting points drawn from the prior.

    With parallel=True each chain runs in its own process. Every chain has its own seed, so the
    draws are the same either way.
    Returns arrays of shape (chains, n_keep, ...) and the acceptance rates after burn-in.
    """
    pilot_seed, *chain_seeds = np.random.SeedSequence(seed).spawn(n_chains + 1)
    steps = tune_steps(data, priors, np.random.default_rng(pilot_seed), n_pilot)

    jobs = [(data, priors, steps, n_burn + n_keep, s) for s in chain_seeds]
    if parallel:
        with ProcessPoolExecutor(max_workers=n_chains) as pool:
            results = list(pool.map(_run_seeded_chain, jobs))
    else:
        results = [_run_seeded_chain(job) for job in jobs]
    draws = {name: np.stack([r[0][name][n_burn:] for r in results])
             for name in ("beta", "mu", "sigma", "theta")}
    acceptance = {"theta": np.mean([r[1][n_burn:].mean(axis=0) for r in results], axis=0),
                  "beta": np.mean([r[2][n_burn:].mean() for r in results])}
    return draws, acceptance, steps
