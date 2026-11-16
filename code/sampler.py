"""Metropolis-within-Gibbs sampler for the hierarchical logistic model of chess openings.

For game i played in opening o(i):
    y_i ~ Bernoulli(logistic(theta_o(i) + beta_g(i) * delta_i))
    theta_o ~ Normal(mu, 1 / tau)
    mu ~ Normal(mu_mean, mu_sd^2), beta_g ~ Normal(beta_mean, beta_sd^2), tau ~ Gamma(shape, rate)

g(i) is the group of game i for the rating slope. By default every game is in one group, which is
the main model of the report; load_games(slope_by="rated") gives rated and unrated games their own slope.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
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
    slope_group: np.ndarray = None  # group of each game for the rating slope; None means one group
    slope_names: tuple = ("all games",)
    family: np.ndarray = None  # first move of each opening, only with load_games(openings="moves")
    family_names: tuple = ()

    @property
    def games_per_opening(self):
        return np.bincount(self.opening, minlength=self.n_openings)

    @property
    def n_slopes(self):
        return len(self.slope_names)

    @property
    def slope_index(self):
        if self.slope_group is None:
            return np.zeros(len(self.y), dtype=int)
        return self.slope_group


@dataclass(frozen=True)
class Steps:
    theta: np.ndarray
    beta: np.ndarray


def load_games(path=PROCESSED, draws_as_half=False, rated_only=False, drop_rating=None, min_plies=None,
               slope_by=None, openings="eco"):
    """Load the games and return them with the ECO code of each opening index.

    By default only decisive games are kept. The options give the variants of the data sensitivity
    analysis: draws_as_half keeps the draws with y = 0.5, rated_only keeps the rated games,
    drop_rating removes the games with a player at exactly that rating, and min_plies removes
    the games shorter than that (the processed file already has at least 10 plies).
    slope_by="rated" gives rated and unrated games their own rating slope.
    openings="moves" replaces the ECO code by the pair (White's first move, Black's reply), and
    records the first move of each pair for the model of sampler_moves.py.
    """
    games = pd.read_csv(path)
    if not draws_as_half:
        games = games[games["winner"] != "draw"]
    if rated_only:
        games = games[games["rated"]]
    if drop_rating is not None:
        games = games[(games["white_rating"] != drop_rating) & (games["black_rating"] != drop_rating)]
    if min_plies is not None:
        games = games[games["turns"] >= min_plies]
    if openings == "eco":
        opening, codes = pd.factorize(games["opening_eco"], sort=True)
    elif openings == "moves":
        opening, codes = pd.factorize(games["first_move"] + " " + games["reply"], sort=True)
    else:
        raise ValueError(f"unknown openings: {openings}")
    y = (games["winner"] == "white").to_numpy(dtype=float) + 0.5 * (games["winner"] == "draw").to_numpy()
    data = Games(y=y, delta=games["delta"].to_numpy(), opening=opening, n_openings=len(codes))
    if openings == "moves":
        family, family_names = pd.factorize(pd.Series(codes).str.split().str[0], sort=True)
        data = replace(data, family=family, family_names=tuple(family_names))
    if slope_by == "rated":
        data = replace(data, slope_group=(~games["rated"]).to_numpy().astype(int),
                       slope_names=("rated games", "unrated games"))
    elif slope_by is not None:
        raise ValueError(f"unknown slope_by: {slope_by}")
    return data, np.asarray(codes)


def simulate_outcomes(data, beta, theta, rng):
    """Return a copy of the games with results drawn from the model; beta has one value per slope."""
    eta = theta[data.opening] + np.atleast_1d(beta)[data.slope_index] * data.delta
    y = (rng.uniform(size=len(eta)) < 1 / (1 + np.exp(-eta))).astype(float)
    return replace(data, y=y)


def draw_from_prior(n_openings, priors, rng, n_slopes=1):
    """Draw (beta, mu, tau, theta) from the prior; beta is a number with one slope, else an array."""
    beta = rng.normal(priors.beta_mean, priors.beta_sd, size=None if n_slopes == 1 else n_slopes)
    mu = rng.normal(priors.mu_mean, priors.mu_sd)
    tau = rng.gamma(priors.tau_shape, 1 / priors.tau_rate)
    theta = rng.normal(mu, 1 / np.sqrt(tau), size=n_openings)
    return beta, mu, tau, theta


def log_likelihood_terms(data, eta):
    return data.y * eta - np.logaddexp(0, eta)


def sum_by_slope(data, values):
    """Sum of values over the games of each slope group."""
    # With one slope, a plain sum keeps the draws identical to the runs reported so far.
    if data.n_slopes == 1:
        return np.array([values.sum()])
    return np.bincount(data.slope_index, values, data.n_slopes)


def initial_steps(data, priors):
    """Proposal scales from the curvature of the likelihood, used only to start the pilot run."""
    prior_tau = priors.tau_shape / priors.tau_rate
    theta_sd = 1 / np.sqrt(0.25 * data.games_per_opening + prior_tau)
    beta_sd = 1 / np.sqrt(0.25 * sum_by_slope(data, data.delta**2) + 1 / priors.beta_sd**2)
    return Steps(theta=ROBERTS_SCALE * theta_sd, beta=ROBERTS_SCALE * beta_sd)


def run_chain(data, priors, steps, n_iter, rng, init=None):
    """Run one chain; returns the draws of every iteration and the Metropolis acceptance counts."""
    J, K, group = data.n_openings, data.n_slopes, data.slope_index
    beta, mu, tau, theta = draw_from_prior(J, priors, rng, K) if init is None else init
    beta = np.atleast_1d(beta).astype(float)

    draws = {"beta": np.empty((n_iter, K)), "mu": np.empty(n_iter), "sigma": np.empty(n_iter),
             "theta": np.empty((n_iter, J))}
    accepted_theta = np.zeros((n_iter, J), dtype=bool)
    accepted_beta = np.zeros((n_iter, K), dtype=bool)

    mu_prior_precision = 1 / priors.mu_sd**2
    for t in range(n_iter):
        # Openings are independent given (beta, mu, tau), so all of them move in one sweep.
        slope = beta[group] * data.delta
        current = np.bincount(data.opening, log_likelihood_terms(data, theta[data.opening] + slope), J)
        proposal = theta + steps.theta * rng.standard_normal(J)
        proposed = np.bincount(data.opening, log_likelihood_terms(data, proposal[data.opening] + slope), J)
        log_ratio = proposed - current - 0.5 * tau * ((proposal - mu)**2 - (theta - mu)**2)
        accept = np.log(rng.uniform(size=J)) < log_ratio
        theta = np.where(accept, proposal, theta)
        accepted_theta[t] = accept

        # The slopes of different groups are independent given theta, so they also move in one sweep.
        intercept = theta[data.opening]
        beta_proposal = beta + steps.beta * rng.standard_normal(K)
        log_ratio = (sum_by_slope(data, log_likelihood_terms(data, intercept + beta_proposal[group] * data.delta))
                     - sum_by_slope(data, log_likelihood_terms(data, intercept + beta[group] * data.delta))
                     - 0.5 * ((beta_proposal - priors.beta_mean)**2
                              - (beta - priors.beta_mean)**2) / priors.beta_sd**2)
        accept = np.log(rng.uniform(size=K)) < log_ratio
        beta = np.where(accept, beta_proposal, beta)
        accepted_beta[t] = accept

        variance = 1 / (mu_prior_precision + J * tau)
        mean = variance * (mu_prior_precision * priors.mu_mean + tau * theta.sum())
        mu = rng.normal(mean, np.sqrt(variance))

        tau = rng.gamma(priors.tau_shape + J / 2,
                        1 / (priors.tau_rate + 0.5 * np.sum((theta - mu)**2)))

        draws["beta"][t], draws["mu"][t], draws["sigma"][t] = beta, mu, 1 / np.sqrt(tau)
        draws["theta"][t] = theta

    return draws, accepted_theta, accepted_beta


def tune_steps(data, priors, rng, n_pilot, run=run_chain):
    """Set each proposal scale to 2.38 posterior standard deviations, measured on a pilot run."""
    draws, _, _ = run(data, priors, initial_steps(data, priors), n_pilot, rng)
    kept = n_pilot // 2
    return Steps(theta=ROBERTS_SCALE * draws["theta"][kept:].std(axis=0),
                 beta=np.array([ROBERTS_SCALE * draws["beta"][kept:, k].std() for k in range(data.n_slopes)]))


def _run_seeded_chain(args):
    run, data, priors, steps, n_iter, seed = args
    return run(data, priors, steps, n_iter, np.random.default_rng(seed))


def fit(data, priors, seed, n_chains=4, n_burn=2000, n_keep=5000, n_pilot=2000, parallel=False,
        keep_burn_in=False, run=run_chain):
    """Tune on a pilot run, then run the chains from dispersed starting points drawn from the prior.

    With parallel=True each chain runs in its own process. Every chain has its own seed, so the
    draws are the same either way. run is the function that runs one chain: run_chain here, or
    the sampler of the model of first moves and replies in sampler_moves.py.
    Returns arrays of shape (chains, n_keep, ...), or (chains, n_burn + n_keep, ...) with
    keep_burn_in=True, and the acceptance rates after burn-in. With several rating slopes, beta and
    its acceptance rate get a last axis with one entry per slope.
    """
    pilot_seed, *chain_seeds = np.random.SeedSequence(seed).spawn(n_chains + 1)
    steps = tune_steps(data, priors, np.random.default_rng(pilot_seed), n_pilot, run)

    jobs = [(run, data, priors, steps, n_burn + n_keep, s) for s in chain_seeds]
    if parallel:
        with ProcessPoolExecutor(max_workers=n_chains) as pool:
            results = list(pool.map(_run_seeded_chain, jobs))
    else:
        results = [_run_seeded_chain(job) for job in jobs]
    first = 0 if keep_burn_in else n_burn
    draws = {name: np.stack([r[0][name][first:] for r in results]) for name in results[0][0]}
    acceptance = {"theta": np.mean([r[1][n_burn:].mean(axis=0) for r in results], axis=0),
                  "beta": np.mean([r[2][n_burn:].mean(axis=0) for r in results], axis=0)}
    if data.n_slopes == 1:
        draws["beta"], acceptance["beta"] = draws["beta"][..., 0], acceptance["beta"][0]
    return draws, acceptance, steps
