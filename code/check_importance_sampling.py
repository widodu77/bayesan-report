"""Importance sampling with a Laplace proposal on the model without openings, checked against MCMC.

Model: P(White wins game i) = logistic(alpha + beta * delta_i), with the priors of mu (for alpha)
and beta in sampler.Priors. With two parameters and 17,682 games the posterior is close to normal,
so the Laplace approximation at the mode is a good proposal (Example 3.4 of the notes). The same
posterior is sampled by random-walk Metropolis-Hastings, and the two must agree within their
Monte Carlo errors.

Output: tables/check_importance_sampling.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from diagnostics import effective_sample_size, monte_carlo_se, split_rhat
from sampler import ROBERTS_SCALE, Priors, load_games

TABLES = Path(__file__).resolve().parent / "tables"
SEED = 20261112
N_IMPORTANCE = 50_000
N_CHAINS, N_BURN, N_KEEP = 4, 2000, 20_000
NAMES = ("alpha", "beta")


class Posterior:
    """Log posterior of (alpha, beta), with its gradient and Hessian."""

    def __init__(self, y, delta, priors):
        self.y, self.design = y, np.column_stack([np.ones_like(delta), delta])
        self.prior_mean = np.array([priors.mu_mean, priors.beta_mean])
        self.prior_precision = np.diag([1 / priors.mu_sd**2, 1 / priors.beta_sd**2])

    def log_density(self, params, chunk=500):
        """Unnormalized log posterior; params has shape (2,) or (n, 2), evaluated in chunks of rows."""
        params = np.atleast_2d(params)
        log_lik = np.empty(len(params))
        for start in range(0, len(params), chunk):
            eta = params[start:start + chunk] @ self.design.T
            log_lik[start:start + chunk] = (self.y * eta - np.logaddexp(0, eta)).sum(axis=1)
        centred = params - self.prior_mean
        return log_lik - 0.5 * np.einsum("ni,ij,nj->n", centred, self.prior_precision, centred)

    def gradient_and_hessian(self, params):
        p = 1 / (1 + np.exp(-(self.design @ params)))
        gradient = self.design.T @ (self.y - p) - self.prior_precision @ (params - self.prior_mean)
        hessian = -(self.design.T * (p * (1 - p))) @ self.design - self.prior_precision
        return gradient, hessian


def laplace(posterior, tol=1e-10, max_iter=100):
    """Mode by Newton's method, and the covariance minus the inverse Hessian there."""
    params = posterior.prior_mean.copy()
    for _ in range(max_iter):
        gradient, hessian = posterior.gradient_and_hessian(params)
        step = np.linalg.solve(hessian, gradient)
        params = params - step
        if np.max(np.abs(step)) < tol:
            return params, np.linalg.inv(-posterior.gradient_and_hessian(params)[1])
    raise RuntimeError("Newton's method did not converge")


def weighted_summary(x, w):
    """Self-normalized estimates of the mean and standard deviation, with their Monte Carlo errors.

    The error of a self-normalized estimate of E[h] is sqrt(sum w_i^2 (h_i - E[h])^2) for weights
    summing to one; the standard deviation uses the delta method on E[(x - mean)^2].
    """
    mean = np.sum(w * x)
    squared = (x - mean)**2
    variance = np.sum(w * squared)
    mean_se = np.sqrt(np.sum(w**2 * (x - mean)**2))
    variance_se = np.sqrt(np.sum(w**2 * (squared - variance)**2))
    return mean, mean_se, np.sqrt(variance), variance_se / (2 * np.sqrt(variance))


def importance_sampling(posterior, mode, covariance, rng):
    chol = np.linalg.cholesky(covariance)
    z = rng.standard_normal((N_IMPORTANCE, 2))
    draws = mode + z @ chol.T
    log_proposal = -0.5 * np.sum(z**2, axis=1)  # up to a constant, which cancels after normalizing
    log_weights = posterior.log_density(draws) - log_proposal
    w = np.exp(log_weights - log_weights.max())
    w /= w.sum()
    return draws, w, 1 / np.sum(w**2)


def metropolis_hastings(posterior, mode, covariance, rng):
    """Random-walk Metropolis-Hastings from dispersed starts, with proposal 2.38^2/d times the Laplace covariance."""
    chol = np.linalg.cholesky(covariance)
    step = ROBERTS_SCALE / np.sqrt(2) * chol
    current = mode + 3 * rng.standard_normal((N_CHAINS, 2)) @ chol.T
    log_current = posterior.log_density(current)
    draws = np.empty((N_CHAINS, N_BURN + N_KEEP, 2))
    accepted = np.zeros(N_CHAINS)
    for t in range(N_BURN + N_KEEP):
        proposal = current + rng.standard_normal((N_CHAINS, 2)) @ step.T
        log_proposal = posterior.log_density(proposal)
        accept = np.log(rng.uniform(size=N_CHAINS)) < log_proposal - log_current
        current[accept], log_current[accept] = proposal[accept], log_proposal[accept]
        draws[:, t] = current
        if t >= N_BURN:
            accepted += accept
    return draws[:, N_BURN:], accepted.mean() / N_KEEP


def chain_summary(chains):
    """Mean and standard deviation of draws of shape (chains, iterations), with their Monte Carlo errors."""
    mean = chains.mean()
    squared = (chains - mean)**2
    sd = np.sqrt(squared.mean())
    return mean, monte_carlo_se(chains), sd, monte_carlo_se(squared) / (2 * sd)


def main():
    data, _ = load_games()
    posterior = Posterior(data.y, data.delta, Priors())
    mode, covariance = laplace(posterior)
    rng = np.random.default_rng(SEED)

    is_draws, weights, weight_ess = importance_sampling(posterior, mode, covariance, rng)
    mh_draws, acceptance = metropolis_hastings(posterior, mode, covariance, rng)

    rows = []
    for k, name in enumerate(NAMES):
        is_mean, is_mean_se, is_sd, is_sd_se = weighted_summary(is_draws[:, k], weights)
        chains = mh_draws[:, :, k]
        mh_mean, mh_mean_se, mh_sd, mh_sd_se = chain_summary(chains)
        rows.append({
            "quantity": name,
            "laplace_mode": mode[k],
            "laplace_sd": np.sqrt(covariance[k, k]),
            "is_mean": is_mean, "is_mean_mcse": is_mean_se, "is_sd": is_sd, "is_sd_mcse": is_sd_se,
            "mh_mean": mh_mean, "mh_mean_mcse": mh_mean_se, "mh_sd": mh_sd, "mh_sd_mcse": mh_sd_se,
            "z_mean": (is_mean - mh_mean) / np.hypot(is_mean_se, mh_mean_se),
            "z_sd": (is_sd - mh_sd) / np.hypot(is_sd_se, mh_sd_se),
            "is_draws": N_IMPORTANCE,
            "is_weight_ess": weight_ess,
            "mh_draws": mh_draws[:, :, k].size,
            "mh_ess": effective_sample_size(chains),
            "mh_rhat": split_rhat(chains),
            "mh_acceptance": acceptance,
        })
    table = pd.DataFrame(rows)
    TABLES.mkdir(exist_ok=True)
    table.to_csv(TABLES / "check_importance_sampling.csv", index=False)
    print(table.T.to_string(header=False))


if __name__ == "__main__":
    main()
