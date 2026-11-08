"""Compare the sampler with deterministic numerical integration on a small simulated data set.

Given (beta, mu, tau) the opening effects are independent, so each one is a one-dimensional
integral and the posterior reduces to a three-dimensional grid. The two methods share only the
data and the priors, so agreement within the Monte Carlo error checks the sampler's updates.

Output: tables/check_quadrature.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from diagnostics import monte_carlo_se, split_rhat
from sampler import Games, Priors, fit

TABLES = Path(__file__).resolve().parent / "tables"
SEED = 20261108
GAMES_PER_OPENING = (40, 20, 8)

BETA_GRID = np.linspace(0.0, 5.5, 81)
MU_GRID = np.linspace(-2.5, 2.5, 81)
LOG_TAU_GRID = np.linspace(-5.0, 6.5, 101)
THETA_GRID = np.linspace(-5.0, 5.0, 801)


def simulate_small_problem(rng):
    opening = np.repeat(np.arange(len(GAMES_PER_OPENING)), GAMES_PER_OPENING)
    delta = rng.normal(0, 0.6, size=len(opening))
    true_theta = np.array([0.5, 0.0, -0.4])
    eta = true_theta[opening] + 1.5 * delta
    y = (rng.uniform(size=len(eta)) < 1 / (1 + np.exp(-eta))).astype(float)
    return Games(y=y, delta=delta, opening=opening, n_openings=len(GAMES_PER_OPENING))


def grid_posterior(data, priors):
    """Posterior means of (beta, mu, sigma, theta) by integration on the grid."""
    J = data.n_openings
    mu_mesh, log_tau_mesh = np.meshgrid(MU_GRID, LOG_TAU_GRID, indexing="ij")
    tau_mesh = np.exp(log_tau_mesh)
    deviation = THETA_GRID[None, None, :] - mu_mesh[:, :, None]
    theta_prior = np.exp(0.5 * log_tau_mesh[:, :, None] - 0.5 * tau_mesh[:, :, None] * deviation**2)

    log_post = np.empty((len(BETA_GRID), *mu_mesh.shape))
    theta_mean = np.empty((*log_post.shape, J))
    for b, beta in enumerate(BETA_GRID):
        log_z = np.zeros(mu_mesh.shape)
        for j in range(J):
            games = data.opening == j
            eta = THETA_GRID[:, None] + beta * data.delta[games]
            log_lik = np.sum(data.y[games] * eta - np.logaddexp(0, eta), axis=1)
            offset = log_lik.max()
            integrand = theta_prior * np.exp(log_lik - offset)
            z = integrand.sum(axis=2)
            log_z += np.log(z) + offset
            theta_mean[b, :, :, j] = (integrand * THETA_GRID).sum(axis=2) / z
        log_post[b] = (log_z - 0.5 * ((beta - priors.beta_mean) / priors.beta_sd)**2
                       - 0.5 * ((mu_mesh - priors.mu_mean) / priors.mu_sd)**2
                       + priors.tau_shape * log_tau_mesh - priors.tau_rate * tau_mesh)

    weights = np.exp(log_post - log_post.max())
    weights /= weights.sum()
    beta_axis = BETA_GRID[:, None, None]
    return np.r_[np.sum(weights * beta_axis),
                 np.sum(weights * mu_mesh[None]),
                 np.sum(weights * np.exp(-0.5 * log_tau_mesh[None])),
                 np.sum(weights[..., None] * theta_mean, axis=(0, 1, 2))]


def names(J):
    return ["beta", "mu", "sigma"] + [f"theta[{j}]" for j in range(J)]


def main():
    rng = np.random.default_rng(SEED)
    priors = Priors()
    data = simulate_small_problem(rng)
    J = data.n_openings

    draws, acceptance, _ = fit(data, priors, SEED + 1, n_burn=5000, n_keep=100_000, n_pilot=5000)
    series = [draws["beta"], draws["mu"], draws["sigma"]] + [draws["theta"][:, :, j] for j in range(J)]
    table = pd.DataFrame({
        "quantity": names(J),
        "mcmc_mean": [s.mean() for s in series],
        "mcmc_mcse": [monte_carlo_se(s) for s in series],
        "grid_mean": grid_posterior(data, priors),
        "rhat": [split_rhat(s) for s in series],
    })
    table["z"] = (table["mcmc_mean"] - table["grid_mean"]) / table["mcmc_mcse"]
    TABLES.mkdir(exist_ok=True)
    table.to_csv(TABLES / "check_quadrature.csv", index=False)

    print(table.round(4).to_string(index=False))
    print(f"acceptance: beta {acceptance['beta']:.2f}, theta {acceptance['theta'].round(2)}")


if __name__ == "__main__":
    main()
