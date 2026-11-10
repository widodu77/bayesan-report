"""Metropolis-within-Gibbs sampler for the model of White's first move and Black's reply.

Each game is classified by c = (m, r): White's first move m, chosen by White alone, and Black's
reply r, chosen by Black alone. For game i:
    y_i ~ Bernoulli(logistic(theta_c(i) + beta_g(i) * delta_i))
    theta_c ~ Normal(gamma_m(c), 1 / tau_reply)   how Black's reply moves the result away from the first move's
    gamma_m ~ Normal(mu, 1 / tau_move)             the value of White's first move
    mu ~ Normal(mu_mean, mu_sd^2), beta_g ~ Normal(beta_mean, beta_sd^2),
    tau_reply, tau_move ~ Gamma(tau_shape, tau_rate), the priors of sampler.Priors.

sigma_move = 1 / sqrt(tau_move) measures how much White's first move matters, and
sigma_reply = 1 / sqrt(tau_reply) how much Black's reply matters once the first move is fixed.

Use it with sampler.fit(..., run=run_chain) on data from load_games(openings="moves").
"""
import numpy as np

from sampler import log_likelihood_terms, sum_by_slope


def draw_from_prior(data, priors, rng):
    """Draw (beta, mu, tau_move, tau_reply, gamma, theta) from the prior."""
    K, M = data.n_slopes, len(data.family_names)
    beta = rng.normal(priors.beta_mean, priors.beta_sd, size=K)
    mu = rng.normal(priors.mu_mean, priors.mu_sd)
    tau_move = rng.gamma(priors.tau_shape, 1 / priors.tau_rate)
    tau_reply = rng.gamma(priors.tau_shape, 1 / priors.tau_rate)
    gamma = rng.normal(mu, 1 / np.sqrt(tau_move), size=M)
    theta = rng.normal(gamma[data.family], 1 / np.sqrt(tau_reply))
    return beta, mu, tau_move, tau_reply, gamma, theta


def run_chain(data, priors, steps, n_iter, rng, init=None):
    """Run one chain; returns the draws of every iteration and the Metropolis acceptance counts."""
    J, K, M = data.n_openings, data.n_slopes, len(data.family_names)
    group, family = data.slope_index, data.family
    replies_per_move = np.bincount(family, minlength=M)
    beta, mu, tau_move, tau_reply, gamma, theta = draw_from_prior(data, priors, rng) if init is None else init

    draws = {"beta": np.empty((n_iter, K)), "mu": np.empty(n_iter), "sigma_reply": np.empty(n_iter),
             "sigma_move": np.empty(n_iter), "gamma": np.empty((n_iter, M)), "theta": np.empty((n_iter, J))}
    accepted_theta = np.zeros((n_iter, J), dtype=bool)
    accepted_beta = np.zeros((n_iter, K), dtype=bool)

    mu_prior_precision = 1 / priors.mu_sd**2
    for t in range(n_iter):
        # Pairs (first move, reply) are independent given the rest, so all of them move in one sweep.
        slope = beta[group] * data.delta
        centre = gamma[family]
        current = np.bincount(data.opening, log_likelihood_terms(data, theta[data.opening] + slope), J)
        proposal = theta + steps.theta * rng.standard_normal(J)
        proposed = np.bincount(data.opening, log_likelihood_terms(data, proposal[data.opening] + slope), J)
        log_ratio = proposed - current - 0.5 * tau_reply * ((proposal - centre)**2 - (theta - centre)**2)
        accept = np.log(rng.uniform(size=J)) < log_ratio
        theta = np.where(accept, proposal, theta)
        accepted_theta[t] = accept

        intercept = theta[data.opening]
        beta_proposal = beta + steps.beta * rng.standard_normal(K)
        log_ratio = (sum_by_slope(data, log_likelihood_terms(data, intercept + beta_proposal[group] * data.delta))
                     - sum_by_slope(data, log_likelihood_terms(data, intercept + beta[group] * data.delta))
                     - 0.5 * ((beta_proposal - priors.beta_mean)**2
                              - (beta - priors.beta_mean)**2) / priors.beta_sd**2)
        accept = np.log(rng.uniform(size=K)) < log_ratio
        beta = np.where(accept, beta_proposal, beta)
        accepted_beta[t] = accept

        # Conjugate Normal update of each first move from the replies played against it.
        precision = tau_move + replies_per_move * tau_reply
        mean = (tau_move * mu + tau_reply * np.bincount(family, theta, M)) / precision
        gamma = rng.normal(mean, 1 / np.sqrt(precision))

        variance = 1 / (mu_prior_precision + M * tau_move)
        mean = variance * (mu_prior_precision * priors.mu_mean + tau_move * gamma.sum())
        mu = rng.normal(mean, np.sqrt(variance))

        tau_reply = rng.gamma(priors.tau_shape + J / 2,
                              1 / (priors.tau_rate + 0.5 * np.sum((theta - gamma[family])**2)))
        tau_move = rng.gamma(priors.tau_shape + M / 2,
                             1 / (priors.tau_rate + 0.5 * np.sum((gamma - mu)**2)))

        draws["beta"][t], draws["mu"][t] = beta, mu
        draws["sigma_reply"][t], draws["sigma_move"][t] = 1 / np.sqrt(tau_reply), 1 / np.sqrt(tau_move)
        draws["gamma"][t], draws["theta"][t] = gamma, theta

    return draws, accepted_theta, accepted_beta
