"""Metropolis-within-Gibbs sampler for the final model: White's first move and Black's reply, one
rating slope per group of games, and one effect per player.

For game i in pair c(i) = (first move, reply), between White player w(i) and Black player b(i):
    y_i ~ Bernoulli(logistic(theta_c(i) + beta_g(i) * delta_i + u_w(i) - u_b(i)))
    theta_c ~ Normal(gamma_m(c), 1 / tau_reply),  gamma_m ~ Normal(mu, 1 / tau_move)
    u_p ~ Normal(0, 1 / tau_player)
    mu ~ Normal(mu_mean, mu_sd^2), beta_g ~ Normal(beta_mean, beta_sd^2),
    tau_reply, tau_move, tau_player ~ Gamma(tau_shape, tau_rate), the priors of sampler.Priors.

u_p is how much stronger (u_p > 0) or weaker player p is than their rating says, on the log-odds
scale. Without it, the games of one player are treated as independent although the same player keeps
beating, or missing, their rating, which makes the posterior intervals too narrow.

The player effects are updated in batches of players who never play each other. Given everything
else, the players of one batch share no game, so they are independent and each gets its own
random-walk Metropolis step in the same sweep; the batches come from a greedy colouring of the
graph whose edges are the games.

Most players have one or two games, so each u_p is held near the current spread and the spread near
the u_p: one-at-a-time updates only creep along this funnel. Two extra Metropolis-Hastings moves
rescale a whole level at once, u -> e^s u with tau_player -> e^(-2s) tau_player, and the replies'
deviations theta - gamma -> e^s (theta - gamma) with tau_reply -> e^(-2s) tau_reply. The normal
prior of the rescaled level is unchanged by the move, so the acceptance ratio is the likelihood
ratio times the Gamma prior ratio of tau and the Jacobian: exp(dloglik - 2 a s - b tau (e^(-2s) - 1)).

The 14,832 player effects are not stored at every iteration: the chain keeps their running sums,
for posterior means and standard deviations, and a draw every u_every iterations.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np

from core.sampler import ROBERTS_SCALE, initial_steps, sum_by_slope


@dataclass(frozen=True)
class Batch:
    players: np.ndarray  # players of the batch
    white_games: np.ndarray  # games whose White player is in the batch
    white_local: np.ndarray  # position of that player in players
    black_games: np.ndarray
    black_local: np.ndarray


@dataclass(frozen=True)
class Steps:
    theta: np.ndarray
    beta: np.ndarray
    u: np.ndarray
    scale_reply: float  # standard deviation of s in the rescaling moves
    scale_player: float


def player_batches(data):
    """Greedy colouring of the players, most active first, so that no game joins two players of one colour."""
    white, black, P = data.white_player, data.black_player, data.n_players
    if np.any(white == black):
        raise ValueError("a game with the same player on both sides cannot be coloured")
    neighbours = [set() for _ in range(P)]
    for w, b in zip(white, black, strict=True):
        neighbours[w].add(b)
        neighbours[b].add(w)
    games = np.bincount(white, minlength=P) + np.bincount(black, minlength=P)
    colour = np.full(P, -1)
    for p in np.argsort(-games, kind="stable"):
        used = {colour[q] for q in neighbours[p]}
        colour[p] = next(c for c in range(len(used) + 1) if c not in used)

    batches = []
    for c in range(colour.max() + 1):
        players = np.flatnonzero(colour == c)
        local = np.full(P, -1)
        local[players] = np.arange(len(players))
        white_games, black_games = np.flatnonzero(colour[white] == c), np.flatnonzero(colour[black] == c)
        batches.append(Batch(players, white_games, local[white[white_games]], black_games,
                             local[black[black_games]]))
    return batches


def games_per_player(data):
    return (np.bincount(data.white_player, minlength=data.n_players)
            + np.bincount(data.black_player, minlength=data.n_players))


def draw_from_prior(data, priors, rng):
    """Draw (beta, mu, tau_move, tau_reply, tau_player, gamma, theta, u) from the prior."""
    beta = rng.normal(priors.beta_mean, priors.beta_sd, size=data.n_slopes)
    mu = rng.normal(priors.mu_mean, priors.mu_sd)
    tau_move, tau_reply, tau_player = rng.gamma(priors.tau_shape, 1 / priors.tau_rate, size=3)
    gamma = rng.normal(mu, 1 / np.sqrt(tau_move), size=len(data.family_names))
    theta = rng.normal(gamma[data.family], 1 / np.sqrt(tau_reply))
    u = rng.normal(0, 1 / np.sqrt(tau_player), size=data.n_players)
    return beta, mu, tau_move, tau_reply, tau_player, gamma, theta, u


def simulate_outcomes(data, beta, theta, u, rng):
    """Return a copy of the games with results drawn from the final model."""
    eta = (theta[data.opening] + np.atleast_1d(beta)[data.slope_index] * data.delta
           + u[data.white_player] - u[data.black_player])
    y = (rng.uniform(size=len(eta)) < 1 / (1 + np.exp(-eta))).astype(float)
    return replace(data, y=y)


def terms(y, eta):
    """Log-likelihood of each game: y log p + (1 - y) log(1 - p) with p = logistic(eta)."""
    return y * eta - np.logaddexp(0, eta)


def initial_full_steps(data, priors):
    """Proposal scales from the curvature of the likelihood, used only to start the pilot run."""
    steps = initial_steps(data, priors)
    prior_tau = priors.tau_shape / priors.tau_rate
    u_sd = 1 / np.sqrt(0.25 * games_per_player(data) + prior_tau)
    return Steps(theta=steps.theta, beta=steps.beta, u=ROBERTS_SCALE * u_sd, scale_reply=0.05, scale_player=0.02)


def run_chain(data, priors, steps, batches, n_iter, rng, accumulate_from=0, u_every=10, init=None):
    """Run one chain.

    Returns the draws of every iteration for all parameters except the player effects; for those,
    the sums of u and u^2 from iteration accumulate_from on, and a draw every u_every iterations.
    """
    J, K, M, P = data.n_openings, data.n_slopes, len(data.family_names), data.n_players
    group, family, opening = data.slope_index, data.family, data.opening
    white, black, y = data.white_player, data.black_player, data.y
    replies_per_move = np.bincount(family, minlength=M)
    beta, mu, tau_move, tau_reply, tau_player, gamma, theta, u = (
        draw_from_prior(data, priors, rng) if init is None else init)

    draws = {"beta": np.empty((n_iter, K)), "mu": np.empty(n_iter), "sigma_move": np.empty(n_iter),
             "sigma_reply": np.empty(n_iter), "sigma_player": np.empty(n_iter),
             "gamma": np.empty((n_iter, M)), "theta": np.empty((n_iter, J))}
    u_sum, u_sumsq, u_draws = np.zeros(P), np.zeros(P), []
    accepted = {"theta": np.zeros(J), "beta": np.zeros(K), "u": np.zeros(P), "scale_reply": 0.0,
                "scale_player": 0.0}
    a, b = priors.tau_shape, priors.tau_rate

    mu_prior_precision = 1 / priors.mu_sd**2
    for t in range(n_iter):
        counting = t >= accumulate_from
        skill = u[white] - u[black]

        # Pairs (first move, reply), given everything else, are independent: one sweep for all.
        offset = beta[group] * data.delta + skill
        current = np.bincount(opening, terms(y, theta[opening] + offset), J)
        proposal = theta + steps.theta * rng.standard_normal(J)
        proposed = np.bincount(opening, terms(y, proposal[opening] + offset), J)
        centre = gamma[family]
        log_ratio = proposed - current - 0.5 * tau_reply * ((proposal - centre)**2 - (theta - centre)**2)
        accept = np.log(rng.uniform(size=J)) < log_ratio
        theta = np.where(accept, proposal, theta)
        accepted["theta"] += counting * accept

        s = steps.scale_reply * rng.standard_normal()
        centre = gamma[family]
        rescaled = centre + np.exp(s) * (theta - centre)
        log_ratio = (terms(y, rescaled[opening] + offset).sum() - terms(y, theta[opening] + offset).sum()
                     - 2 * a * s - b * tau_reply * (np.exp(-2 * s) - 1))
        if np.log(rng.uniform()) < log_ratio:
            theta, tau_reply = rescaled, tau_reply * np.exp(-2 * s)
            accepted["scale_reply"] += counting

        intercept = theta[opening] + skill
        beta_proposal = beta + steps.beta * rng.standard_normal(K)
        log_ratio = (sum_by_slope(data, terms(y, intercept + beta_proposal[group] * data.delta))
                     - sum_by_slope(data, terms(y, intercept + beta[group] * data.delta))
                     - 0.5 * ((beta_proposal - priors.beta_mean)**2
                              - (beta - priors.beta_mean)**2) / priors.beta_sd**2)
        accept = np.log(rng.uniform(size=K)) < log_ratio
        beta = np.where(accept, beta_proposal, beta)
        accepted["beta"] += counting * accept

        # Player effects, one batch of players who never meet at a time.
        eta = theta[opening] + beta[group] * data.delta + skill
        for batch in batches:
            step = steps.u[batch.players] * rng.standard_normal(len(batch.players))
            wg, bg = batch.white_games, batch.black_games
            new_white = eta[wg] + step[batch.white_local]
            new_black = eta[bg] - step[batch.black_local]
            change = (np.bincount(batch.white_local, terms(y[wg], new_white) - terms(y[wg], eta[wg]),
                                  len(batch.players))
                      + np.bincount(batch.black_local, terms(y[bg], new_black) - terms(y[bg], eta[bg]),
                                    len(batch.players)))
            old = u[batch.players]
            log_ratio = change - 0.5 * tau_player * ((old + step)**2 - old**2)
            accept = np.log(rng.uniform(size=len(batch.players))) < log_ratio
            u[batch.players] = np.where(accept, old + step, old)
            eta[wg] = np.where(accept[batch.white_local], new_white, eta[wg])
            eta[bg] = np.where(accept[batch.black_local], new_black, eta[bg])
            if counting:
                accepted["u"][batch.players] += accept

        s = steps.scale_player * rng.standard_normal()
        rescaled_eta = eta + (np.exp(s) - 1) * (u[white] - u[black])
        log_ratio = (terms(y, rescaled_eta).sum() - terms(y, eta).sum()
                     - 2 * a * s - b * tau_player * (np.exp(-2 * s) - 1))
        if np.log(rng.uniform()) < log_ratio:
            u, tau_player = np.exp(s) * u, tau_player * np.exp(-2 * s)
            accepted["scale_player"] += counting

        # Conjugate Normal and Gamma updates of the hierarchy.
        precision = tau_move + replies_per_move * tau_reply
        mean = (tau_move * mu + tau_reply * np.bincount(family, theta, M)) / precision
        gamma = rng.normal(mean, 1 / np.sqrt(precision))
        variance = 1 / (mu_prior_precision + M * tau_move)
        mu = rng.normal(variance * (mu_prior_precision * priors.mu_mean + tau_move * gamma.sum()), np.sqrt(variance))
        tau_reply = rng.gamma(priors.tau_shape + J / 2,
                              1 / (priors.tau_rate + 0.5 * np.sum((theta - gamma[family])**2)))
        tau_move = rng.gamma(priors.tau_shape + M / 2, 1 / (priors.tau_rate + 0.5 * np.sum((gamma - mu)**2)))
        tau_player = rng.gamma(priors.tau_shape + P / 2, 1 / (priors.tau_rate + 0.5 * np.sum(u**2)))

        draws["beta"][t], draws["mu"][t] = beta, mu
        draws["sigma_move"][t], draws["sigma_reply"][t] = 1 / np.sqrt(tau_move), 1 / np.sqrt(tau_reply)
        draws["sigma_player"][t] = 1 / np.sqrt(tau_player)
        draws["gamma"][t], draws["theta"][t] = gamma, theta
        if counting:
            u_sum += u
            u_sumsq += u**2
            if (t - accumulate_from) % u_every == 0:
                u_draws.append(u.astype(np.float32))

    n_counted = max(n_iter - accumulate_from, 1)
    acceptance = {name: count / n_counted for name, count in accepted.items()}
    players = {"sum": u_sum, "sumsq": u_sumsq, "count": n_iter - accumulate_from,
               "draws": np.array(u_draws, dtype=np.float32).reshape(-1, P)}
    return draws, players, acceptance


def tune_steps(data, priors, batches, rng, n_pilot):
    """Set each proposal scale to 2.38 posterior standard deviations, measured on a pilot run."""
    kept = n_pilot // 2
    initial = initial_full_steps(data, priors)
    draws, players, _ = run_chain(data, priors, initial, batches, n_pilot, rng,
                                  accumulate_from=kept, u_every=n_pilot)
    u_mean = players["sum"] / players["count"]
    u_sd = np.sqrt(np.maximum(players["sumsq"] / players["count"] - u_mean**2, 0))
    theta_sd = draws["theta"][kept:].std(axis=0)
    # A parameter that never moved in the pilot keeps its initial step, rather than a step of 0 that
    # would freeze it for the whole run.
    return Steps(theta=np.where(theta_sd > 0, ROBERTS_SCALE * theta_sd, initial.theta),
                 beta=np.array([ROBERTS_SCALE * draws["beta"][kept:, k].std() for k in range(data.n_slopes)]),
                 u=np.where(u_sd > 0, ROBERTS_SCALE * u_sd, initial.u),
                 scale_reply=max(ROBERTS_SCALE * np.log(draws["sigma_reply"][kept:]).std(), 1e-3),
                 scale_player=max(ROBERTS_SCALE * np.log(draws["sigma_player"][kept:]).std(), 1e-3))


def _run_seeded_chain(args):
    data, priors, steps, batches, n_iter, n_burn, u_every, seed = args
    return run_chain(data, priors, steps, batches, n_iter, np.random.default_rng(seed), n_burn, u_every)


def fit(data, priors, seed, n_chains=4, n_burn=2000, n_keep=5000, n_pilot=2000, parallel=False, u_every=10,
        keep_burn_in=False):
    """Tune on a pilot run, then run the chains from dispersed starting points drawn from the prior.

    Returns the draws, of shape (chains, n_keep, ...) or (chains, n_burn + n_keep, ...) with
    keep_burn_in=True; the player effects (posterior mean and sd over all chains, and the kept draws
    of shape (chains, n_keep // u_every, players)); and the acceptance rates after burn-in.
    """
    pilot_seed, *chain_seeds = np.random.SeedSequence(seed).spawn(n_chains + 1)
    batches = player_batches(data)
    steps = tune_steps(data, priors, batches, np.random.default_rng(pilot_seed), n_pilot)

    jobs = [(data, priors, steps, batches, n_burn + n_keep, n_burn, u_every, s) for s in chain_seeds]
    if parallel:
        with ProcessPoolExecutor(max_workers=n_chains) as pool:
            results = list(pool.map(_run_seeded_chain, jobs))
    else:
        results = [_run_seeded_chain(job) for job in jobs]

    first = 0 if keep_burn_in else n_burn
    draws = {name: np.stack([r[0][name][first:] for r in results]) for name in results[0][0]}
    count = sum(r[1]["count"] for r in results)
    u_mean = sum(r[1]["sum"] for r in results) / count
    u_sd = np.sqrt(np.maximum(sum(r[1]["sumsq"] for r in results) / count - u_mean**2, 0))
    players = {"mean": u_mean, "sd": u_sd, "draws": np.stack([r[1]["draws"] for r in results])}
    acceptance = {name: np.mean([r[2][name] for r in results], axis=0) for name in results[0][2]}
    if data.n_slopes == 1:
        draws["beta"], acceptance["beta"] = draws["beta"][..., 0], acceptance["beta"][0]
    return draws, players, acceptance
