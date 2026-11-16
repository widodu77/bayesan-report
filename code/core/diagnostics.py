"""Convergence and accuracy diagnostics for draws of shape (chains, iterations)."""
import numpy as np


def split_rhat(draws):
    """Split-Rhat of Definition 6.2 of the notes."""
    n = draws.shape[1] // 2
    sequences = np.concatenate([draws[:, :n], draws[:, -n:]])
    within = sequences.var(axis=1, ddof=1).mean()
    variance_of_means = sequences.mean(axis=1).var(ddof=1)
    return np.sqrt(((n - 1) / n * within + variance_of_means) / within)


def autocovariance(chain):
    """Autocovariances of one chain at every lag, computed by FFT."""
    n = len(chain)
    centred = chain - chain.mean()
    spectrum = np.fft.rfft(centred, n=2 * n)
    return np.fft.irfft(spectrum * np.conj(spectrum))[:n] / n


def effective_sample_size(draws):
    """Multi-chain ESS, truncating the sum of autocorrelations at Geyer's initial positive sequence."""
    m, n = draws.shape
    within = draws.var(axis=1, ddof=1).mean()
    variance_of_means = draws.mean(axis=1).var(ddof=1)
    total_variance = (n - 1) / n * within + variance_of_means
    if total_variance == 0:
        return np.nan

    mean_autocovariance = np.mean([autocovariance(chain) for chain in draws], axis=0)
    rho = 1 - (within * (n - 1) / n - mean_autocovariance) / total_variance
    rho[0] = 1.0

    tau = -1.0
    for k in range(0, n - 1, 2):
        pair = rho[k] + rho[k + 1]
        if pair < 0:
            break
        tau += 2 * pair
    return m * n / max(tau, 1e-12)


def monte_carlo_se(draws):
    return draws.std(ddof=1) / np.sqrt(effective_sample_size(draws))


def summarize(draws):
    """Posterior summary and diagnostics of one scalar quantity."""
    low, median, high = np.percentile(draws, [5, 50, 95])
    return {"mean": draws.mean(), "sd": draws.std(ddof=1), "q05": low, "median": median, "q95": high,
            "rhat": split_rhat(draws), "ess": effective_sample_size(draws),
            "mcse": monte_carlo_se(draws)}
