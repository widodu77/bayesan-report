"""Prior sensitivity: refit the model on the real games under other priors and compare.

Output: tables/sensitivity_priors.csv
"""
from dataclasses import replace
from pathlib import Path

from sampler import Priors, load_games
from sensitivity import compare

TABLES = Path(__file__).resolve().parent / "tables"


def variants():
    base = Priors()
    return [
        ("baseline", "tau ~ Gamma(1, 0.05), mu ~ N(0, 0.5^2), beta ~ N(log 10, 1)", base),
        ("tau Gamma(2, 0.1)", "same prior mean of tau, half the variance", replace(base, tau_shape=2.0, tau_rate=0.1)),
        ("tau Gamma(0.5, 0.005)", "diffuse: prior mean 100 and standard deviation 141",
         replace(base, tau_shape=0.5, tau_rate=0.005)),
        ("mu sd 0.25", "mu ~ N(0, 0.25^2)", replace(base, mu_sd=0.25)),
        ("mu sd 1", "mu ~ N(0, 1)", replace(base, mu_sd=1.0)),
        ("beta N(0, 1)", "beta centred on 0 instead of the Elo value log 10", replace(base, beta_mean=0.0)),
    ]


def main():
    data, codes = load_games()
    table = compare([(name, description, data, codes, priors) for name, description, priors in variants()])
    TABLES.mkdir(exist_ok=True)
    table.to_csv(TABLES / "sensitivity_priors.csv", index=False)
    columns = ["variant", "beta_mean", "mu_mean", "sigma_mean", "sigma_q05", "sigma_q95", "advantage_mean",
               "rank_correlation_with_baseline", "top_10_shared_with_baseline", "largest_rhat", "smallest_ess"]
    print(table[columns].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
