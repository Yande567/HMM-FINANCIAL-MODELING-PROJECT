"""
Run:  python notebooks/demo_feature_scaling.py
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import adjusted_rand_score

SEED = 42
T = 1250
WINDOW = 10
SEEDS_TO_TRY = (42, 7, 2024, 101, 5)

A_TRUE = np.array(
    [
        [0.990, 0.007, 0.003],
        [0.010, 0.985, 0.005],
        [0.020, 0.030, 0.950],
    ]
)
MU_TRUE = np.array([0.00060, -0.00090, 0.00010])
SIGMA_TRUE = np.array([0.0070, 0.0170, 0.0300])


def build_features():
    """Simulate returns from known regimes, then build [return, volatility]."""
    rng = np.random.default_rng(SEED)
    states = np.zeros(T, dtype=int)
    for t in range(1, T):
        states[t] = rng.choice(3, p=A_TRUE[states[t - 1]])
    returns = rng.normal(MU_TRUE[states], SIGMA_TRUE[states])

    volatility = pd.Series(returns).rolling(WINDOW).std().to_numpy()
    # The first WINDOW-1 days have no complete window, so they are NaN.
    valid = ~np.isnan(volatility)
    return returns[valid], volatility[valid], states[valid]


def main():
    returns, volatility, truth = build_features()

    x_daily = np.column_stack([returns, volatility])
    x_annual = np.column_stack([returns, volatility * np.sqrt(252)])

    print(f"\nvolatility feature scale ratio: "
          f"{x_annual[:, 1].std() / x_daily[:, 1].std():.2f}x")
    print(f"daily  feature sds: return {x_daily[:, 0].std():.4f}, "
          f"volatility {x_daily[:, 1].std():.4f}")
    print(f"annual feature sds: return {x_annual[:, 0].std():.4f}, "
          f"volatility {x_annual[:, 1].std():.4f}   <- volatility now dominates\n")

    for seed in SEEDS_TO_TRY:
        kwargs = dict(n_components=3, covariance_type="full",
                      n_iter=1000, random_state=seed)
        m_daily = GaussianHMM(**kwargs).fit(x_daily)
        m_annual = GaussianHMM(**kwargs).fit(x_annual)

        s_daily = m_daily.predict(x_daily)
        s_annual = m_annual.predict(x_annual)

        print(
            f"seed {seed:>4}: "
            f"ARI(daily vs annualised) = {adjusted_rand_score(s_daily, s_annual):.3f}"
            f"   |  vs truth: daily {adjusted_rand_score(truth, s_daily):.3f}, "
            f"annualised {adjusted_rand_score(truth, s_annual):.3f}"
        )

    print("\nIf the two scalings were equivalent in practice, every ARI in the "
          "first column would be 1.000.\n")


if __name__ == "__main__":
    main()
