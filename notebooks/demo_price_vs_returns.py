"""
Controlled experiment: why the HMM is fed LOG RETURNS and not PRICES.

THE ARGUMENT
------------
A Gaussian HMM assumes each hidden state emits observations from a *fixed*
distribution N(mu_i, Sigma_i). The S&P 500 price level trends upward for
decades, so there is no such thing as a stationary "bull price" — the index at
4,000 was a bull market in 2021 and a bear market in 2022. Log returns, by
contrast, are approximately stationary: the distribution of daily returns in a
2022 bear market genuinely resembles that of a 2008 bear market.

Rather than assert this, we test it. We generate data from a Markov chain whose
regimes we PLANT ourselves, so the correct answer is known, then fit the same
GaussianHMM twice — once on the price level, once on the log returns — and
score each against the truth with the Adjusted Rand Index.

RESULT (seed 42, 1250 days)
---------------------------
    Observation fed to HMM     ARI vs truth   Switches decoded (truth: 20)
    price level                    -0.005                  2
    log return                     +0.914                 16

The price-based model does not fail randomly; it fails systematically. With no
stationary structure to latch onto it partitions the TIMELINE into three price
bands (mean levels ~104, ~67, ~37, occupying days 0-350, 351-591 and 592-1249)
and never revisits a state after leaving it. That is a chart of "when was the
index expensive", not a regime detector.

Run:  python notebooks/demo_price_vs_returns.py
"""

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import adjusted_rand_score

SEED = 42
T = 1250  # ~5 trading years

# --- 1. Plant three KNOWN regimes via a Markov chain -----------------------
# Diagonal entries are large because real regimes persist. Expected duration in
# state i is 1 / (1 - A[i, i]): 100, 67 and 20 trading days respectively.
A_TRUE = np.array(
    [
        [0.990, 0.007, 0.003],  # 0 = Bull      drift up, calm
        [0.010, 0.985, 0.005],  # 1 = Bear      drift down, choppy
        [0.020, 0.030, 0.950],  # 2 = High-vol  no drift, violent
    ]
)
MU_TRUE = np.array([0.00060, -0.00090, 0.00010])
SIGMA_TRUE = np.array([0.0070, 0.0170, 0.0300])


def simulate(rng):
    """Return (true_states, log_returns, price_path)."""
    states = np.zeros(T, dtype=int)
    for t in range(1, T):
        states[t] = rng.choice(3, p=A_TRUE[states[t - 1]])
    returns = rng.normal(MU_TRUE[states], SIGMA_TRUE[states])
    # A price path is just the compounded return path. Because these are LOG
    # returns, compounding is a cumulative sum inside an exponential.
    prices = 100 * np.exp(np.cumsum(returns))
    return states, returns, prices


def fit_and_report(name, observations, true_states):
    X = observations.reshape(-1, 1)  # hmmlearn always wants 2D: (n_samples, n_features)
    model = GaussianHMM(
        n_components=3, covariance_type="full", n_iter=500, random_state=7
    ).fit(X)
    decoded = model.predict(X)  # Viterbi

    ari = adjusted_rand_score(true_states, decoded)
    print(f"--- {name} ---")
    print(f"  Adjusted Rand Index vs truth: {ari:+.3f}   (1.0 = perfect, 0.0 = chance)")
    for i in range(3):
        idx = np.where(decoded == i)[0]
        print(
            f"  state {i}: fitted mean = {model.means_[i, 0]:+.5f}"
            f"   days = {idx.size:4d}"
            f"   occupies day-index {idx.min():4d} to {idx.max():4d}"
        )
    print(f"  regime switches decoded: {int((np.diff(decoded) != 0).sum())}\n")


def main():
    rng = np.random.default_rng(SEED)
    true_states, returns, prices = simulate(rng)

    shares = [np.mean(true_states == i) for i in range(3)]
    print(
        f"\nTrue regime share:  Bull {shares[0]:.0%}   "
        f"Bear {shares[1]:.0%}   High-vol {shares[2]:.0%}"
    )
    print(f"True regime switches: {int((np.diff(true_states) != 0).sum())}\n")

    fit_and_report("PRICE LEVEL as observation", prices, true_states)
    fit_and_report("LOG RETURN as observation", returns, true_states)


if __name__ == "__main__":
    main()
