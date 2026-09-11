"""
Sprint Tasks 3.1 - 3.3 — chronological split, feature scaling, and fitting.

WHAT BAUM-WELCH IS DOING HERE

We have no regime labels, so this is unsupervised: EM must infer the transition
matrix A, the emission parameters (mu_i, Sigma_i) and the initial distribution
pi from unlabelled returns alone.

EM is hill-climbing on a non-convex likelihood surface, so where it stops
depends on where it started. A single fit reports one local optimum and cannot
tell you whether a better one exists. Each restart is an independent climb; we
keep the best and report the SPREAD, because that spread is the evidence for
whether the regimes are a property of the data or an artefact of
initialisation (Note 02).

WHY THE FEATURES ARE STANDARDISED

Our raw features have variances of 1.3e-4 and 2.9e-5. hmmlearn's defaults are
calibrated for data on a scale of order 1: min_covar=0.001 and covars_prior=0.01
are respectively 35x and 350x LARGER than our smallest variance. The prior
therefore dominates the data it is meant to regularise.

Measured on the training set, 3 states, full covariance, 20 restarts:

    raw features, hmmlearn defaults          8/20 restarts CRASHED
                                             (covars not positive-definite)
    raw features, hand-tuned tiny priors     0/20 crashed, logL 7856
    standardised features, defaults          0/20 crashed

Standardising to mean 0, sd 1 puts the data in the range hmmlearn's defaults
expect, and removes the need for scale-specific magic numbers that would have
to be re-tuned whenever a feature changes.

It also settles the Note 02 problem completely. There we kept ANNUALISE_VOLATILITY
off so the two features stayed *comparable* (a 2.14x ratio). Standardisation makes
them *identical* in scale (1.00x), so k-means initialisation weights neither
feature over the other. As a bonus the annualisation question disappears
altogether: standardising after multiplying a column by any constant gives
exactly the same numbers, because the constant cancels in (cx - mean)/sd.

THE SCALER IS FITTED ON TRAINING DATA ONLY

Its mean and standard deviation are statistics computed FROM data. Computing
them over the full series would fold knowledge of the test period into every
training row — the textbook leakage case, and the exact contrast we drew when
deciding it was fine to compute the rolling window before splitting. A
backward-looking window at time t uses only the past; a dataset-wide mean does not.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from src import config
from src.features import build_observation_matrix, load_dataset


# ===========================================================================
# Task 3.1 — chronological split
# ===========================================================================
def split_chronologically(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split at config.TRAIN_END_DATE. Train is strictly before test."""
    train = frame.loc[frame.index < config.TRAIN_END_DATE]
    test = frame.loc[frame.index >= config.TRAIN_END_DATE]

    if train.empty or test.empty:
        raise ValueError(
            f"TRAIN_END_DATE={config.TRAIN_END_DATE} produced an empty split "
            f"(train={len(train)}, test={len(test)})."
        )
    # The guarantee the whole evaluation rests on: cheap to assert, disastrous
    # to get wrong silently.
    if train.index.max() >= test.index.min():
        raise ValueError("Train and test overlap in time — the split is not chronological.")

    return train, test


# ===========================================================================
# The fitted artefact — scaler and HMM travel together
# ===========================================================================
@dataclass
class RegimeModel:
    """A fitted HMM plus the scaler it was trained with.

    These must never be separated. An HMM fitted on standardised features will
    produce nonsense if handed raw ones, so the object that gets serialised for
    the Streamlit app carries both.
    """

    hmm: GaussianHMM
    scaler: StandardScaler
    feature_columns: list[str] = field(default_factory=lambda: list(config.FEATURE_COLUMNS))
    restart_scores: np.ndarray | None = None

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """DataFrame -> the standardised observation matrix the HMM expects."""
        return self.scaler.transform(build_observation_matrix(frame))

    def score(self, frame: pd.DataFrame) -> float:
        """Forward algorithm: log P(observations | model)."""
        return self.hmm.score(self.transform(frame))

    def means_original_units(self) -> np.ndarray:
        """Fitted state means, converted back out of standardised units."""
        return self.scaler.inverse_transform(self.hmm.means_)

    def covariances_original_units(self) -> np.ndarray:
        """Fitted covariances, converted back. Sigma_orig = D Sigma_std D."""
        scale = np.diag(self.scaler.scale_)
        return np.array([scale @ cov @ scale for cov in self.hmm.covars_])

    # --- Task 3.4 — decoding ----------------------------------------------
    def decode(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the input frame with the decoded regime and its posterior.

        Two different questions are answered here, and the difference matters:

        `hmm.predict`      Viterbi. The single most likely SEQUENCE of states.
                           It maximises the probability of the whole path, so the
                           transition term penalises switching and one violent day
                           inside a calm regime does not flip the label.

        `hmm.predict_proba` Forward-backward. P(state_t = i | the ENTIRE series),
                           independently per day. A confidence, not a path.

        These can disagree, and legitimately so: Viterbi may keep a day in state 0
        because leaving and returning costs more than the one day gains, even when
        that day's own posterior favours state 1. Days where they disagree are the
        regime boundaries — the genuinely ambiguous ones.
        """
        observations = self.transform(frame)

        out = frame.copy()
        out["regime"] = self.hmm.predict(observations)          # Viterbi path
        posteriors = self.hmm.predict_proba(observations)       # forward-backward

        for i in range(self.hmm.n_components):
            out[f"prob_state_{i}"] = posteriors[:, i]
        out["confidence"] = posteriors.max(axis=1)
        out["argmax_state"] = posteriors.argmax(axis=1)
        out["viterbi_differs"] = out["regime"] != out["argmax_state"]

        return out


# ===========================================================================
# Tasks 3.2 / 3.3 — fitting
# ===========================================================================
def fit_with_restarts(
    observations: np.ndarray,
    n_components: int | None = None,
    covariance_type: str | None = None,
    n_restarts: int | None = None,
    verbose: bool = True,
) -> tuple[GaussianHMM, np.ndarray]:
    """Fit n_restarts times from different starts; return the best model and all scores."""
    n_components = n_components or config.N_REGIMES
    covariance_type = covariance_type or config.COVARIANCE_TYPE
    n_restarts = n_restarts or config.N_RESTARTS

    best_model, best_score = None, -np.inf
    scores, converged, failures = [], 0, 0

    for seed in range(n_restarts):
        candidate = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=config.N_ITER,
            tol=config.TOL,
            random_state=config.RANDOM_SEED + seed,
        )
        try:
            # hmmlearn warns when EM stops on a delta below floating-point
            # precision. Near the optimum that is a stopping signal, not a
            # failure, and 20 of them would drown the output.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                candidate.fit(observations)
            score = candidate.score(observations)
        except (ValueError, np.linalg.LinAlgError):
            # A degenerate restart — usually a state that collapsed onto too few
            # points. Discard it and keep going; that is what restarts are for.
            failures += 1
            continue

        scores.append(score)
        converged += bool(candidate.monitor_.converged)
        if score > best_score:
            best_model, best_score = candidate, score

    if best_model is None:
        raise RuntimeError(
            f"All {n_restarts} restarts failed for {n_components} states / "
            f"'{covariance_type}' covariance."
        )

    scores = np.asarray(scores)

    if verbose:
        distinct = len(np.unique(scores.round(2)))
        print(f"\n--- Fit: {n_components} states, '{covariance_type}' covariance, "
              f"{n_restarts} restarts ---")
        print(f"  successful restarts : {len(scores)}/{n_restarts}"
              + (f"   ({failures} degenerate, discarded)" if failures else ""))
        print(f"  converged           : {converged}/{len(scores)}")
        print(f"  best log-likelihood : {scores.max():.2f}")
        print(f"  worst               : {scores.min():.2f}")
        print(f"  spread              : {scores.max() - scores.min():.2f}")
        print(f"  distinct optima     : {distinct}"
              "   <- 1 means every restart reached the same solution")
        if distinct > 1:
            at_best = int((scores.round(2) == round(scores.max(), 2)).sum())
            print(f"  restarts at the best: {at_best}/{len(scores)}")

    return best_model, scores


def fit_regime_model(train: pd.DataFrame, verbose: bool = True) -> RegimeModel:
    """Fit the scaler on training data, then the HMM on the scaled features."""
    scaler = StandardScaler().fit(build_observation_matrix(train))
    x_train = scaler.transform(build_observation_matrix(train))

    hmm, scores = fit_with_restarts(x_train, verbose=verbose)
    return RegimeModel(hmm=hmm, scaler=scaler, restart_scores=scores)


# ===========================================================================
# Model selection — is 3 states actually the right number?
# ===========================================================================
def compare_models(
    observations: np.ndarray,
    state_counts=(2, 3, 4, 5),
    covariance_types=("full", "diag"),
    n_restarts: int = 10,
) -> pd.DataFrame:
    """Score candidate configurations by BIC. Lower is better.

    BIC = -2 * logL + k * ln(n), where k is the number of free parameters. The
    penalty is what stops us adding states forever: every extra state fits the
    training data better, and BIC charges for the privilege.

    Valid only because every candidate sees IDENTICALLY scaled features — a
    rescaled feature set shifts logL by a constant and makes BIC incomparable.
    """
    rows = []
    for covariance_type in covariance_types:
        for k in state_counts:
            try:
                model, scores = fit_with_restarts(
                    observations, k, covariance_type, n_restarts, verbose=False
                )
            except RuntimeError:
                continue
            rows.append({
                "states": k,
                "covariance": covariance_type,
                "logL": model.score(observations),
                "AIC": model.aic(observations),
                "BIC": model.bic(observations),
                "restart_spread": scores.max() - scores.min(),
            })

    table = pd.DataFrame(rows).sort_values("BIC").reset_index(drop=True)
    table["dBIC"] = table["BIC"] - table["BIC"].min()
    return table


# ===========================================================================
# Reporting
# ===========================================================================
def describe_model(model: RegimeModel) -> pd.DataFrame:
    """Per-state summary in ORIGINAL units: emission means, persistence, duration."""
    means = model.means_original_units()
    hmm = model.hmm

    rows = []
    for i in range(hmm.n_components):
        persistence = hmm.transmat_[i, i]
        rows.append({
            "state": i,
            "mean_return": means[i, 0],
            "ann_return_%": means[i, 0] * config.TRADING_DAYS_PER_YEAR * 100,
            "mean_volatility": means[i, 1],
            "ann_vol_%": means[i, 1] * np.sqrt(config.TRADING_DAYS_PER_YEAR) * 100,
            "persistence": persistence,
            "exp_duration_days": 1.0 / (1.0 - persistence) if persistence < 1 else np.inf,
        })
    return pd.DataFrame(rows)


# ===========================================================================
# Task 3.5 — naming the states from the fitted parameters
# ===========================================================================
GATEWAY_THRESHOLD = 0.005
# A transition probability below this counts as "effectively never happens".


def label_states(model: RegimeModel, verbose: bool = True) -> dict[int, str]:
    """Derive human names for the fitted states. NEVER hardcoded.

    EM's likelihood is invariant to permuting state labels, so state index 0
    carries no meaning — all n! orderings score identically and which one you get
    is decided by the random initialisation. Names must therefore be read off the
    fitted parameters at runtime.

    The rules, applied in order:

      1. Highest mean return              -> "Bull"
      2. Lowest mean return               -> "Bear" if that return is negative,
                                             otherwise "High Volatility"
      3. Anything in between              -> "Transition" if it is the only route
                                             between the two extremes, else "Sideways"

    Rule 3 is a claim about the transition matrix, not a default. If A[bull, bear]
    and A[bear, bull] are both effectively zero, the market cannot move between the
    extremes without passing through the middle state — which makes "Transition" a
    description of measured structure rather than a convenient word.
    """
    stats = describe_model(model).sort_values("ann_return_%", ascending=False)
    transmat = model.hmm.transmat_

    top = int(stats.iloc[0]["state"])
    bottom = int(stats.iloc[-1]["state"])
    middles = [int(s) for s in stats.iloc[1:-1]["state"]]

    labels: dict[int, str] = {top: "Bull"}
    reasons: dict[int, str] = {top: "highest mean return"}

    bottom_return = stats.iloc[-1]["ann_return_%"]
    if bottom_return < 0:
        labels[bottom] = "Bear"
        reasons[bottom] = f"lowest mean return, and it is negative ({bottom_return:+.1f}%/yr)"
    else:
        labels[bottom] = "High Volatility"
        reasons[bottom] = (f"lowest mean return but still positive ({bottom_return:+.1f}%/yr), "
                           "so named for its volatility rather than its direction")

    is_gateway = (
        transmat[top, bottom] < GATEWAY_THRESHOLD
        and transmat[bottom, top] < GATEWAY_THRESHOLD
    )
    for state in middles:
        if is_gateway and len(middles) == 1:
            labels[state] = "Transition"
            reasons[state] = (
                f"the only route between the extremes: A[{top}->{bottom}]="
                f"{transmat[top, bottom]:.4f} and A[{bottom}->{top}]="
                f"{transmat[bottom, top]:.4f}, both below {GATEWAY_THRESHOLD}"
            )
        else:
            labels[state] = "Sideways"
            reasons[state] = "intermediate return, and not a required gateway"

    if verbose:
        print("\n--- Task 3.5 regime labelling (derived, not hardcoded) ---")
        for state in sorted(labels):
            row = stats[stats["state"] == state].iloc[0]
            print(f"  state {state} -> {labels[state]:<16}"
                  f" ({row['ann_return_%']:+6.1f}%/yr, {row['ann_vol_%']:5.1f}% vol)")
            print(f"          because {reasons[state]}")

    return labels


def regime_episodes(decoded: pd.DataFrame, labels: dict[int, str]) -> pd.DataFrame:
    """Collapse the day-by-day regime path into contiguous episodes."""
    regime = decoded["regime"]
    block = (regime != regime.shift()).cumsum()

    episodes = decoded.groupby(block).agg(
        regime=("regime", "first"),
        start=("regime", lambda s: s.index[0]),
        end=("regime", lambda s: s.index[-1]),
        days=("regime", "size"),
    )
    episodes["label"] = episodes["regime"].map(labels)
    episodes["return_%"] = [
        (decoded.loc[a:b, "log_return"].sum()) * 100
        for a, b in zip(episodes["start"], episodes["end"])
    ]
    return episodes.reset_index(drop=True)


# ===========================================================================
# Export for the Streamlit app (Task 4.2 input)
# ===========================================================================
def export_model(model: RegimeModel, labels: dict[int, str]) -> None:
    """Serialise the model, its scaler and the label map as a single object."""
    import joblib

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "regime_model": model,
            "labels": labels,
            "feature_columns": list(config.FEATURE_COLUMNS),
            "train_end_date": config.TRAIN_END_DATE,
            "n_regimes": config.N_REGIMES,
            "covariance_type": config.COVARIANCE_TYPE,
        },
        config.MODEL_FILE,
    )
    print(f"\n  exported -> {config.MODEL_FILE}")


def load_exported_model() -> dict:
    """Load what export_model saved. This is what app.py calls."""
    import joblib

    if not config.MODEL_FILE.exists():
        raise FileNotFoundError(
            f"{config.MODEL_FILE} not found — run 'python -m src.model' first."
        )
    return joblib.load(config.MODEL_FILE)


def print_transition_matrix(model: RegimeModel) -> None:
    hmm = model.hmm
    print("\n  Transition matrix A (row = from, column = to):")
    print("           " + "".join(f"   to {j}  " for j in range(hmm.n_components)))
    for i in range(hmm.n_components):
        cells = "".join(f"  {hmm.transmat_[i, j]:7.4f}" for j in range(hmm.n_components))
        print(f"    from {i} {cells}")


if __name__ == "__main__":
    dataset = load_dataset()
    train, test = split_chronologically(dataset)

    print("--- Task 3.1 chronological split ---")
    print(f"  train : {len(train):4d} rows  {train.index.min().date()} to {train.index.max().date()}")
    print(f"  test  : {len(test):4d} rows  {test.index.min().date()} to {test.index.max().date()}")
    print(f"  no overlap verified : {train.index.max() < test.index.min()}")

    scaler = StandardScaler().fit(build_observation_matrix(train))
    x_train = scaler.transform(build_observation_matrix(train))
    print(f"\n  scaler fitted on TRAIN ONLY")
    print(f"    centre : {scaler.mean_}")
    print(f"    scale  : {scaler.scale_}")
    print(f"    scaled feature sds : {x_train.std(axis=0)}   <- both exactly 1.00")

    print("\n\n=== MODEL SELECTION (training data only) ===")
    comparison = compare_models(x_train)
    print(comparison.to_string(
        index=False,
        formatters={
            "logL": "{:.1f}".format, "AIC": "{:.1f}".format, "BIC": "{:.1f}".format,
            "dBIC": "{:.1f}".format, "restart_spread": "{:.2f}".format,
        },
    ))

    print("\n\n=== FITTING THE CONFIGURED MODEL ===")
    regime_model = fit_regime_model(train)

    print("\n--- Fitted states (indices arbitrary until Task 3.5 names them) ---")
    print(describe_model(regime_model).to_string(index=False, float_format="{:.6f}".format))
    print_transition_matrix(regime_model)

    train_ll, test_ll = regime_model.score(train), regime_model.score(test)
    print(f"\n  train logL : {train_ll:10.2f}  over {len(train)} rows"
          f"   -> {train_ll / len(train):+.4f} per observation")
    print(f"  test  logL : {test_ll:10.2f}  over {len(test)} rows"
          f"   -> {test_ll / len(test):+.4f} per observation  (never seen during fitting)")

    # --- Task 3.5: name the states before decoding, so output is readable ---
    labels = label_states(regime_model)

    # --- Task 3.4: decode the FULL series -------------------------------
    # Fitting used training data only. Decoding runs over everything, which is
    # legitimate: Viterbi applies an already-fitted model, and the test year is
    # genuinely out-of-sample inference rather than training.
    decoded = regime_model.decode(dataset)
    decoded["label"] = decoded["regime"].map(labels)
    decoded.to_csv(config.PROCESSED_DIR / "decoded_regimes.csv")

    print("\n--- Task 3.4 Viterbi decoding (full series) ---")
    print(f"  rows decoded        : {len(decoded)}")
    print(f"  mean confidence     : {decoded['confidence'].mean():.3f}")
    print(f"  days below 0.60     : {int((decoded['confidence'] < 0.60).sum())}"
          f"  ({(decoded['confidence'] < 0.60).mean():.1%})")
    print(f"  Viterbi vs argmax   : disagree on {int(decoded['viterbi_differs'].sum())} days"
          "   <- the smoothing at work; these are boundary days")

    print("\n  Share of days by regime:")
    for state in sorted(labels):
        mask = decoded["regime"] == state
        print(f"    {labels[state]:<16} {mask.mean():5.1%}"
              f"   mean confidence {decoded.loc[mask, 'confidence'].mean():.3f}")

    episodes = regime_episodes(decoded, labels)
    print(f"\n--- Regime episodes: {len(episodes)} over five years ---")
    print("  Ten longest:")
    longest = episodes.nlargest(10, "days")
    for _, row in longest.iterrows():
        print(f"    {row['start'].date()} to {row['end'].date()}  "
              f"{row['days']:4d}d  {row['label']:<16} {row['return_%']:+7.2f}%")

    print("\n  Median episode length by regime:")
    for state in sorted(labels):
        lengths = episodes.loc[episodes["regime"] == state, "days"]
        print(f"    {labels[state]:<16} median {lengths.median():5.1f}d"
              f"   longest {lengths.max():4d}d   episodes {len(lengths)}")

    # --- Sanity check against events the model was never told about ------
    print("\n--- Sanity check: what did the model call known episodes? ---")
    for name, start, end in [
        ("2022 bear market  ", "2022-01-03", "2022-10-14"),
        ("2023 recovery     ", "2023-03-01", "2023-07-31"),
        ("Apr 2025 shock    ", "2025-03-25", "2025-05-15"),
        ("test year overall ", config.TRAIN_END_DATE, "2026-08-31"),
    ]:
        window = decoded.loc[start:end]
        if window.empty:
            continue
        shares = window["label"].value_counts(normalize=True)
        summary = "  ".join(f"{lab} {share:.0%}" for lab, share in shares.items())
        print(f"  {name}: {summary}")

    export_model(regime_model, labels)
