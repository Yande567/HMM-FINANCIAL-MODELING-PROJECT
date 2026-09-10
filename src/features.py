"""
Sprint Tasks 1.5 and 1.6 — rolling volatility and the observation matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data_loader import load_prices
from src.preprocessing import prepare


# ===========================================================================
# Task 1.5 — rolling volatility
# ===========================================================================
def add_rolling_volatility(frame: pd.DataFrame) -> pd.DataFrame:
    """Append the rolling standard deviation of log returns."""
    out = frame.copy()

    # pandas defaults to min_periods=window, so the first (window-1) rows come
    # back as NaN. That default is correct and must not be overridden:
    # min_periods=1 would compute a "standard deviation" from two observations
    # and return a confidently wrong, downward-biased number instead of an
    # honest NaN.
    out["volatility"] = out["log_return"].rolling(
        window=config.VOLATILITY_WINDOW
    ).std()  # ddof=1 (sample sd) by default — correct for an estimate

    if config.ANNUALISE_VOLATILITY:
        out["volatility"] *= np.sqrt(config.TRADING_DAYS_PER_YEAR)

    return out


def drop_warmup(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the leading rows with no complete volatility window."""
    before = len(frame)
    out = frame.dropna(subset=config.FEATURE_COLUMNS)
    dropped = before - len(out)

    expected = config.VOLATILITY_WINDOW - 1
    if dropped != expected:
        raise ValueError(
            f"Expected to drop exactly {expected} warm-up rows, dropped {dropped}. "
            "This means NaNs appeared somewhere other than the window warm-up — "
            "investigate before continuing."
        )
    return out


# ===========================================================================
# Task 1.6 — observation matrix
# ===========================================================================
def build_observation_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Return the (T, 2) array hmmlearn consumes.

    Column order follows config.FEATURE_COLUMNS, which fixes the meaning of
    model.means_[i][0] (return) and [i][1] (volatility) for every downstream
    interpretation, including the Task 3.5 regime labelling.
    """
    matrix = frame.loc[:, config.FEATURE_COLUMNS].to_numpy(dtype=np.float64)

    if matrix.ndim != 2 or matrix.shape[1] != len(config.FEATURE_COLUMNS):
        raise ValueError(f"Expected a (T, {len(config.FEATURE_COLUMNS)}) matrix, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Observation matrix contains NaN or inf")

    return np.ascontiguousarray(matrix)


# ===========================================================================
# Pipeline entry point
# ===========================================================================
def build_dataset(save: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Tasks 1.2 - 1.6 end to end: download, validate, returns, volatility, matrix."""
    prepared = prepare(load_prices(), verbose=verbose)
    features = drop_warmup(add_rolling_volatility(prepared))

    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        features.to_csv(config.PROCESSED_CSV)
        if verbose:
            print(f"\n  saved -> {config.PROCESSED_CSV}")

    if verbose:
        _describe_features(features)

    return features


def load_dataset() -> pd.DataFrame:
    """Read the processed dataset saved by build_dataset()."""
    if not config.PROCESSED_CSV.exists():
        raise FileNotFoundError(
            f"{config.PROCESSED_CSV} not found — run 'python -m src.features' first."
        )
    return pd.read_csv(config.PROCESSED_CSV, index_col="Date", parse_dates=True)


def _describe_features(frame: pd.DataFrame) -> None:
    matrix = build_observation_matrix(frame)
    returns, volatility = frame["log_return"], frame["volatility"]

    print("\n--- Task 1.5 rolling volatility ---")
    print(f"  window              : {config.VOLATILITY_WINDOW} trading days"
          f"{' (annualised)' if config.ANNUALISE_VOLATILITY else ' (daily scale)'}")
    print(f"  mean                : {volatility.mean():.6f}")
    print(f"  min / max           : {volatility.min():.6f} / {volatility.max():.6f}")
    print(f"  max / min ratio     : {volatility.max() / volatility.min():.1f}x"
          "   <- the separation the model exploits")
    print(f"  calmest 10 days end : {volatility.idxmin().date()}")
    print(f"  wildest 10 days end : {volatility.idxmax().date()}")

    print("\n--- Feature scales (both must stay comparable) ---")
    print(f"  log_return  sd      : {returns.std():.6f}")
    print(f"  volatility  sd      : {volatility.std():.6f}")
    print(f"  ratio               : {max(returns.std(), volatility.std()) / min(returns.std(), volatility.std()):.2f}x")

    # Signed return vs volatility is ~0 by construction: volatility is a
    # MAGNITUDE, and up-days and down-days both raise it, so the sign cancels.
    # The informative pair is |return| vs volatility. The leverage effect (falls
    # raise FUTURE volatility) shows up only at a lag.
    print(f"\n  corr(return,  vol)  : {returns.corr(volatility):+.3f}"
          "   <- ~0 expected: sign cancels")
    print(f"  corr(|return|, vol) : {returns.abs().corr(volatility):+.3f}"
          "   <- magnitude relationship")
    print(f"  corr(return, vol+10): {returns.corr(volatility.shift(-10)):+.3f}"
          "   <- leverage effect, appears at a lag")

    print("\n--- Task 1.6 observation matrix ---")
    print(f"  shape               : {matrix.shape}")
    print(f"  dtype               : {matrix.dtype}")
    print(f"  C-contiguous        : {matrix.flags['C_CONTIGUOUS']}")
    print(f"  columns             : {config.FEATURE_COLUMNS}")
    print(f"  date range          : {frame.index.min().date()} to {frame.index.max().date()}")
    print(f"  first row           : [{matrix[0, 0]:+.6f}, {matrix[0, 1]:.6f}]")
    print(f"  last row            : [{matrix[-1, 0]:+.6f}, {matrix[-1, 1]:.6f}]")

    train = frame.loc[: config.TRAIN_END_DATE]
    test = frame.loc[config.TRAIN_END_DATE:]
    print(f"\n  chronological split at {config.TRAIN_END_DATE} (Task 3.1 preview):")
    print(f"    train : {len(train):4d} rows  {train.index.min().date()} to {train.index.max().date()}")
    print(f"    test  : {len(test):4d} rows  {test.index.min().date()} to {test.index.max().date()}")


if __name__ == "__main__":
    build_dataset()
