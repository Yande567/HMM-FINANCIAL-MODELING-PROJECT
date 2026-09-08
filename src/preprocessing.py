"""
Sprint Tasks 1.3 and 1.4 — validate the raw download, then compute log returns.

THE CENTRAL DECISION: NON-TRADING DAYS ARE NOT FILLED IN.

The obvious-looking move is to reindex onto a continuous calendar and
forward-fill the gaps:

    df = df.asfreq("D").ffill()          # <-- DO NOT DO THIS

That inserts roughly 660 fake rows into a 1,254-row dataset, every one of them
carrying a return of exactly zero. Three things break at once:

  1. The volatility feature collapses. A 10-day rolling standard deviation
     computed over a window that is 40% artificial zeros understates real
     volatility badly, and unevenly — a window spanning a long weekend is
     diluted more than one that does not.
  2. The transition matrix inflates. More rows inside each regime raises every
     diagonal entry A[i,i], so the model reports regimes lasting far longer
     than they truly do. Since we quote expected duration as 1/(1-A[i,i]),
     the headline number is directly corrupted.
  3. A third Gaussian state appears out of nowhere. A spike of hundreds of
     exactly-zero returns is a distribution the model will happily allocate a
     state to — a "regime" that is really just the weekend.

An HMM has no notion of calendar time. It sees an ordered SEQUENCE of
observations, nothing more. Trading days are already that sequence, so the
correct handling of a non-trading day is to leave it absent.

The cost of that choice, which belongs in the write-up: a Friday-to-Monday
return spans three calendar days but is treated identically to a
Tuesday-to-Wednesday return. Monday returns therefore carry slightly more
variance than the model assumes. This is a known and accepted simplification;
the alternative (a time-inhomogeneous transition matrix) is well beyond scope.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data_loader import _resolve_price_column


class DataValidationError(Exception):
    """Raised when the raw data has a defect that would corrupt the model."""


# ===========================================================================
# Task 1.3 — validation
# ===========================================================================
def validate(frame: pd.DataFrame, verbose: bool = True) -> dict:
    """Audit the raw frame. Raises on fatal defects, warns on suspicious ones."""
    price_column = _resolve_price_column(frame)
    prices = frame[price_column]

    fatal: list[str] = []
    warnings: list[str] = []

    # --- Fatal: these would silently produce wrong returns -----------------
    if frame.index.duplicated().any():
        dupes = frame.index[frame.index.duplicated()].tolist()
        fatal.append(f"{len(dupes)} duplicate date(s), e.g. {dupes[:3]}")

    if not frame.index.is_monotonic_increasing:
        fatal.append("index is not sorted ascending — returns would be scrambled")

    weekend = frame.index[frame.index.dayofweek >= 5]
    if len(weekend):
        fatal.append(f"{len(weekend)} weekend row(s), e.g. {weekend[:3].tolist()} "
                     "— equity markets do not trade; these are filled data")

    if prices.isna().any():
        fatal.append(f"{int(prices.isna().sum())} missing value(s) in {price_column}")

    if (prices <= 0).any():
        fatal.append(f"{int((prices <= 0).sum())} non-positive price(s) — log() undefined")

    # --- Warnings: worth knowing, not worth stopping for -------------------
    gaps = frame.index.to_series().diff().dt.days.dropna().astype(int)
    holiday_gaps = int((gaps > 3).sum() + (gaps == 2).sum())

    stale = int((prices.diff() == 0).sum())
    if stale:
        warnings.append(f"{stale} day(s) with a price identical to the previous day "
                        "— possible forward-filling upstream")

    if "Volume" in frame.columns:
        zero_volume = frame.index[frame["Volume"] == 0]
        if len(zero_volume):
            warnings.append(
                f"{len(zero_volume)} day(s) reporting zero volume: "
                f"{[str(d.date()) for d in zero_volume]}. Prices on these days are "
                "intact, so returns are unaffected — do NOT drop the rows."
            )

    # OHLC internal consistency, if those columns are present
    if {"High", "Low", "Open", "Close"} <= set(frame.columns):
        bad_high = int((frame["High"] < frame[["Open", "Close"]].max(axis=1) - 1e-6).sum())
        bad_low = int((frame["Low"] > frame[["Open", "Close"]].min(axis=1) + 1e-6).sum())
        if bad_high or bad_low:
            warnings.append(f"OHLC inconsistency: {bad_high} High too low, {bad_low} Low too high")

    report = {
        "rows": len(frame),
        "first_date": frame.index.min(),
        "last_date": frame.index.max(),
        "price_column": price_column,
        "trading_days_per_year": frame.groupby(frame.index.year).size().to_dict(),
        "gap_counts": gaps.value_counts().sort_index().to_dict(),
        "holiday_closures": holiday_gaps,
        "fatal": fatal,
        "warnings": warnings,
    }

    if verbose:
        _print_report(report)

    if fatal:
        raise DataValidationError(
            "Raw data failed validation:\n  - " + "\n  - ".join(fatal)
        )
    return report


def _print_report(report: dict) -> None:
    print("\n--- Task 1.3 validation ---")
    print(f"  rows                : {report['rows']}")
    print(f"  range               : {report['first_date'].date()} to {report['last_date'].date()}")
    print(f"  price column        : {report['price_column']}")
    print(f"  rows per year       : {report['trading_days_per_year']}")
    print(f"  calendar gap sizes  : {report['gap_counts']}"
          "   (1=consecutive, 3=weekend, 2 or 4=holiday)")
    print(f"  holiday closures    : {report['holiday_closures']}")
    for warning in report["warnings"]:
        print(f"  WARNING: {warning}")
    if not report["fatal"]:
        print("  no fatal defects")


# ===========================================================================
# Task 1.3 — cleaning
# ===========================================================================
def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce the raw OHLCV frame to a single validated price series.

    Deliberately does NOT reindex onto a calendar — see the module docstring.
    """
    price_column = _resolve_price_column(frame)

    cleaned = frame.loc[:, [price_column]].rename(columns={price_column: "price"})
    cleaned = cleaned[~cleaned.index.duplicated(keep="first")].sort_index()

    # dropna() here removes rows the source never had data for. It is not a
    # substitute for validation: by this point validate() has already
    # established there are none, so this is a guard, not a repair.
    before = len(cleaned)
    cleaned = cleaned.dropna()
    if len(cleaned) < before:
        print(f"  dropped {before - len(cleaned)} row(s) with missing price")

    return cleaned


# ===========================================================================
# Task 1.4 — log returns
# ===========================================================================
def add_log_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Append r_t = ln(P_t / P_t-1) and drop the undefined first row."""
    out = frame.copy()
    out["log_return"] = np.log(out["price"]).diff()

    # The first row has no previous price, so its return is NaN by definition.
    # Dropping it is correct; filling it with 0 would invent a flat day that
    # never happened and drag the first volatility window towards zero.
    return out.dropna(subset=["log_return"])


def prepare(frame: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Full Task 1.3 + 1.4 pipeline: validate, clean, add log returns."""
    validate(frame, verbose=verbose)
    prepared = add_log_returns(clean(frame))
    if verbose:
        _describe_returns(prepared["log_return"])
    return prepared


def _describe_returns(returns: pd.Series) -> None:
    annualised = returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    print("\n--- Task 1.4 log returns ---")
    print(f"  observations        : {len(returns)}")
    print(f"  mean (daily)        : {returns.mean():+.6f}")
    print(f"  sd (daily)          : {returns.std():.6f}")
    print(f"  sd (annualised)     : {annualised * 100:.1f}%")
    print(f"  skewness            : {returns.skew():+.3f}   (0 for a normal distribution)")
    print(f"  excess kurtosis     : {returns.kurtosis():+.3f}   (0 for a normal distribution)")
    print(f"  min / max           : {returns.min() * 100:+.2f}% / {returns.max() * 100:+.2f}%")

    # Under a normal distribution, |r| > 3 sd should occur on ~0.27% of days.
    beyond_3sd = int((returns.abs() > 3 * returns.std()).sum())
    expected = 0.0027 * len(returns)
    print(f"  |r| > 3 sd          : {beyond_3sd} days observed vs {expected:.1f} expected "
          "under a normal — evidence of fat tails (Phase 2 quantifies this)")


if __name__ == "__main__":
    from src.data_loader import load_prices

    prepare(load_prices())
