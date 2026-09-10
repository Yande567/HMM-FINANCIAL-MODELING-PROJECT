"""
Sprint Task 1.2 — acquire 5 years of S&P 500 (^GSPC) daily data.

Cache first, network second. yfinance scrapes unofficial Yahoo endpoints that
rate-limit and break without notice, so the pipeline must be runnable from
data/raw/gspc_raw.csv alone.

The cache is validated, not just checked for existence. A metadata sidecar
records the exact request that produced the CSV; if config no longer matches
it, the cache is refused and re-downloaded. Without this, changing START_DATE
would silently return the old date range and every downstream number would
describe an experiment that was never run.

Usage:
    from src.data_loader import load_prices
    prices = load_prices()                    # cache if valid, else download
    prices = load_prices(force_refresh=True)  # always re-download

    python -m src.data_loader                 # from the project root
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from src import config

METADATA_FILE = config.RAW_DIR / "gspc_raw.meta.json"


# ===========================================================================
# Network layer — the only function in the project that touches the internet
# ===========================================================================
def _fetch_from_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV bars. Isolated so the rest of the module is testable."""
    # auto_adjust=False is deliberate: yfinance changed this default to True,
    # which drops 'Adj Close' and returns adjusted values in 'Close' instead.
    frame = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
    )
    if frame is None or frame.empty:
        raise RuntimeError(
            f"yfinance returned no rows for {ticker} between {start} and {end}.\n"
            "  1. Check the internet connection / proxy.\n"
            "  2. Yahoo may be rate-limiting this IP — wait and retry.\n"
            "  3. The scraper may be broken: pip install --upgrade yfinance"
        )
    return frame


# ===========================================================================
# Shape normalisation
# ===========================================================================
def _normalise_columns(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten yfinance's ('Adj Close', '^GSPC') MultiIndex to plain names."""
    frame = frame.copy()

    if isinstance(frame.columns, pd.MultiIndex):
        # Find the level holding the ticker rather than assuming it is level 1,
        # since yfinance's group_by option can swap them.
        ticker_levels = [
            i for i in range(frame.columns.nlevels)
            if ticker in frame.columns.get_level_values(i)
        ]
        frame.columns = frame.columns.droplevel(ticker_levels[0] if ticker_levels else 1)

    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "Date"
    return frame.sort_index()


def _resolve_price_column(frame: pd.DataFrame) -> str:
    """Return the price column to use, tolerating yfinance's auto_adjust modes."""
    if config.PRICE_COLUMN in frame.columns:
        return config.PRICE_COLUMN
    if "Close" in frame.columns:
        print(f"  WARNING: '{config.PRICE_COLUMN}' absent; using 'Close' ")
        return "Close"
    raise KeyError(
        f"Neither '{config.PRICE_COLUMN}' nor 'Close' found. "
        f"Columns present: {list(frame.columns)}"
    )


# ===========================================================================
# Cache layer
# ===========================================================================
def _current_request() -> dict:
    """The request config currently describes, compared against cached metadata."""
    return {
        "ticker": config.TICKER,
        "start_date": config.START_DATE,
        "end_date": config.END_DATE,
    }


def _write_cache(frame: pd.DataFrame) -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.RAW_PRICES_CSV)

    metadata = _current_request() | {
        "rows": int(len(frame)),
        "first_date": str(frame.index.min().date()),
        "last_date": str(frame.index.max().date()),
        "columns": list(frame.columns),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "pandas_version": pd.__version__,
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _read_cache() -> pd.DataFrame | None:
    """Return the cached frame, or None if absent or stale."""
    if not (config.RAW_PRICES_CSV.exists() and METADATA_FILE.exists()):
        return None

    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in _current_request().items()
        if metadata.get(key) != value
    }
    if mismatches:
        print("  Cache is STALE — config no longer matches what was downloaded:")
        for key, (cached, requested) in mismatches.items():
            print(f"    {key}: cached={cached!r}  requested={requested!r}")
        return None

    # index_col + parse_dates is what preserves the DatetimeIndex. A CSV stores
    # no dtypes, so without them the index returns as strings and every later
    # date slice and resample silently misbehaves.
    frame = pd.read_csv(config.RAW_PRICES_CSV, index_col="Date", parse_dates=True)
    print(f"  Loaded cache: {len(frame)} rows, "
          f"{frame.index.min().date()} to {frame.index.max().date()} "
          f"(downloaded {metadata.get('downloaded_at_utc', 'unknown')})")
    return frame


# ===========================================================================
# Public entry point
# ===========================================================================
def load_prices(force_refresh: bool = False) -> pd.DataFrame:
    """Return daily ^GSPC bars for the configured window, indexed by date."""
    print(f"Loading {config.TICKER} from {config.START_DATE} to {config.END_DATE}")

    if force_refresh:
        print("  force_refresh=True — bypassing cache")
    else:
        cached = _read_cache()
        if cached is not None:
            return cached

    print("  Downloading from Yahoo Finance...")
    frame = _normalise_columns(
        _fetch_from_yahoo(config.TICKER, config.START_DATE, config.END_DATE),
        config.TICKER,
    )
    _write_cache(frame)
    print(f"  Downloaded and cached {len(frame)} rows -> {config.RAW_PRICES_CSV}")
    return frame


def describe(frame: pd.DataFrame) -> None:
    """Print a short acceptance report. Task 1.3 does the real validation."""
    price_column = _resolve_price_column(frame)
    prices = frame[price_column]
    span_years = (frame.index.max() - frame.index.min()).days / 365.25

    print("\n--- Acquisition summary ---")
    print(f"  rows (trading days) : {len(frame)}")
    print(f"  date range          : {frame.index.min().date()} to {frame.index.max().date()}")
    print(f"  calendar span       : {span_years:.2f} years")
    print(f"  expected ~252/yr    : {span_years * config.TRADING_DAYS_PER_YEAR:.0f} rows")
    print(f"  price column used   : {price_column}")
    print(f"  price range         : {prices.min():,.2f} to {prices.max():,.2f}")
    print(f"  missing values      : {int(frame.isna().sum().sum())}")
    print(f"  duplicate dates     : {int(frame.index.duplicated().sum())}")
    print(f"  monotonic index     : {frame.index.is_monotonic_increasing}")


if __name__ == "__main__":
    describe(load_prices())
