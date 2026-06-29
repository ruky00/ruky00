"""
Data utilities.

The framework is designed to run completely offline, so the default data
source is a regime-aware synthetic OHLCV generator. It stitches together
bull / bear / sideways segments with different drift and volatility, which
makes it useful for stress-testing strategies across market conditions.

A thin CSV loader is included for when you want to plug in real data
(Yahoo Finance / exchange exports etc.). Expected columns:
    date, open, high, low, close, volume
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _segment(n, start_price, mu, sigma, rng):
    """Geometric-Brownian-Motion close path for one regime segment."""
    daily = rng.normal(mu, sigma, n)
    log_path = np.cumsum(daily)
    return start_price * np.exp(log_path)


def synthetic_ohlcv(
    n_days: int = 2520,            # ~10 trading years
    start_price: float = 100.0,
    seed: int | None = 42,
    freq: str = "B",
) -> pd.DataFrame:
    """
    Generate a regime-aware synthetic OHLCV series.

    Regimes rotate through bull / sideways / bear with realistic relative
    drift and volatility so that downstream backtests see all conditions.
    """
    rng = np.random.default_rng(seed)

    # (annualised-ish drift per day, daily vol) for each regime
    regimes = {
        "bull":     (0.0006, 0.011),
        "sideways": (0.0000, 0.008),
        "bear":     (-0.0007, 0.018),
    }
    order = ["bull", "sideways", "bear", "sideways", "bull", "bear", "sideways", "bull"]

    closes = []
    labels = []
    price = start_price
    remaining = n_days
    i = 0
    while remaining > 0:
        name = order[i % len(order)]
        mu, sigma = regimes[name]
        seg_len = min(remaining, rng.integers(180, 360))
        seg = _segment(seg_len, price, mu, sigma, rng)
        closes.append(seg)
        labels.extend([name] * seg_len)
        price = seg[-1]
        remaining -= seg_len
        i += 1

    close = np.concatenate(closes)[:n_days]
    labels = labels[:n_days]

    # Build OHLC around the close path
    noise = rng.normal(0, 0.004, n_days)
    open_ = close * (1 + noise)
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n_days)))

    base_vol = rng.lognormal(mean=13.0, sigma=0.4, size=n_days)
    # Volume expands with absolute daily return (panic / euphoria proxy)
    ret = np.r_[0.0, np.diff(np.log(close))]
    volume = base_vol * (1 + 6 * np.abs(ret))

    idx = pd.bdate_range(end=pd.Timestamp("2025-12-31"), periods=n_days, freq=freq)
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "regime": labels,
        },
        index=idx,
    )
    df.index.name = "date"
    return df


def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV data from a CSV with a `date` column."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    expected = {"open", "high", "low", "close", "volume"}
    missing = expected - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df.columns = [c.lower() for c in df.columns]
    return df
