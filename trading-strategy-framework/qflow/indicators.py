"""
Vectorised technical indicators (pandas / numpy).

Each function takes a price Series (or OHLC DataFrame) and returns a Series
aligned to the input index. Settings are exposed as arguments so they can be
swept during optimisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(price: pd.Series, window: int) -> pd.Series:
    return price.rolling(window).mean()


def ema(price: pd.Series, span: int) -> pd.Series:
    return price.ewm(span=span, adjust=False).mean()


def rsi(price: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    tr = true_range(df)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def macd(price: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, signal_line, histogram)."""
    macd_line = ema(price, fast) - ema(price, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(price: pd.Series, window: int = 20, n_std: float = 2.0):
    """Return (mid, upper, lower, %b)."""
    mid = sma(price, window)
    sd = price.rolling(window).std(ddof=0)
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    pct_b = (price - lower) / (upper - lower)
    return mid, upper, lower, pct_b


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average Directional Index — trend-strength gauge (0-100)."""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    atr_ = tr.ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / window, adjust=False
    ).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / window, adjust=False
    ).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, adjust=False).mean().fillna(0.0)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)
