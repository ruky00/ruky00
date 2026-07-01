"""
Intraday strategies — designed for intraday bars (5m/15m/…), flat overnight.

Unlike the daily strategies reused naively on 5-minute bars (which lose to costs),
these use genuinely intraday structure that resets every session:

    vwap_reversion      : fade deviations from the day's VWAP (mean-reversion)
    opening_range       : break of the first-N-bars high/low (opening-range breakout)
    intraday_momentum   : fast/slow EMA cross, no new entries near the close
    intraday_auto       : pick breakout when the day trends (ADX), else VWAP reversion

Each returns a ``StrategySignal`` (execution="intraday"). Backtest them with
``run_backtest(..., flatten_eod=True)`` on intraday bars, and validate with
``optimize.rolling_walk_forward(..., bt_kwargs={"flatten_eod": True})``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategies import StrategySignal
from . import indicators as ind


def _day(df):
    """Group key = calendar session (resets VWAP / opening range each day)."""
    return df.index.normalize()


def vwap_reversion(df, dev=0.004, exit_dev=0.0015, atr_window=14, allow_short=True):
    """Long when price is `dev` below the day's VWAP, short when above; exit near VWAP."""
    day = _day(df)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vv = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    vwap = pv / vv
    z = df["close"] / vwap - 1.0
    sig = pd.Series(np.nan, index=df.index)
    sig[z < -dev] = 1
    if allow_short:
        sig[z > dev] = -1
    sig[z.abs() < exit_dev] = 0
    sig = sig.groupby(day).ffill().fillna(0)
    return StrategySignal(sig.astype(int), ind.atr(df, atr_window),
                          dict(dev=dev, exit_dev=exit_dev), execution="intraday")


def opening_range(df, or_bars=6, atr_window=14, allow_short=True):
    """Break of the first `or_bars` bars' high/low (opening-range breakout)."""
    day = _day(df)
    bar_idx = df.groupby(day).cumcount()
    is_or = bar_idx < or_bars
    or_high = df["high"].where(is_or).groupby(day).transform("max")
    or_low = df["low"].where(is_or).groupby(day).transform("min")
    after = bar_idx >= or_bars
    sig = pd.Series(np.nan, index=df.index)
    sig[after & (df["close"] > or_high)] = 1
    if allow_short:
        sig[after & (df["close"] < or_low)] = -1
    sig = sig.groupby(day).ffill().fillna(0)
    return StrategySignal(sig.astype(int), ind.atr(df, atr_window),
                          dict(or_bars=or_bars), execution="intraday")


def intraday_momentum(df, fast=9, slow=21, atr_window=14, no_entry_last=6, allow_short=True):
    """Fast/slow EMA cross; no new entries in the last `no_entry_last` bars of the day."""
    day = _day(df)
    ef, es = ind.ema(df["close"], fast), ind.ema(df["close"], slow)
    ok = df.groupby(day).cumcount(ascending=False) >= no_entry_last
    sig = pd.Series(0, index=df.index)
    sig[(ef > es) & ok] = 1
    if allow_short:
        sig[(ef < es) & ok] = -1
    return StrategySignal(sig.astype(int), ind.atr(df, atr_window),
                          dict(fast=fast, slow=slow), execution="intraday")


def intraday_auto(df, adx_threshold=25.0, atr_window=14):
    """Regime switch: opening-range breakout when the day trends, else VWAP reversion."""
    adx = ind.adx(df, 14).values
    vr = vwap_reversion(df).signal.values
    orb = opening_range(df).signal.values
    sig = np.where(adx > adx_threshold, orb, vr)
    return StrategySignal(pd.Series(sig, index=df.index).astype(int),
                          ind.atr(df, atr_window),
                          dict(mode="intraday_auto", adx_threshold=adx_threshold),
                          execution="intraday")


INTRADAY_REGISTRY = {
    "vwap_reversion": vwap_reversion,
    "opening_range": opening_range,
    "intraday_momentum": intraday_momentum,
    "intraday_auto": intraday_auto,
}

# Parameter grids for walk-forward / selection on intraday bars.
INTRADAY_GRIDS = {
    "vwap_reversion": {"dev": [0.003, 0.005, 0.008], "exit_dev": [0.001, 0.002]},
    "opening_range": {"or_bars": [3, 6, 12]},
    "intraday_momentum": {"fast": [5, 9], "slow": [21, 34], "no_entry_last": [3, 6]},
}
