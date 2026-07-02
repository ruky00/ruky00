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


def _volume(df):
    """Volume for VWAP weighting; FX feeds report no volume -> equal weights
    (the VWAP degrades gracefully to the session's running mean price)."""
    v = df["volume"].fillna(0)
    return v if v.sum() > 0 else pd.Series(1.0, index=df.index)


def vwap_reversion(df, dev=0.004, exit_dev=0.0015, atr_window=14, allow_short=True):
    """Long when price is `dev` below the day's VWAP, short when above; exit near VWAP."""
    day = _day(df)
    vol = _volume(df)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = (tp * vol).groupby(day).cumsum()
    vv = vol.groupby(day).cumsum().replace(0, np.nan)
    vwap = pv / vv
    z = df["close"] / vwap - 1.0
    sig = pd.Series(np.nan, index=df.index)
    sig[z < -dev] = 1
    if allow_short:
        sig[z > dev] = -1
    sig[z.abs() < exit_dev] = 0
    sig = sig.groupby(day).ffill().fillna(0)
    return StrategySignal(sig.astype(int), ind.atr(df, atr_window),
                          dict(dev=dev, exit_dev=exit_dev), execution="intraday",
                          diag=z / dev)   # ±1.0 = entry threshold reached


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


def vwap_snap(df, dev_z=2.0, exit_z=0.3, rsi_window=7, rsi_low=30.0, rsi_high=70.0,
              skip_open=2, no_entry_last=2, z_window=60, atr_window=14,
              allow_short=True):
    """
    High-win-rate VWAP reversion ("snap-back") — the funded bot's flagship.

    Improvements over plain vwap_reversion, each a standard quant filter:
      * adaptive entry: deviation from VWAP in *z-score* units (rolling std of
        the deviation), so the threshold self-adjusts to each symbol's vol;
      * RSI confirmation: only fade when short-term momentum is also stretched;
      * session-time filter: skip the chaotic first bars, none near the close.

    Trade it with asymmetric exits (tight TP ~1 ATR, wide SL ~2 ATR, optional
    time-stop) — see EXIT_PRESETS. Deep entries + short targets = many small
    wins, few larger losses; positive expectancy when the intraday reversion
    regime is present (the walk-forward decides that per symbol).
    """
    day = _day(df)
    vol = _volume(df)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = (tp * vol).groupby(day).cumsum()
    vv = vol.groupby(day).cumsum().replace(0, np.nan)
    z_raw = df["close"] / (pv / vv) - 1.0
    zstd = z_raw.rolling(z_window, min_periods=max(10, z_window // 3)).std().replace(0, np.nan)
    z = z_raw / zstd
    rsi = ind.rsi(df["close"], rsi_window)
    bar = df.groupby(day).cumcount()
    left = df.groupby(day).cumcount(ascending=False)
    ok = (bar >= skip_open) & (left >= no_entry_last)
    sig = pd.Series(np.nan, index=df.index)
    sig[(z < -dev_z) & (rsi < rsi_low) & ok] = 1
    if allow_short:
        sig[(z > dev_z) & (rsi > rsi_high) & ok] = -1
    sig[z.abs() < exit_z] = 0
    sig = sig.groupby(day).ffill().fillna(0)
    return StrategySignal(sig.astype(int), ind.atr(df, atr_window),
                          dict(dev_z=dev_z, exit_z=exit_z, rsi_low=rsi_low,
                               rsi_high=rsi_high), execution="intraday",
                          diag=z / dev_z)  # ±1.0 = entry threshold reached


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
    "vwap_snap": vwap_snap,
    "vwap_reversion": vwap_reversion,
    "opening_range": opening_range,
    "intraday_momentum": intraday_momentum,
    "intraday_auto": intraday_auto,
}

# Parameter grids for walk-forward / selection on intraday bars.
# vwap_reversion's dev is a raw fraction, so the grid must span BOTH stock-scale
# (0.3-0.8% intraday stretches) and FX-scale vol (EURGBP moves ~0.4%/day — its
# tradeable VWAP stretches are 0.05-0.15%). vwap_snap needs no such split: its
# z-score threshold self-adapts to each symbol's vol.
INTRADAY_GRIDS = {
    "vwap_snap": {"dev_z": [1.8, 2.0, 2.4], "rsi_low": [25.0, 30.0], "exit_z": [0.3, 0.5]},
    "vwap_reversion": {"dev": [0.0008, 0.0015, 0.003, 0.005, 0.008],
                       "exit_dev": [0.0003, 0.001, 0.002]},
    "opening_range": {"or_bars": [3, 6, 12]},
    "intraday_momentum": {"fast": [5, 9], "slow": [21, 34], "no_entry_last": [3, 6]},
}

# How each strategy wants to be EXITED. Mean reversion earns its win rate with an
# asymmetric bracket: short target (high probability of being touched), wider
# stop, and a time-stop so stale trades don't drift into the close. Breakout /
# momentum need the opposite (let winners run). The lab, the selector and the
# bot all read these so backtest == live behaviour.
EXIT_PRESETS = {
    "vwap_snap":         {"stop_atr": 2.0, "target_atr": 1.0, "max_bars": 8},
    "vwap_reversion":    {"stop_atr": 2.0, "target_atr": 1.0, "max_bars": 8},
    "opening_range":     {"stop_atr": 1.5, "target_atr": 2.5, "max_bars": 0},
    "intraday_momentum": {"stop_atr": 1.5, "target_atr": 2.5, "max_bars": 0},
    "intraday_auto":     {"stop_atr": 1.5, "target_atr": 2.0, "max_bars": 0},
}
