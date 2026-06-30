"""
Reference strategies.

Each strategy is a callable that takes an OHLCV DataFrame and returns a
`StrategySignal` containing:
    * `signal` : target position series (-1 / 0 / +1)
    * `atr`    : ATR series the backtester uses for stops/targets
    * `params` : the settings used (handy for reporting / optimisation)

The three strategies are intentionally different in character so that a
portfolio of them is diversified across market regimes:

    1. Trend Following (EMA cross + ADX filter)   -> trending markets
    2. Mean Reversion (RSI + Bollinger)           -> ranging markets
    3. Volatility Breakout (Donchian + ATR)       -> expansion / momentum
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import indicators as ind


@dataclass
class StrategySignal:
    signal: pd.Series
    atr: pd.Series
    params: dict
    execution: str = "swing"   # "swing" (multi-day, ATR stops) or "intraday" (open->close)


# --------------------------------------------------------------------------- #
# 1. Trend Following
# --------------------------------------------------------------------------- #
def trend_following(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
    adx_window: int = 14,
    adx_threshold: float = 20.0,
    atr_window: int = 14,
    allow_short: bool = True,
) -> StrategySignal:
    """Go with the trend only when ADX confirms a real trend is present."""
    ema_fast = ind.ema(df["close"], fast)
    ema_slow = ind.ema(df["close"], slow)
    adx = ind.adx(df, adx_window)
    atr = ind.atr(df, atr_window)

    long = (ema_fast > ema_slow) & (adx > adx_threshold)
    short = (ema_fast < ema_slow) & (adx > adx_threshold)

    signal = pd.Series(0, index=df.index)
    signal[long] = 1
    if allow_short:
        signal[short] = -1
    return StrategySignal(
        signal=signal,
        atr=atr,
        params=dict(fast=fast, slow=slow, adx_window=adx_window,
                    adx_threshold=adx_threshold, atr_window=atr_window),
    )


# --------------------------------------------------------------------------- #
# 2. Mean Reversion
# --------------------------------------------------------------------------- #
def mean_reversion(
    df: pd.DataFrame,
    rsi_window: int = 2,
    rsi_buy: float = 10.0,
    rsi_exit: float = 55.0,
    trend_window: int = 200,
    bb_window: int = 20,
    atr_window: int = 14,
) -> StrategySignal:
    """
    Buy short-term oversold dips *inside* a longer-term uptrend (classic
    Connors-style 2-period RSI), exit on mean reversion back to the middle.
    """
    rsi = ind.rsi(df["close"], rsi_window)
    trend = ind.sma(df["close"], trend_window)
    atr = ind.atr(df, atr_window)
    _, _, _, pct_b = ind.bollinger(df["close"], bb_window)

    uptrend = df["close"] > trend
    enter = uptrend & (rsi < rsi_buy)
    exit_ = (rsi > rsi_exit) | (pct_b > 0.8)

    signal = pd.Series(float("nan"), index=df.index)
    signal[enter] = 1
    signal[exit_] = 0
    signal = signal.ffill().fillna(0)
    return StrategySignal(
        signal=signal,
        atr=atr,
        params=dict(rsi_window=rsi_window, rsi_buy=rsi_buy, rsi_exit=rsi_exit,
                    trend_window=trend_window, bb_window=bb_window,
                    atr_window=atr_window),
    )


# --------------------------------------------------------------------------- #
# 3. Volatility Breakout (Donchian channel)
# --------------------------------------------------------------------------- #
def volatility_breakout(
    df: pd.DataFrame,
    channel: int = 55,
    exit_channel: int = 20,
    atr_window: int = 14,
    vol_filter: int = 100,
    allow_short: bool = True,
) -> StrategySignal:
    """
    Turtle-style breakout: enter on a new N-day high/low, ride momentum, exit
    on the opposite shorter channel. A volatility-expansion filter avoids
    chopping inside dead ranges.
    """
    # Break out on a new N-day *closing* high/low (more tradeable than the
    # max-of-highs, which the close rarely exceeds).
    upper = df["close"].rolling(channel).max()
    lower = df["close"].rolling(channel).min()
    exit_up = df["close"].rolling(exit_channel).max()
    exit_dn = df["close"].rolling(exit_channel).min()
    atr = ind.atr(df, atr_window)
    atr_med = atr.rolling(vol_filter).median()
    expanding = atr > atr_med  # only trade when vol is above its own median

    long_entry = ((df["close"] >= upper.shift(1)) & expanding).values
    short_entry = ((df["close"] <= lower.shift(1)) & expanding).values
    long_exit = (df["close"] <= exit_dn.shift(1)).values
    short_exit = (df["close"] >= exit_up.shift(1)).values

    # Explicit position state: entries take priority over exits, and the
    # exit channel only flattens the matching direction.
    pos = 0
    out = []
    for i in range(len(df)):
        if pos == 0:
            if long_entry[i]:
                pos = 1
            elif short_entry[i] and allow_short:
                pos = -1
        elif pos == 1:
            if long_exit[i] and not long_entry[i]:
                pos = 0
        elif pos == -1:
            if short_exit[i] and not short_entry[i]:
                pos = 0
        out.append(pos)
    signal = pd.Series(out, index=df.index)
    return StrategySignal(
        signal=signal,
        atr=atr,
        params=dict(channel=channel, exit_channel=exit_channel,
                    atr_window=atr_window, vol_filter=vol_filter),
    )


REGISTRY = {
    "trend_following": trend_following,
    "mean_reversion": mean_reversion,
    "volatility_breakout": volatility_breakout,
}
