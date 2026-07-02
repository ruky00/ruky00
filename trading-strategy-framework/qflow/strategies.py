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
    # optional live diagnostic: signed fraction of the entry threshold reached
    # (-1/+1 = a short/long entry fires). Lets the bot show "how close" a quiet
    # symbol is to signalling instead of an opaque "flat".
    diag: pd.Series | None = None


# --------------------------------------------------------------------------- #
# 1. Trend Following
# --------------------------------------------------------------------------- #
def trend_following(
    df: pd.DataFrame,
    fast: int = 50,
    slow: int = 200,
    adx_window: int = 14,
    adx_threshold: float = 20.0,
    atr_window: int = 14,
    vol_window: int = 20,
    vol_confirm: bool = True,
    allow_short: bool = True,
) -> StrategySignal:
    """
    Trend following = EMA cross (default 50/200 golden/death cross) confirmed by
    ADX (real trend, not chop) and by *volume* (participation behind the move).
    """
    ema_fast = ind.ema(df["close"], fast)
    ema_slow = ind.ema(df["close"], slow)
    adx = ind.adx(df, adx_window)
    atr = ind.atr(df, atr_window)

    if vol_confirm:
        vol_ok = df["volume"] > ind.sma(df["volume"], vol_window)
    else:
        vol_ok = pd.Series(True, index=df.index)

    long = (ema_fast > ema_slow) & (adx > adx_threshold) & vol_ok
    short = (ema_fast < ema_slow) & (adx > adx_threshold) & vol_ok

    signal = pd.Series(0, index=df.index)
    signal[long] = 1
    if allow_short:
        signal[short] = -1
    return StrategySignal(
        signal=signal,
        atr=atr,
        params=dict(fast=fast, slow=slow, adx_window=adx_window,
                    adx_threshold=adx_threshold, atr_window=atr_window,
                    vol_window=vol_window, vol_confirm=vol_confirm),
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
    bb_std: float = 2.0,
    pct_b_buy: float = 0.05,
    atr_window: int = 14,
) -> StrategySignal:
    """
    Buy short-term oversold dips *inside* a longer-term uptrend: enter when a
    2-period RSI is washed out OR price pierces the lower Bollinger band
    (%B < pct_b_buy), but only above the 200-SMA trend filter. Exit on reversion
    back toward the middle band / RSI recovery. ATR drives risk sizing.
    """
    rsi = ind.rsi(df["close"], rsi_window)
    trend = ind.sma(df["close"], trend_window)
    atr = ind.atr(df, atr_window)
    _, _, _, pct_b = ind.bollinger(df["close"], bb_window, bb_std)

    uptrend = df["close"] > trend
    enter = uptrend & ((rsi < rsi_buy) | (pct_b < pct_b_buy))
    exit_ = (rsi > rsi_exit) | (pct_b > 0.8)

    signal = pd.Series(float("nan"), index=df.index)
    signal[enter] = 1
    signal[exit_] = 0
    signal = signal.ffill().fillna(0)
    return StrategySignal(
        signal=signal,
        atr=atr,
        params=dict(rsi_window=rsi_window, rsi_buy=rsi_buy, rsi_exit=rsi_exit,
                    trend_window=trend_window, bb_window=bb_window, bb_std=bb_std,
                    pct_b_buy=pct_b_buy, atr_window=atr_window),
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
    squeeze: bool = True,
    squeeze_lookback: int = 20,
    bb_window: int = 20,
    allow_short: bool = True,
) -> StrategySignal:
    """
    Donchian breakout with a volatility *squeeze* precondition: only take a new
    N-day closing high/low when (a) ATR is expanding above its median AND (b)
    volatility was recently *compressed* (Bollinger band-width below its median in
    the last `squeeze_lookback` bars) — the classic coiled-spring setup. Exit on
    the opposite shorter channel.
    """
    upper = df["close"].rolling(channel).max()
    lower = df["close"].rolling(channel).min()
    exit_up = df["close"].rolling(exit_channel).max()
    exit_dn = df["close"].rolling(exit_channel).min()
    atr = ind.atr(df, atr_window)
    atr_med = atr.rolling(vol_filter).median()
    expanding = atr > atr_med  # only trade when vol is above its own median

    if squeeze:
        mid, up_bb, lo_bb, _ = ind.bollinger(df["close"], bb_window)
        bb_width = (up_bb - lo_bb) / mid.replace(0, float("nan"))
        was_squeezed = bb_width < bb_width.rolling(vol_filter).median()
        # a squeeze occurred within the recent lookback -> spring is coiled
        recently_squeezed = was_squeezed.rolling(squeeze_lookback).max().fillna(0).astype(bool)
    else:
        recently_squeezed = pd.Series(True, index=df.index)

    long_entry = ((df["close"] >= upper.shift(1)) & expanding & recently_squeezed).values
    short_entry = ((df["close"] <= lower.shift(1)) & expanding & recently_squeezed).values
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
                    atr_window=atr_window, vol_filter=vol_filter,
                    squeeze=squeeze, squeeze_lookback=squeeze_lookback,
                    bb_window=bb_window),
    )


# --------------------------------------------------------------------------- #
# Adaptive — pick the best strategy per bar from the market regime
# --------------------------------------------------------------------------- #
def adaptive(df: pd.DataFrame, atr_window: int = 14, allow_short: bool = True) -> StrategySignal:
    """
    Regime-switching meta-strategy. For each bar it reads the market regime and
    uses the signal of the strategy that fits it:

        trending (bull/bear + ADX)  -> trend_following
        high volatility (ranging)   -> volatility_breakout
        calm / sideways             -> mean_reversion

    This is what "let the bot choose the best strategy from the candles" means.
    """
    from . import regime
    reg = regime.detect(df)
    tf = trend_following(df, allow_short=allow_short)
    mr = mean_reversion(df)
    vb = volatility_breakout(df, allow_short=allow_short)
    atr = ind.atr(df, atr_window)

    trend = reg["trend"].values
    vol = reg["volatility"].values
    tfs, mrs, vbs = tf.signal.values, mr.signal.values, vb.signal.values
    chosen = []
    out = []
    for i in range(len(df)):
        if trend[i] in ("bull", "bear"):
            out.append(int(tfs[i])); chosen.append("trend_following")
        elif vol[i] == "high":
            out.append(int(vbs[i])); chosen.append("volatility_breakout")
        else:
            out.append(int(mrs[i])); chosen.append("mean_reversion")
    signal = pd.Series(out, index=df.index)
    sig = StrategySignal(signal=signal, atr=atr,
                         params=dict(mode="adaptive", atr_window=atr_window))
    sig.chosen = pd.Series(chosen, index=df.index)   # which strategy per bar
    return sig


# Default parameter grids for the walk-forward-optimised adaptive strategy.
DEFAULT_WF_GRIDS = {
    "trend_following": {"fast": [20, 50], "slow": [100, 200], "adx_threshold": [15, 20, 25]},
    "mean_reversion": {"rsi_buy": [5, 10, 15], "rsi_exit": [55, 65], "pct_b_buy": [0.02, 0.05]},
    "volatility_breakout": {"channel": [40, 55], "exit_channel": [10, 20],
                            "squeeze_lookback": [15, 25]},
}


def _yearly_optimized_signal(df, name, grid, train_years, bt_kwargs):
    """Signal for one sub-strategy where each calendar year uses the params that
    a rolling walk-forward re-optimised on that year's trailing window."""
    from . import optimize
    fn = REGISTRY[name]
    pmap = optimize.walk_forward_params(df, name, grid, train_years=train_years,
                                        bt_kwargs=bt_kwargs)
    sig = fn(df).signal.copy()                     # default for uncovered years
    for year, params in pmap.items():
        yearly = fn(df, **params).signal
        mask = df.index.year == year
        sig[mask] = yearly[mask]
    return sig


def adaptive_walk_forward(df: pd.DataFrame,
                          grids: dict | None = None,
                          train_years: int = 4,
                          atr_window: int = 14,
                          bt_kwargs: dict | None = None) -> StrategySignal:
    """
    Like ``adaptive`` (regime picks the sub-strategy per bar), but each
    sub-strategy uses **per-year walk-forward-optimised parameters** instead of
    fixed defaults — so the bot re-tunes each strategy every year automatically.
    Slower (runs grid search per fold); best for daily use / backtests.
    """
    from . import regime
    grids = grids or DEFAULT_WF_GRIDS
    bt_kwargs = bt_kwargs or {"capital": 10_000.0, "risk_per_trade": 0.01}
    tf = _yearly_optimized_signal(df, "trend_following", grids["trend_following"],
                                  train_years, bt_kwargs).values
    mr = _yearly_optimized_signal(df, "mean_reversion", grids["mean_reversion"],
                                  train_years, bt_kwargs).values
    vb = _yearly_optimized_signal(df, "volatility_breakout", grids["volatility_breakout"],
                                  train_years, bt_kwargs).values
    atr = ind.atr(df, atr_window)
    reg = regime.detect(df)
    trend, vol = reg["trend"].values, reg["volatility"].values
    out, chosen = [], []
    for i in range(len(df)):
        if trend[i] in ("bull", "bear"):
            out.append(int(tf[i])); chosen.append("trend_following")
        elif vol[i] == "high":
            out.append(int(vb[i])); chosen.append("volatility_breakout")
        else:
            out.append(int(mr[i])); chosen.append("mean_reversion")
    sig = StrategySignal(signal=pd.Series(out, index=df.index), atr=atr,
                         params=dict(mode="adaptive_wf", train_years=train_years))
    sig.chosen = pd.Series(chosen, index=df.index)
    return sig


def adaptive_status(df: pd.DataFrame) -> dict:
    """What the adaptive strategy is doing on the LATEST bar (for live display)."""
    from . import regime
    reg = regime.detect(df).iloc[-1]
    sig = adaptive(df)
    return {
        "trend": reg["trend"],
        "volatility": reg["volatility"],
        "active_strategy": sig.chosen.iloc[-1],
        "signal": int(sig.signal.iloc[-1]),   # -1 short / 0 flat / +1 long
    }


REGISTRY = {
    "trend_following": trend_following,
    "mean_reversion": mean_reversion,
    "volatility_breakout": volatility_breakout,
    "auto": adaptive,
    "auto_wf": adaptive_walk_forward,   # auto + per-year walk-forward-optimised params
}

# Register the dedicated intraday strategies (vwap_reversion, opening_range,
# intraday_momentum, intraday_auto) so walk_forward / portfolio_selector / the
# bot can reach them by name through REGISTRY. They expect intraday bars and a
# backtest run with flatten_eod=True. Imported last to avoid a circular import
# (intraday_strategies imports StrategySignal from this module).
from . import intraday_strategies as _intraday   # noqa: E402
REGISTRY.update(_intraday.INTRADAY_REGISTRY)
INTRADAY_GRIDS = _intraday.INTRADAY_GRIDS
EXIT_PRESETS = _intraday.EXIT_PRESETS
