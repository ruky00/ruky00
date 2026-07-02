"""
Market-regime detection.

Classifies each bar along three axes and produces a single label plus a
human-readable recommendation of which strategy *type* suits the environment.

    Trend      : bull / bear / sideways   (200-SMA slope + price location + ADX)
    Volatility : low / normal / high      (ATR% vs its own rolling distribution)
    Volume     : contracting / expanding  (volume vs its moving average)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def detect(df: pd.DataFrame,
           trend_window: int = 200,
           adx_window: int = 14,
           atr_window: int = 14,
           vol_lookback: int = 252) -> pd.DataFrame:
    close = df["close"]
    sma = ind.sma(close, trend_window)
    slope = sma.diff(20)
    adx = ind.adx(df, adx_window)

    atr = ind.atr(df, atr_window)
    atr_pct = atr / close
    atr_rank = atr_pct.rolling(vol_lookback).rank(pct=True)

    vol_ma = df["volume"].rolling(20).mean()
    vol_ratio = df["volume"] / vol_ma

    trend = pd.Series("sideways", index=df.index)
    trend[(close > sma) & (slope > 0) & (adx > 20)] = "bull"
    trend[(close < sma) & (slope < 0) & (adx > 20)] = "bear"

    volatility = pd.Series("normal", index=df.index)
    volatility[atr_rank < 0.33] = "low"
    volatility[atr_rank > 0.66] = "high"

    volume = np.where(vol_ratio > 1.1, "expanding", "contracting")

    out = pd.DataFrame(
        {
            "trend": trend,
            "volatility": volatility,
            "volume": volume,
            "adx": adx,
            "atr_pct": atr_pct,
        },
        index=df.index,
    )
    return out


_RECOMMENDATION = {
    ("bull", "low"): ("Trend following / buy-the-dip", "Fading strength, heavy shorting"),
    ("bull", "normal"): ("Trend following + breakout", "Counter-trend shorts"),
    ("bull", "high"): ("Breakout with wider stops, smaller size", "Mean reversion shorts into strength"),
    ("bear", "low"): ("Short rallies / defensive cash", "Aggressive longs"),
    ("bear", "normal"): ("Trend-following shorts", "Bottom-picking longs"),
    ("bear", "high"): ("Reduce size, wait; tactical mean-reversion bounces only", "Leverage, breakout longs"),
    ("sideways", "low"): ("Mean reversion (range fade)", "Breakout strategies — they will whipsaw"),
    ("sideways", "normal"): ("Mean reversion, range trading", "Trend following"),
    ("sideways", "high"): ("Stay small; straddle/volatility plays", "Directional trend bets"),
}


def recommend(regime_row: pd.Series) -> dict:
    """Map a single regime row to a strategy recommendation."""
    key = (regime_row["trend"], regime_row["volatility"])
    best, avoid = _RECOMMENDATION.get(key, ("Mean reversion", "Trend following"))
    return {
        "trend": regime_row["trend"],
        "volatility": regime_row["volatility"],
        "volume": regime_row["volume"],
        "best_strategy": best,
        "avoid": avoid,
    }


def current(df: pd.DataFrame) -> dict:
    """Convenience: regime + recommendation for the most recent bar."""
    reg = detect(df)
    return recommend(reg.iloc[-1])
