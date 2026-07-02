"""
Cross-sectional multi-factor model.

Given a panel of prices for several assets, score each asset on four factors,
z-score them cross-sectionally, combine with weights and rank. The top names
form an equal-or-risk-weighted long portfolio, rebalanced on a fixed cadence.

Factors
-------
Momentum   : 12-1 month total return (skip the most recent month)
Value      : inverse of price vs its own 2-year mean (cheapness proxy when no
             fundamentals are available; replace with B/P, E/P etc. for equities)
Volatility : negative of realised vol (low-vol anomaly -> prefer calmer names)
Trend      : distance of price above its 200-day SMA (persistence)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {"momentum": 0.35, "value": 0.20, "volatility": 0.20, "trend": 0.25}


def _zscore_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def compute_factors(prices: pd.DataFrame) -> dict:
    """`prices`: DataFrame indexed by date, one column per asset (close)."""
    rets = prices.pct_change()

    momentum = prices.shift(21) / prices.shift(252) - 1.0           # 12-1 momentum
    value = -(prices / prices.rolling(504).mean() - 1.0)            # cheapness proxy
    volatility = -rets.rolling(63).std() * np.sqrt(252)            # low-vol preference
    trend = prices / prices.rolling(200).mean() - 1.0              # above/below 200d

    return {
        "momentum": momentum,
        "value": value,
        "volatility": volatility,
        "trend": trend,
    }


def composite_score(prices: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    weights = weights or DEFAULT_WEIGHTS
    factors = compute_factors(prices)
    score = None
    for name, w in weights.items():
        z = _zscore_cross_section(factors[name])
        score = z * w if score is None else score + z * w
    return score


def build_portfolio(prices: pd.DataFrame,
                    weights: dict | None = None,
                    top_n: int = 3,
                    rebalance: str = "ME") -> pd.DataFrame:
    """
    Return a DataFrame of target weights (rows=rebalance dates, cols=assets).
    `rebalance`: pandas offset alias, e.g. 'ME' (month-end), 'W', 'QE'.
    """
    score = composite_score(prices, weights)
    rebal_dates = prices.resample(rebalance).last().index
    rebal_dates = rebal_dates[rebal_dates.isin(score.index) | True]

    target = pd.DataFrame(0.0, index=score.index, columns=prices.columns)
    for dt in score.index:
        row = score.loc[dt].dropna()
        if len(row) == 0:
            continue
        winners = row.sort_values(ascending=False).head(top_n).index
        target.loc[dt, winners] = 1.0 / len(winners)

    # only act on rebalance dates, hold in between
    mask = pd.Series(False, index=target.index)
    mask.loc[mask.index.isin(rebal_dates)] = True
    target = target.where(mask).ffill().fillna(0.0)
    return target


def backtest_panel(prices: pd.DataFrame,
                   target_weights: pd.DataFrame,
                   capital: float = 10_000.0,
                   cost_bps: float = 5.0) -> pd.Series:
    """Simple long-only panel backtest from target weights -> equity curve."""
    rets = prices.pct_change().fillna(0.0)
    w = target_weights.reindex(rets.index).ffill().fillna(0.0).shift(1).fillna(0.0)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    gross = (w * rets).sum(axis=1)
    net = gross - turnover * cost_bps / 1e4
    equity = capital * (1 + net).cumprod()
    equity.name = "equity"
    return equity
