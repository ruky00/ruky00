"""
Portfolio construction.

Two practical allocation schemes that don't require a full optimiser:

    * inverse-volatility ("risk parity lite") — each asset contributes a
      similar amount of risk; calmer assets get more weight.
    * risk-budgeted by tolerance — scale equity exposure vs a cash/bond sleeve
      according to the investor's risk tolerance.

Plus a simple expected-return / risk estimator from historical data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def inverse_vol_weights(returns: pd.DataFrame) -> pd.Series:
    vol = returns.std(ddof=0) * np.sqrt(TRADING_DAYS)
    inv = 1.0 / vol.replace(0, np.nan)
    w = inv / inv.sum()
    return w.fillna(0.0)


def risk_tolerance_overlay(weights: pd.Series, tolerance: str = "medium") -> pd.Series:
    """
    Scale the risky sleeve vs cash based on tolerance. The remainder is held
    in 'CASH'.
    """
    equity_fraction = {"low": 0.40, "medium": 0.70, "high": 1.00}[tolerance]
    scaled = weights * equity_fraction
    scaled["CASH"] = 1.0 - equity_fraction
    return scaled


def expected_stats(returns: pd.DataFrame, weights: pd.Series) -> dict:
    w = weights.reindex(returns.columns).fillna(0.0).values
    mu = returns.mean().values * TRADING_DAYS
    cov = returns.cov().values * TRADING_DAYS
    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(w @ cov @ w))
    sharpe = port_ret / port_vol if port_vol else 0.0
    return {
        "expected_return": port_ret,
        "expected_vol": port_vol,
        "expected_sharpe": sharpe,
    }


def historical_drawdown(returns: pd.DataFrame, weights: pd.Series) -> float:
    w = weights.reindex(returns.columns).fillna(0.0)
    port = (returns * w).sum(axis=1)
    equity = (1 + port).cumprod()
    dd = (equity / equity.cummax() - 1.0).min()
    return float(dd)


def construct(prices: pd.DataFrame, tolerance: str = "medium") -> dict:
    """End-to-end: prices -> weights + expected stats."""
    returns = prices.pct_change().dropna()
    base = inverse_vol_weights(returns)
    final = risk_tolerance_overlay(base, tolerance)

    risky = final.drop("CASH")
    risky = risky / risky.sum() if risky.sum() else risky  # for stat calc
    stats = expected_stats(returns, risky)
    # de-scale expected return/vol by the equity fraction actually deployed
    eq_frac = 1.0 - final["CASH"]
    stats["expected_return"] *= eq_frac
    stats["expected_vol"] *= eq_frac
    stats["historical_max_dd"] = historical_drawdown(returns, base) * eq_frac

    return {"weights": final, "stats": stats}
