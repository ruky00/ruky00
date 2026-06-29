"""
Performance metrics computed from an equity curve and/or a return series.

All functions assume daily data by default (252 trading days / year).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    n_years = len(equity) / periods_per_year
    if n_years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1.0)


def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    excess = returns - rf / periods_per_year
    std = excess.std(ddof=0)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / std)


def sortino(returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    excess = returns - rf / periods_per_year
    downside = excess.clip(upper=0.0)
    dd_std = np.sqrt((downside ** 2).mean())
    if dd_std == 0 or np.isnan(dd_std):
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / dd_std)


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min())


def calmar(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return float(cagr(equity, periods_per_year) / mdd)


def volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=0) * np.sqrt(periods_per_year))


def win_rate(trade_pnls) -> float:
    arr = np.asarray(trade_pnls, dtype=float)
    if arr.size == 0:
        return 0.0
    return float((arr > 0).mean())


def profit_factor(trade_pnls) -> float:
    arr = np.asarray(trade_pnls, dtype=float)
    gains = arr[arr > 0].sum()
    losses = -arr[arr < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def avg_recovery_time(equity: pd.Series) -> float:
    """Average number of periods spent below a prior equity peak."""
    dd = drawdown_series(equity)
    in_dd = dd < 0
    lengths = []
    count = 0
    for flag in in_dd:
        if flag:
            count += 1
        elif count > 0:
            lengths.append(count)
            count = 0
    if count > 0:
        lengths.append(count)
    return float(np.mean(lengths)) if lengths else 0.0


def summary(equity: pd.Series, returns: pd.Series, trade_pnls=None) -> dict:
    out = {
        "Total Return": total_return(equity),
        "CAGR": cagr(equity),
        "Volatility": volatility(returns),
        "Sharpe": sharpe(returns),
        "Sortino": sortino(returns),
        "Calmar": calmar(equity),
        "Max Drawdown": max_drawdown(equity),
        "Avg Recovery (days)": avg_recovery_time(equity),
    }
    if trade_pnls is not None:
        out["Trades"] = int(len(trade_pnls))
        out["Win Rate"] = win_rate(trade_pnls)
        out["Profit Factor"] = profit_factor(trade_pnls)
    return out


def summary_table(stats: dict) -> str:
    """Render a metrics dict as a readable text table."""
    rows = []
    for k, v in stats.items():
        if isinstance(v, float):
            if any(t in k for t in ("Return", "CAGR", "Drawdown", "Volatility", "Win Rate")):
                rows.append(f"{k:<22} {v*100:>10.2f}%")
            else:
                rows.append(f"{k:<22} {v:>11.2f}")
        else:
            rows.append(f"{k:<22} {v:>11}")
    width = max(len(r) for r in rows)
    line = "-" * width
    return "\n".join([line, *rows, line])
