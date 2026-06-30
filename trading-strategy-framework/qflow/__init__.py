"""
qflow — a compact, dependency-light quant trading research framework.

Modules
-------
data        : synthetic OHLCV generation (regime-aware) + CSV loading
feeds       : real market data (Binance / Stooq / Yahoo / bundled GitHub samples)
indicators  : vectorised technical indicators (SMA/EMA/RSI/ATR/MACD/BB/ADX)
metrics     : performance statistics (CAGR, Sharpe, Sortino, max DD, win rate)
backtest    : event-light vectorised backtester with ATR-based risk sizing
strategies  : reference strategies (trend, mean-reversion, breakout)
daily       : intraday edge strategies (gap-fade, cross-market lead-lag)
regime      : market-regime classification (trend / volatility / volume)
multifactor : momentum + value + volatility + trend cross-sectional model
montecarlo  : trade-bootstrap Monte-Carlo robustness analysis
portfolio   : risk-based portfolio construction & allocation
risk        : position sizing and risk/reward helpers
risk_governor: portfolio kill-switches (daily loss, drawdown, heat, streak)
paper       : persistent paper-trading engine for forward testing
optimize    : grid search + walk-forward (out-of-sample) parameter tuning
anomalies   : daily-edge scanner (overnight, gaps, day-of-week, lead-lag)
edge_lab    : repeatable-edge lab (significance + consistency + OOS persistence)
news        : provider-agnostic news + finance sentiment (RSS/Finnhub/Bloomberg)

This is an educational research toolkit. Nothing here is financial advice.
See README.md for the full disclaimer.
"""

__version__ = "0.1.0"

from . import (  # noqa: F401
    data,
    feeds,
    indicators,
    metrics,
    backtest,
    strategies,
    daily,
    regime,
    multifactor,
    montecarlo,
    portfolio,
    risk,
    risk_governor,
    paper,
    optimize,
    anomalies,
    edge_lab,
    news,
)
