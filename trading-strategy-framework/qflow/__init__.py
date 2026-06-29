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
regime      : market-regime classification (trend / volatility / volume)
multifactor : momentum + value + volatility + trend cross-sectional model
montecarlo  : trade-bootstrap Monte-Carlo robustness analysis
portfolio   : risk-based portfolio construction & allocation
risk        : position sizing and risk/reward helpers
paper       : persistent paper-trading engine for forward testing

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
    regime,
    multifactor,
    montecarlo,
    portfolio,
    risk,
    paper,
)
