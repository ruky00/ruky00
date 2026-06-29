# qflow — Quant Trading Strategy Framework

A compact, dependency-light Python framework for **researching, backtesting,
and stress-testing systematic trading strategies**. It runs fully **offline**
on a regime-aware synthetic data generator, so you can clone and run it
anywhere — then swap in your own CSV data when you're ready.

> ⚠️ **Educational use only — not financial advice.** All strategies, numbers,
> and backtests are illustrative and produced on synthetic data. Backtested
> results do not predict future performance. See [`docs/DISCLAIMER.md`](docs/DISCLAIMER.md).

---

## What's inside

| Module | Purpose |
|--------|---------|
| `qflow/data.py` | Regime-aware synthetic OHLCV generator + CSV loader |
| `qflow/indicators.py` | SMA, EMA, RSI, ATR, MACD, Bollinger, ADX, z-score |
| `qflow/strategies.py` | Trend-following, mean-reversion, volatility-breakout |
| `qflow/backtest.py` | Bar-by-bar engine: risk-based sizing, costs, trade ledger |
| `qflow/metrics.py` | CAGR, Sharpe, Sortino, Calmar, max DD, win rate, profit factor |
| `qflow/risk.py` | Position sizing, R:R, expectancy, Kelly, ATR stops |
| `qflow/regime.py` | Trend / volatility / volume regime classification |
| `qflow/multifactor.py` | Momentum + value + volatility + trend cross-sectional model |
| `qflow/montecarlo.py` | Trade-bootstrap Monte-Carlo robustness analysis |
| `qflow/portfolio.py` | Inverse-vol allocation + risk-tolerance overlay |

The full quant walkthrough — covering strategy generation, backtesting,
risk/reward, regime detection, multi-factor models, optimization, portfolio
construction, trade setups, Monte-Carlo, drawdowns, macro overlays, and edge
detection — is in **[`docs/PLAYBOOK.md`](docs/PLAYBOOK.md)**.

---

## Quick start

```bash
pip install -r requirements.txt        # numpy + pandas

python examples/run_all.py             # full end-to-end demo (offline)
python tests/test_framework.py         # 10 correctness tests
```

### Minimal example

```python
from qflow import data, strategies, backtest

df  = data.synthetic_ohlcv(n_days=2520, seed=42)          # ~10y daily
sig = strategies.trend_following(df, fast=20, slow=50)    # EMA cross + ADX gate
res = backtest.run_backtest(
    df, sig.signal, sig.atr,
    capital=10_000, risk_per_trade=0.01,                  # risk 1% / trade
    stop_atr=2.0, target_atr=4.0,                         # 2:1 reward:risk
)
print(res.report())          # CAGR, Sharpe, max DD, win rate, ...
```

### Use your own data

```python
from qflow import data
df = data.load_csv("data/BTCUSD.csv")   # columns: date,open,high,low,close,volume
```

---

## Design principles

- **No look-ahead** — fills happen at the *next* bar's open; indicators are
  computed causally.
- **Risk-first** — every trade risks a fixed fraction of equity; the stop
  distance (in ATR) determines size, never a fixed share count.
- **Honest backtests** — commission + slippage modelled; a full trade ledger
  drives win rate, profit factor, and Monte-Carlo resampling.
- **Light dependencies** — only `numpy` and `pandas`.

## Project layout

```
trading-strategy-framework/
├── qflow/            # the framework package
├── examples/         # run_all.py — end-to-end walkthrough
├── tests/            # test_framework.py — correctness checks
├── docs/             # PLAYBOOK.md (the quant guide) + DISCLAIMER.md
├── requirements.txt
└── LICENSE           # MIT
```

## License

MIT — see [`LICENSE`](LICENSE). Provided "as is", without warranty of any kind.
