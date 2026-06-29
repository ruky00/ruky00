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
| `qflow/feeds.py` | **Real** market data — Binance / Stooq / Yahoo + bundled GitHub samples |
| `qflow/indicators.py` | SMA, EMA, RSI, ATR, MACD, Bollinger, ADX, z-score |
| `qflow/strategies.py` | Trend-following, mean-reversion, volatility-breakout |
| `qflow/backtest.py` | Bar-by-bar engine: risk-based sizing, costs, trade ledger |
| `qflow/metrics.py` | CAGR, Sharpe, Sortino, Calmar, max DD, win rate, profit factor |
| `qflow/risk.py` | Position sizing, R:R, expectancy, Kelly, ATR stops |
| `qflow/regime.py` | Trend / volatility / volume regime classification |
| `qflow/multifactor.py` | Momentum + value + volatility + trend cross-sectional model |
| `qflow/montecarlo.py` | Trade-bootstrap Monte-Carlo robustness analysis |
| `qflow/portfolio.py` | Inverse-vol allocation + risk-tolerance overlay |
| `qflow/optimize.py` | Grid search + **walk-forward** (out-of-sample) tuning |
| `qflow/anomalies.py` | **Daily-edge scanner** — overnight, gaps, day-of-week, lead-lag |
| `qflow/paper.py` | **Paper-trading engine** — persistent forward test with virtual money |

The full quant walkthrough — covering strategy generation, backtesting,
risk/reward, regime detection, multi-factor models, optimization, portfolio
construction, trade setups, Monte-Carlo, drawdowns, macro overlays, and edge
detection — is in **[`docs/PLAYBOOK.md`](docs/PLAYBOOK.md)**.

---

## Quick start

```bash
pip install -r requirements.txt        # numpy + pandas

python examples/run_all.py             # full end-to-end demo (synthetic)
python examples/compare_strategies.py  # backtest all strategies on REAL data
python examples/find_edges.py          # hunt daily inefficiencies + walk-forward tuning
python tests/test_framework.py         # 15 correctness tests
```

### Finding daily edges & optimising

`examples/find_edges.py` scans real data for recurring daily inefficiencies and
backtests the tradeable ones net of costs. On the bundled samples it surfaces a
**gap-fade edge on high-volatility names** (TSLA, Sharpe 0.77) and a
**cross-market lead-lag** (one asset leading another by a day, Sharpe ~0.6), and
walk-forward-optimises mean-reversion out-of-sample. Details in
**[`docs/EDGES.md`](docs/EDGES.md)**.

### Forward-test with fake money (the path to going live)

Run a strategy on a **virtual $10,000** account against real data. State
persists between runs, so you run it once a day for a week or two and it resumes
where it left off — a true forward test, not a re-run.

```bash
# preview the whole workflow now (replays recent history, one bar = one day)
python examples/paper_trade.py --strategy mean_reversion --symbol AAPL --reset --replay 15

# the daily routine (run after the close; automate with cron)
python examples/paper_trade.py --strategy mean_reversion --symbol AAPL --step --report
```

It journals every fill, tracks the equity curve, and prints a **go-live
readiness gate**. Full plan: **[`docs/PAPER_TRADING.md`](docs/PAPER_TRADING.md)**.

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

### Real data

```python
from qflow import feeds
df = feeds.get("AAPL", "github")                      # bundled real sample (offline-safe)
df = feeds.get("BTCUSDT", "binance", interval="1d")   # crypto, no API key
df = feeds.get("aapl.us", "stooq")                    # equities / ETFs / FX
df = feeds.get("AAPL", "yahoo", rng="10y")            # 10y daily
# or your own CSV (columns: date,open,high,low,close,volume)
from qflow import data
df = data.load_csv("data/BTCUSD.csv")
```

Fetched series are cached under `data/` and reused. In a locked-down network
only the bundled GitHub samples are reachable; the other feeds work from an
ordinary machine.

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
├── qflow/            # the framework package (data, feeds, strategies, backtest, paper, ...)
├── examples/         # run_all · compare_strategies · find_edges · paper_trade
├── tests/            # test_framework.py — 15 correctness checks
├── data/samples/     # bundled REAL sample datasets (AAPL, TSLA)
├── docs/             # PLAYBOOK · EDGES · PAPER_TRADING · DISCLAIMER
├── requirements.txt
└── LICENSE           # MIT
```

## License

MIT — see [`LICENSE`](LICENSE). Provided "as is", without warranty of any kind.
