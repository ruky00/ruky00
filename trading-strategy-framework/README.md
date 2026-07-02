# qflow — Quant Trading Strategy Framework

A compact, dependency-light Python framework for **researching, backtesting,
and stress-testing systematic trading strategies**. It runs fully **offline**
on a regime-aware synthetic data generator, so you can clone and run it
anywhere — then swap in your own CSV data when you're ready.

> ⚠️ **Educational use only — not financial advice.** All strategies, numbers,
> and backtests are illustrative and produced on synthetic data. Backtested
> results do not predict future performance. See [`docs/DISCLAIMER.md`](docs/DISCLAIMER.md).

---

## Two products

| | **The Lab** (research + daily swing) | **The Funded Bot** (live intraday) |
|---|---|---|
| Code | `qflow/` · `examples/research/` · `examples/live/` | `bot/intraday_bot.py` |
| Job | Find & validate robust `(symbol, strategy)` edges, forward-test the daily book | Trade the validated strategies **intraday** for recurring P&L on a funded/IBKR account |
| Bars | daily (5–10y backtest) | 5-minute, live |

The Lab decides **what** to trade; the Bot **trades it** — same strategies, risk
engine and broker layer, different timeframe. Full plan & roadmap:
**[`docs/ROADMAP.md`](docs/ROADMAP.md)**.

```bash
# The Funded Bot — live intraday on IBKR paper (Python 3.12 venv, IB Gateway on 4002)
python bot/intraday_bot.py --strategy auto --symbols NVDA,AMD,TSLA,AAPL --kill-switches

# Fully autonomous funded-exam run: self-select the best (strategy, interval) per
# symbol via walk-forward, then trade the prop-firm challenge with greedy-but-capped
# sizing (presses with cushion, de-risks near the limits, locks the pass at target)
python bot/intraday_bot.py --auto-select --funded --allow-short \
    --symbols NVDA,AMD,TSLA,AAPL \
    --profit-target 0.08 --max-daily-loss 0.05 --max-total-drawdown 0.10
```

---

## What's inside

| Module | Purpose |
|--------|---------|
| `qflow/data.py` | Regime-aware synthetic OHLCV generator + CSV loader |
| `qflow/feeds.py` | **Real** market data — Binance / Stooq / Yahoo + bundled GitHub samples |
| `qflow/indicators.py` | SMA, EMA, RSI, ATR, MACD, Bollinger, ADX, z-score |
| `qflow/strategies.py` | Trend (EMA 50/200 + ADX + volume) · mean-reversion (RSI + Bollinger) · breakout (Donchian + ATR squeeze) · `auto` |
| `qflow/intraday_strategies.py` | **Intraday** strategies — `vwap_snap` (z-score VWAP snap-back, the funded flagship) · VWAP reversion · opening-range breakout · intraday momentum · `intraday_auto` + per-strategy EXIT_PRESETS |
| `qflow/backtest.py` | Bar-by-bar engine: risk-based sizing, costs, trade ledger |
| `qflow/metrics.py` | CAGR, Sharpe, Sortino, Calmar, max DD, win rate, profit factor |
| `qflow/risk.py` | Position sizing, R:R, expectancy, Kelly, ATR stops |
| `qflow/regime.py` | Trend / volatility / volume regime classification |
| `qflow/multifactor.py` | Momentum + value + volatility + trend cross-sectional model |
| `qflow/montecarlo.py` | Trade-bootstrap Monte-Carlo robustness analysis |
| `qflow/portfolio.py` | Inverse-vol allocation + risk-tolerance overlay |
| `qflow/optimize.py` | Grid search + **walk-forward** + **rolling calendar walk-forward** |
| `qflow/anomalies.py` | **Daily-edge scanner** — overnight, gaps, day-of-week, lead-lag |
| `qflow/edge_lab.py` | **Repeatable-edge lab** — significance + consistency + OOS persistence |
| `qflow/universe.py` | **Universe scanner** — sweep IBEX-35 / S&P-500 for robust patterns |
| `qflow/dual_listing.py` | **España↔US lead-lag** — Santander/BBVA/Telefónica cross-listings |
| `qflow/news.py` | **News + sentiment** — provider-agnostic (RSS/Finnhub/Bloomberg/FinBERT) trade overlay |
| `qflow/risk_governor.py` | **Kill-switches** — daily-loss / drawdown / heat / streak circuit breakers |
| `qflow/funded.py` | **Funded-exam engine** — profit target + daily-loss / drawdown limits with greedy-but-capped dynamic sizing |
| `qflow/intraday_select.py` | **Autonomous picker** — walk-forwards the intraday strategies across intervals, keeps the best OOS (+ cache) |
| `qflow/journal.py` | **Trade journal** — append-only CSV of entries/exits → live win-rate / expectancy |
| `qflow/broker.py` | **Execution** — PaperBroker · IBKRBroker (stocks) · MT5Broker (FundedNext, two-way) · WebhookBroker (Lucid via TradersPost) |
| `qflow/paper.py` | **Paper-trading engine** — persistent forward test with virtual money |
| `qflow/portfolio_runner.py` | **Basket runner** — one strategy across many symbols, shared IBKR session |
| `qflow/portfolio_selector.py` | **Auto universe filter** — keep only robust (symbol, strategy) pairs by walk-forward |

The full quant walkthrough — covering strategy generation, backtesting,
risk/reward, regime detection, multi-factor models, optimization, portfolio
construction, trade setups, Monte-Carlo, drawdowns, macro overlays, and edge
detection — is in **[`docs/PLAYBOOK.md`](docs/PLAYBOOK.md)**.

---

## Quick start

```bash
pip install -r requirements.txt        # numpy + pandas

python examples/research/run_all.py             # full end-to-end demo (synthetic)
python examples/research/compare_strategies.py  # backtest all strategies on REAL data
python examples/research/find_edges.py          # hunt daily inefficiencies + walk-forward tuning
python examples/research/repeatable_edges.py    # rank patterns by year-to-year repeatability
python examples/research/scan_universe.py       # scan a whole universe + España<->US dual listings
python examples/research/walk_forward.py        # rolling calendar walk-forward (train N yrs -> test next)
python examples/research/select_portfolio.py    # auto-filter a basket to robust (symbol, strategy) pairs
python examples/research/backtest_intraday.py   # daily strategies on 5m bars (they lose to costs)
python examples/research/intraday_lab.py        # intraday strategies: pick the interval + walk-forward
python tests/test_framework.py         # 53 correctness tests
```

### Finding daily edges & optimising

`examples/research/find_edges.py` scans real data for recurring daily inefficiencies and
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
python examples/live/paper_trade.py --strategy mean_reversion --symbol AAPL --reset --replay 15

# the daily routine (run after the close; automate with cron)
python examples/live/paper_trade.py --strategy mean_reversion --symbol AAPL --step --report
```

It journals every fill, tracks the equity curve, and prints a **go-live
readiness gate**. Full plan: **[`docs/PAPER_TRADING.md`](docs/PAPER_TRADING.md)**.

### News-aware trading (optional)

A provider-agnostic news + sentiment layer can veto or down-size trades that
fight fresh adverse headlines. Free RSS by default; Bloomberg is one optional
adapter (only worth it if you already pay for a Terminal — see the honest
cost analysis in **[`docs/NEWS.md`](docs/NEWS.md)**).

```bash
python examples/research/news_demo.py                                   # offline demo
python examples/live/paper_trade.py --symbol AAPL --news rss --step # live overlay
python examples/live/paper_trade.py --symbol AAPL --news rss --finbert --step  # FinBERT sentiment
```

### Kill-switches (risk governor)

Account-level circuit breakers — daily-loss, max-drawdown, portfolio-heat and
loss-streak halts — that veto new entries when things go wrong, with state that
persists across daily runs. Details: **[`docs/RISK.md`](docs/RISK.md)**.

```bash
python examples/live/paper_trade.py --symbol TSLA --kill-switches --max-drawdown 0.08 --step
```

### Live execution (Interactive Brokers)

A broker layer turns signals into real orders. `IBKRBroker` submits **native
bracket orders** — every entry carries a stop-loss and take-profit that live on
IBKR's servers (honoured even if your script dies). Defaults to the **paper
port**; live ports require `allow_live=True`. Backtests never send orders. Concepts:
**[`docs/BROKER.md`](docs/BROKER.md)** · step-by-step IB Gateway setup:
**[`docs/SETUP_IBKR.md`](docs/SETUP_IBKR.md)**.

```bash
# forward-test through IB Gateway paper (port 4002) with kill-switches + news
python examples/live/paper_trade.py --symbol AAPL --source yahoo --strategy mean_reversion \
    --risk 0.005 --kill-switches --broker ibkr --ibkr-port 4002 --news rss --step
```
```python
from qflow.paper import PaperTrader
from qflow import broker
pt = PaperTrader("AAPL", source="yahoo", strategy="mean_reversion",
                 risk_limits={"max_drawdown": 0.08},
                 broker=broker.IBKRBroker(port=4002))   # 4002 IB Gateway paper
pt.step()    # the only path that places real orders
```

### Run a whole basket (grouped execution)

`portfolio_run.py` trades a basket of volatile names at once with the adaptive
`auto` strategy, sharing one IBKR session, with per-symbol + portfolio-level
kill-switches. Details: **[`docs/PORTFOLIO.md`](docs/PORTFOLIO.md)**.

```bash
python examples/research/backtest_portfolio.py     # validate the basket (in/out-of-sample)
python examples/live/portfolio_run.py --symbols TSLA,NVDA,AMD,COIN,PLTR --source yahoo \
    --strategy auto --risk 0.005 --kill-switches \
    --broker ibkr --ibkr-port 4002 --news rss --loop
# optional intraday gap-fade on the same basket (two scheduled phases):
python examples/live/gap_fade_routine.py --phase open  --symbols TSLA,NVDA,AMD --broker ibkr
python examples/live/gap_fade_routine.py --phase close --symbols TSLA,NVDA,AMD --broker ibkr
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
├── qflow/               # the shared engine (data, feeds, strategies, backtest, risk, broker, ...)
├── bot/
│   └── intraday_bot.py  # THE FUNDED-ACCOUNT INTRADAY BOT (product 2)
├── examples/
│   ├── research/        # backtest · walk-forward · edge-finding · universe selection (the lab)
│   └── live/            # daily-swing paper/portfolio runners + IBKR utilities
├── tests/               # test_framework.py — 53 correctness checks
├── data/samples/        # bundled REAL data (AAPL, TSLA, NVDA, AMD, NFLX, AMZN, MSFT, GOOGL)
├── docs/                # ROADMAP · FUNDED · PLAYBOOK · EDGES · PORTFOLIO · RISK · BROKER · SETUP_IBKR · PAPER_TRADING · NEWS · DISCLAIMER
├── requirements.txt
└── LICENSE              # MIT
```

## License

MIT — see [`LICENSE`](LICENSE). Provided "as is", without warranty of any kind.
