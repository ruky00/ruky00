# Portfolio — trade a basket of symbols at once

> **Educational use only — not financial advice.** Start on paper, keep the
> kill-switches on, validate before risking capital. See [`DISCLAIMER.md`](DISCLAIMER.md).

Instead of babysitting one ticker, point the bot at a **basket of volatile names**.
`qflow/portfolio_runner.py` runs one strategy (default the adaptive `auto`, which
picks the best sub-strategy per candle) across all of them, **sharing one IBKR
session**, with per-symbol kill-switches plus a portfolio-level drawdown breaker.

## The three pieces

| Goal | Command |
|------|---------|
| **Backtest** the basket first | `python examples/research/backtest_portfolio.py` |
| **Run** the basket (swing, at close) | `python examples/live/portfolio_run.py ...` |
| **Gap-fade** (intraday, volatile names) | `python examples/live/gap_fade_routine.py --phase open/close ...` |

## 1. Validate the basket (backtest + out-of-sample)

```bash
python examples/research/backtest_portfolio.py     # edit BASKET; use 10y yahoo data
```
Reports per-symbol and **aggregate** CAGR/Sharpe/maxDD, the `auto` strategy-usage
mix, and an **in-sample vs out-of-sample** split. The portfolio max-drawdown is
usually smaller than any single name — that diversification is the whole point.
Only proceed if OOS Sharpe stays positive and near in-sample.

## 2. Run the basket (the grouped live command)

```bash
# offline preview
python examples/live/portfolio_run.py --symbols AAPL,TSLA --source github \
    --broker paper --reset --replay 300 --kill-switches

# live: 5 volatile names through IB Gateway paper, auto strategy, loop
python examples/live/portfolio_run.py --symbols TSLA,NVDA,AMD,COIN,PLTR --source yahoo \
    --strategy auto --risk 0.005 --allocation equal \
    --kill-switches --portfolio-max-drawdown 0.15 \
    --broker ibkr --ibkr-port 4002 --news rss --loop --loop-interval 3600
```

- **One shared IBKR session** for the whole basket (IBKR allows only one).
- Capital split across symbols: `--allocation equal` or `inverse_vol`.
- `--risk` is per-trade, per-symbol (start at 0.5%).
- `--loop` steps every interval; idempotent, so each symbol trades once per new
  daily bar. `report()` shows each symbol's active strategy + a portfolio total.
- **Two circuit breakers**: per-symbol (`--kill-switches`) and portfolio-wide
  (`--portfolio-max-drawdown`, latches a halt on total-equity drawdown).
- **Correlation filter** (`--max-correlation`, default 0.85): blocks a new entry
  in a name that is highly correlated with a position you already hold, so you
  don't stack the same bet under different tickers (e.g. two semis).
- **Strategy choice**: `--strategy auto` (regime picks per bar, fixed params) or
  `--strategy auto_wf` (same, but each sub-strategy uses per-year walk-forward-
  optimised params — heavier, re-tunes yearly).

Spanish names (data on Yahoo `.MC`, order on Madrid in EUR) work too — pass
`--currency EUR --primary BM` and a `--symbols` list of `.MC` tickers.

## 3. Gap-fade routine (optional, intraday, flat overnight)

Two scheduled phases per day on the volatile basket:

```bash
python examples/live/gap_fade_routine.py --phase open  --symbols TSLA,NVDA,AMD \
    --source yahoo --broker ibkr --ibkr-port 4002       # ~15:35 Spain (US open)
python examples/live/gap_fade_routine.py --phase close --symbols TSLA,NVDA,AMD \
    --source yahoo --broker ibkr --ibkr-port 4002       # ~21:55 Spain (US close)
```
`open` fades each opening gap with a market order (buy gap-downs, short gap-ups)
and records state; `close` flattens everything. Nothing is held overnight.

## Scheduling on Windows (Task Scheduler)

| Task | Trigger (Spain) | Action |
|------|-----------------|--------|
| Swing basket | daily 22:05 | `portfolio_run.py --step ...` (after US close) |
| Gap-fade open | weekdays 15:35 | `gap_fade_routine.py --phase open ...` |
| Gap-fade close | weekdays 21:55 | `gap_fade_routine.py --phase close ...` |

IB Gateway must be running and logged in (paper) for all of them. For unattended
24/7, add IBC to auto-login IB Gateway.

## Honest expectations

- **Diversification across names helps more than tuning one strategy.** A basket
  of 5+ uncorrelated volatile names is the single biggest robustness win.
- `auto` aims to be *steadier across regimes*, not the highest return.
- Short single-name backtests look mediocre; that is expected. Validate on 10y and
  the exact names you'll trade, then **paper-test for 1–2 weeks** and only go live
  (`--ibkr-port 4001 --ibkr-allow-live`, minimum size) if the readiness gates hold.
