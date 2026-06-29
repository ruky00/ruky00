# Paper-Trading Runbook — from backtest to (maybe) live

> **Educational use only — not financial advice.** Paper trading checks your
> *plumbing and discipline*. One or two weeks of daily bars is **far too little
> data to prove an edge**. Your statistical evidence is the long backtest +
> Monte-Carlo; the forward test confirms the system behaves in real time before
> any real money is at risk. See [`DISCLAIMER.md`](DISCLAIMER.md).

This is the exact two-week plan you asked for: trade a chosen strategy with
**virtual $10,000** on **real data**, journal every fill, and only consider a
real account once a conservative readiness gate passes.

---

## The pipeline

```
1. Pick a candidate   →  examples/compare_strategies.py   (backtest on real data)
2. Stress it          →  Monte-Carlo + drawdown (docs/PLAYBOOK.md §9, §10)
3. Forward test       →  examples/paper_trade.py  (this doc — 1–2 weeks)
4. Readiness gate     →  pt.readiness() / --report
5. Go live small      →  only if gates pass, minimum size, real broker adapter
```

## Step 1 — choose the candidate (already done on real data)

```bash
python examples/compare_strategies.py
```

On the bundled real samples (AAPL, TSLA), `mean_reversion` is the only strategy
with a positive Sharpe and a tiny drawdown — so it's the natural candidate.
Re-run with your own instrument/timeframe before committing:

```python
from qflow import feeds
feeds.get("BTCUSDT", "binance", interval="1d", refresh=True)   # crypto
feeds.get("AAPL",    "yahoo",   rng="10y")                     # 10y equities
feeds.get("eurusd",  "stooq")                                  # FX
```

## Step 2 — forward test with fake money (run daily for 1–2 weeks)

Run **once per day, after the close**. State persists between runs under
`data/paper/<symbol>_<strategy>/`, so each run resumes the same virtual account.

```bash
# daily: pull the latest bar, act on it, then show status
python examples/paper_trade.py --strategy mean_reversion --symbol AAPL --step
python examples/paper_trade.py --strategy mean_reversion --symbol AAPL --report
```

Automate it with cron (weekdays at 17:30):

```cron
30 17 * * 1-5  cd /path/to/trading-strategy-framework && \
  /usr/bin/python3 examples/paper_trade.py --strategy mean_reversion --symbol AAPL --step \
  >> data/paper/cron.log 2>&1
```

**Preview the whole two weeks right now** (replays recent history, one bar = one
simulated day) so you can see the journal, equity curve and readiness output
without waiting:

```bash
python examples/paper_trade.py --strategy trend_following --symbol TSLA --reset --replay 15
```

### What gets recorded
| File | Contents |
|------|----------|
| `state.json` | full account state (resumes the forward test) |
| `journal.csv` | every OPEN/CLOSE with price, shares, reason, pnl, equity |
| `equity.csv` | daily equity curve |

### What to watch each day
- **Did it fire the trades you expected?** Cross-check `journal.csv` against the
  chart. Mismatch = a data or signal bug, not an edge.
- **Is risk per trade ~1%?** Each stop-out should lose ≈$100 on a $10k account.
  The engine sizes by ATR so this is automatic — verify it.
- **Equity & drawdown** sane and matching backtest expectations.

## Step 3 — the go-live readiness gate

`--report` prints a 5-gate checklist (also available as `pt.readiness()`):

| Gate | Why |
|------|-----|
| Ran ≥ 10 sessions | enough live bars to trust the operations |
| No drawdown worse than −10% | catastrophic-risk circuit breaker |
| Account not in the red (> −2%) | the system isn't bleeding in real time |
| ≥ 3 closed trades | the logic actually executed round trips |
| Profit factor ≥ 1 (if traded) | losers aren't dwarfing winners |

Verdicts: **GO** (all gates) / **ALMOST** (one open) / **NOT READY**.

> These gates are necessary, not sufficient. Passing them means "the machine
> works and didn't embarrass itself" — **not** "this is profitable." Profit
> evidence comes from the multi-year backtest and Monte-Carlo, which must also
> be healthy.

## Step 4 — going live (only if everything passes)

1. **Start at minimum size** (or the smallest the broker allows). Risk 0.25–0.5%
   per trade for the first month, not 1–2%.
2. **Add a real broker adapter.** The engine is broker-agnostic; replace the
   simulated fill in `qflow/paper.py::_process_bar` with real order calls:
   - Crypto: [`ccxt`](https://github.com/ccxt/ccxt) (Binance/Bybit/…)
   - US equities: [`alpaca-py`](https://github.com/alpacahq/alpaca-py)
   Place a **bracket order** (entry + stop + take-profit) so the stop lives at
   the exchange, not just in your script.
3. **Keep the kill-switches**: max daily loss, max open risk (portfolio heat),
   and a "halt trading if equity < 50-day average" rule.
4. **Reconcile daily**: broker fills vs `journal.csv`. Any drift = stop and fix.

## Honest expectations

- On the bundled samples, the trend/breakout strategies **underperform buy &
  hold** and have negative Sharpe; mean-reversion is marginally positive. That
  is the realistic outcome of testing simple rules on a couple of single names
  over a short window. **Edge is hard.** Treat green paper results as a license
  to test bigger, not to bet big.
- Re-validate on the *exact* instrument, timeframe, fees and slippage you will
  trade. Costs and slippage kill more strategies than bad signals do.
```bash
python examples/paper_trade.py --strategy mean_reversion --symbol AAPL --report
```
