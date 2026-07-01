# Roadmap — what we have, where we are, where we're going

> **Educational use only — not financial advice.** See [`DISCLAIMER.md`](DISCLAIMER.md).

## Two products, one codebase

| | **Product 1 — The Lab** | **Product 2 — The Funded Bot** |
|---|---|---|
| What | Research & validation engine (daily swing) | Live intraday earner on a funded/IBKR account |
| Where | `qflow/` + `examples/research/` + `examples/live/` | `bot/intraday_bot.py` |
| Job | Find & validate robust (symbol, strategy) edges | Trade the validated strategies intraday for recurring P&L |
| Data | 5–10y daily bars (backtest) / daily live | 5-minute bars, live |
| Horizon | Days–weeks (swing) | Minutes–hours (flat overnight) |
| Output | Backtests, walk-forward, shortlists, paper journal | Live orders with ATR SL/TP, per-position P&L |

The Lab decides **what is worth trading**; the Bot **trades it**. Same strategies
(`qflow.strategies`), same risk engine, same broker layer — different timeframe
and purpose.

---

## ✅ What we have (done)

**The Lab (product 1)**
- Real data feeds (Yahoo / Stooq / Binance + bundled offline samples).
- Strategies: trend-following, mean-reversion, volatility-breakout, and the
  regime-adaptive `auto` / `auto_wf` (per-year walk-forward-tuned).
- Backtester (no look-ahead, costs), metrics, Monte-Carlo, drawdown analysis.
- Edge discovery: overnight/gap/day-of-week/lead-lag scanner + repeatability lab.
- **Rolling walk-forward** + **portfolio_selector** (auto-filter a basket to the
  robust (symbol, strategy) pairs — drops the losers, keeps the winners).
- Multi-symbol **portfolio runner** with shared IBKR session, capital allocation,
  correlation filter and a portfolio-level drawdown circuit breaker.
- News + sentiment overlay (RSS/Finnhub/Bloomberg/FinBERT).
- Paper-trading engine (persistent forward test + go-live readiness gate).
- 36 passing tests.

**The Funded Bot (product 2)**
- `bot/intraday_bot.py`: runs qflow strategies on 5-minute bars, ATR SL/TP
  bracket orders that live on IBKR, risk-based sizing, per-position SL/TP + P&L
  display, account-level kill-switches.
- IBKR execution: bracket orders, free delayed market data, real order-status
  reporting, TIF/preset handling, client-id isolation.

## 📍 Where we are (now)
- End-to-end **works on IBKR paper**: signals → orders with SL/TP → fills saved
  in IBKR. The intraday bot trades live on the paper account.
- Both products share the same strategy/risk/broker code.

## 🚀 Where we're going (next)

**Short term — make the funded bot trustworthy**
1. **Per-trade journal → CSV** for the bot (entry/exit/SL/TP/P&L per trade) to
   review and compute live win-rate / expectancy.
2. **Intraday backtester** on 5-minute bars (realistic costs/slippage) so the
   intraday strategies are *validated* the way the daily ones are — today they're
   validated on daily data, which is only a proxy.
3. **Session controls**: auto-flat before the close, no new entries in the last
   30 min, max trades/day, max concurrent positions.
4. **Funded-account rules engine**: encode the prop-firm limits (max daily loss,
   max total drawdown, profit target) as hard kill-switches so the bot can never
   breach them.

**Medium term — robustness & automation**
5. **Real-time data path**: pull 5m bars from IBKR directly (not delayed Yahoo)
   for tighter fills; make delayed-vs-realtime explicit.
6. **Auto-restart / 24-5 operation**: IBC to auto-login IB Gateway + a supervisor
   that restarts the bot and reconciles positions daily.
7. **Walk-forward the intraday strategies** and run `portfolio_selector` on the
   intraday timeframe to pick the intraday (symbol, strategy) shortlist.
8. **Broker abstraction**: add Alpaca / ccxt adapters behind the same interface
   (crypto runs 24/7 and has no PDT/borrow constraints — a natural funded-bot fit).

**Longer term — edge & scale**
9. Smarter execution (limit/adaptive orders, VWAP entries) to cut slippage.
10. Portfolio of *strategies* (not just symbols): allocate capital across the
    edges the lab keeps validating, rebalanced automatically.
11. Live monitoring dashboard + alerts (Telegram/email) for fills, halts, drawdown.

## The path to going live (discipline gate)
```
Lab: backtest + walk-forward + selector  →  shortlist of robust (symbol, strategy)
   →  paper forward-test (readiness = GO)  →  funded bot on PAPER 2-4 weeks
   →  funded account, MINIMUM size, kill-switches + funded-rules engine ON
```
Never skip a step. Green paper ≠ profitable — it means the machine works. The edge
evidence is the walk-forward + selector, and even that is a hypothesis until it
survives live.
