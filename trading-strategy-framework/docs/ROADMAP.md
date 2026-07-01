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
- **Dedicated intraday strategies** (VWAP reversion, opening-range breakout,
  intraday momentum, and the `intraday_auto` combiner) with per-strategy
  parameter grids, an interval sweep (5m/15m/30m via `data.resample_ohlcv`) and
  intraday walk-forward (`examples/research/intraday_lab.py`).
- 49 passing tests.

**The Funded Bot (product 2)**
- `bot/intraday_bot.py`: runs qflow strategies on 5-minute bars, ATR SL/TP
  bracket orders that live on IBKR, risk-based sizing, per-position SL/TP + P&L
  display, account-level kill-switches.
- IBKR execution: bracket orders, free delayed market data, real order-status
  reporting, TIF/preset handling, client-id isolation.
- **Autonomous mode (`--auto-select`)**: the bot picks its own
  (strategy, interval) per symbol by walk-forwarding the intraday strategies
  out-of-sample at startup — no manual strategy choice.
- **Funded-exam engine (`--funded`, `qflow/funded.py`)**: encodes the prop-firm
  challenge (profit target + hard daily-loss / total-drawdown limits, static or
  trailing) with **dynamic greedy-but-capped sizing** — full/boosted size while
  there's cushion, throttling down as it nears a limit, stopping *before* it can
  breach, and locking the pass once the target is hit.

## 📍 Where we are (now)
- End-to-end **works on IBKR paper**: signals → orders with SL/TP → fills saved
  in IBKR. The intraday bot trades live on the paper account.
- Both products share the same strategy/risk/broker code.

## 🚀 Where we're going (next)

**Short term — make the funded bot trustworthy**
1. ✅ **Per-trade journal → CSV** (`qflow/journal.py`, `--journal`): entry/exit,
   SL/TP, risk%, strategy/interval per trade → live win-rate / avg-win / expectancy.
2. ✅ **Intraday backtester** on 5-minute bars with end-of-day flattening
   (`run_backtest(..., flatten_eod=True)`, `examples/research/backtest_intraday.py`).
   First finding: the daily strategies run *naively* on 5m bars **lose money**
   (costs eat the many small trades).
2b. ✅ **Dedicated intraday strategies + interval selection + walk-forward**
   (`qflow/intraday_strategies.py`, `examples/research/intraday_lab.py`): VWAP
   reversion, opening-range breakout, intraday momentum and `intraday_auto`,
   swept across 5m/15m/30m and validated out-of-sample. Finding on synthetic
   data: coarser bars (30m) bleed far less to costs than 5m, and only the
   VWAP-reversion edge survives the walk-forward — a template to re-run on real
   5m bars for the names the bot will trade.
3. **Session controls**: auto-flat before the close, no new entries in the last
   30 min, max trades/day, max concurrent positions.
4. ✅ **Funded-account rules engine** (`qflow/funded.py`, `--funded` / `--lucid`):
   profit target + daily-loss + total-drawdown (static / trailing / **EOD
   trailing**) + **consistency rule**, greedy-but-capped dynamic sizing that
   de-risks before it can breach and locks the pass at target. Ships a **Lucid
   Trading preset** (`FundedAccount.from_lucid`, 25/50/100/150K). See
   [`FUNDED.md`](FUNDED.md).

**Medium term — robustness & automation**
5. **Real-time data path**: pull 5m bars from IBKR directly (not delayed Yahoo)
   for tighter fills; make delayed-vs-realtime explicit.
6. **Auto-restart / 24-5 operation**: IBC to auto-login IB Gateway + a supervisor
   that restarts the bot and reconciles positions daily.
7. ✅ **Walk-forward the intraday strategies** and let the bot self-select the
   intraday (symbol, strategy, interval) shortlist (`qflow/intraday_select.py`,
   `--auto-select`). *Next:* cache the selection to disk + re-select on a
   schedule instead of every startup.
8. **Futures broker for Lucid**: Lucid is a *futures* prop firm (Tradovate /
   Rithmic / CQG — not IBKR). ✅ **Webhook connector built** (`WebhookBroker`,
   `--broker webhook`) to route orders to Lucid via TradersPost/CrossTrade —
   one-way for now. *Next:* a two-way `TradovateBroker` (REST + WebSocket) for
   real fills/equity back and **contract sizing** by tick value; then Alpaca /
   ccxt for equities / crypto. See [`FUNDED.md`](FUNDED.md).

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
