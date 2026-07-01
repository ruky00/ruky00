# Funded-account mode — passing a prop-firm exam (greedy but capped)

> **Educational use only — not financial advice.** See [`DISCLAIMER.md`](DISCLAIMER.md).

The bot can run itself against a prop-firm challenge: it self-selects a strategy
per symbol, sizes trades to press toward the profit target while there's cushion,
and de-risks *before* it can breach a loss limit. This doc explains the engine
(`qflow/funded.py`), the Lucid Trading preset, the per-trade journal, the
selection cache, and the concrete steps to connect to a real Lucid account.

## The engine (`qflow/funded.py`)

`FundedAccount` encodes a challenge as hard limits and drives sizing:

| Rule | Meaning |
|---|---|
| `profit_target` | reach +X% → **PASS** (locks; stops opening new risk) |
| `max_daily_loss` | lose > Y% from the day's start → **FAIL** |
| `max_total_drawdown` | equity falls > Z% below the sliding high-water base → **FAIL** |
| `drawdown_mode` | `static` (floor = start − allowance) · `trailing` (intraday peak) · `eod` (highest **end-of-day** balance) |
| `consistency_pct` | your single best day must be ≤ this share of total profit |
| `min_trading_days` | must trade at least N distinct days |

**Greedy but not reckless — dynamic sizing.** Per-trade risk = `base_risk ×`
multiplier, where the multiplier scales with the *tighter* of the daily / total
loss cushions:

- Both cushions full → up to `max_risk_mult` (default 1.5×) — press toward target.
- Approaching a limit → smoothly down toward `min_risk_mult` (0.25×).
- `stop_buffer` (default 0.80) of an allowance consumed → **stop opening** (daily
  → for today; total → until done) so it de-risks *before* a breach, never after.
- Target hit → pass locked, no new risk.
- Consistency: once today's profit already fills the allowed share of total
  profit, stop adding to it — spread gains across days.

```
eq 100000  risk 0.60%   (full cushion → 0.4% × 1.5 boost, greedy)
eq  99000  risk 0.50%   (throttling down)
eq  97000  risk 0.30%   (near the daily limit)
eq  95800  BLOCKED      (84% of the 5% daily allowance used → stop for today)
eq 108500  BLOCKED      (+8.5% ≥ target → pass locked)
```

## Lucid Trading preset

Lucid is a **futures** prop firm with **EOD trailing drawdown** and (on the
LucidFlex evaluation) a **50% consistency rule**. `FundedAccount.from_lucid(size)`
sets it up from the account size:

| Account | Profit target | Trailing DD (EOD) | Daily loss* |
|---|---|---|---|
| 25K | $1,250 | $1,000 | none |
| 50K | $3,000 | $2,000 | ~$1,200 |
| 100K | $6,000 | $3,000 | ~$2,000 |
| 150K | $9,000 | $4,500 | ~$2,700 |

- **EOD trailing**: the drawdown floor only moves at the daily close, tracking the
  highest end-of-day balance − allowance; intraday spikes don't move it. Once you
  reach target it converts to static.
- **Consistency (eval)**: largest single day ≤ 50% of total profit; removed once funded.
- No time limit.

`*` Daily-loss figures are **approximate** — confirm the exact DLL for your plan
and override with `--max-daily-loss`. Target and trailing drawdown are well-sourced.

Run it:

```bash
# fully autonomous Lucid 50K evaluation on paper, self-selecting per symbol,
# journaling every trade, caching the walk-forward selection for 24h
python bot/intraday_bot.py --auto-select --lucid 50 --allow-short \
    --symbols MES,MNQ \
    --select-cache logs/select.json --journal logs/lucid50.csv
```

The live status line shows: `profit +X%/6% · day-loss X% · DD-floor $48,000 ·
cushion day/total X%/X% · days N/1 · ⚠️ consistency` (if the best-day rule is at risk).

## Per-trade journal (`qflow/journal.py`)

`--journal PATH` appends one CSV row per entry (with strategy, interval, risk%,
SL/TP) and one per exit (approximate P&L from the last mark). Read it back for a
live win-rate / average-win / expectancy — the evidence that the edge is real
before risking real evaluation fees. At shutdown the bot prints the summary.

## Selection cache (`--select-cache`)

Walk-forwarding every symbol at each startup is slow. `--select-cache logs/x.json`
stores each symbol's chosen (strategy, interval) with a timestamp and reuses it
until `--reselect-hours` (default 24) elapses, then re-runs the walk-forward and
rewrites the cache.

## Next steps — connecting to a real Lucid account

**Important:** Lucid is a **futures** prop firm. It does **not** use Interactive
Brokers. It runs on futures data feeds (**Rithmic** / **CQG**) and platforms
(NinjaTrader, Tradovate, TradingView, Quantower, Sierra Chart). The current bot
speaks to IBKR (equities), so connecting to Lucid needs a futures execution path
and futures instruments. Concretely:

1. **Pick the connection.**
   - **Tradovate API** (REST + WebSocket) — LucidFlex uses the Tradovate data
     connection and Tradovate has a documented API; best fit for native Python.
   - **Webhook bridge** (TradersPost / CrossTrade) — lowest-code: the bot POSTs a
     signal and the bridge routes the order into Tradovate/NinjaTrader. Good for a
     first integration.
   - **Rithmic API** (R|Protocol, protobuf) — used by NinjaTrader/LucidBlack; more
     work but the most direct fills.
2. **Add a broker adapter** next to `IBKRBroker` (`qflow/broker.py`) implementing
   the same interface (`connect`, `account`, `positions`, `place_bracket`,
   `flatten`, `cancel_all`) — e.g. `TradovateBroker` or `WebhookBroker`. The bot
   loop, funded engine, journal and selection are broker-agnostic and reuse as-is.
3. **Switch the universe to futures.** Trade `MES`/`MNQ` (micros) or `ES`/`NQ`,
   pull 5-minute futures bars, and **size in contracts** using each contract's
   tick value and the dollar risk (`FundedAccount.risk_fraction()` × equity ÷
   per-contract stop $), not share notional. This replaces the equity `size()`.
4. **Validate first.** Run `--auto-select --lucid <size>` on the **evaluation /
   paper** account for 2–4 weeks, watch the journal's win-rate/expectancy and the
   exam status, and only then take a funded account at minimum size.

The discipline gate still applies: green paper ≠ profitable — it means the machine
works. See [`ROADMAP.md`](ROADMAP.md).
