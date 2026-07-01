# Funded-account mode — passing a prop-firm exam (greedy but capped)

> **Educational use only — not financial advice.** See [`DISCLAIMER.md`](DISCLAIMER.md).

The bot can run itself against a prop-firm challenge: it self-selects a strategy
per symbol, sizes trades to press toward the profit target while there's cushion,
and de-risks *before* it can breach a loss limit. This doc explains the engine
(`qflow/funded.py`), the **FundedNext** (MT5, two-way) and **Lucid** (futures,
webhook) presets and connectors, the per-trade journal, and the selection cache.

**Which firm?** FundedNext (CFD/Forex on MT5) is the recommended path — MT5 gives
a real two-way API. Lucid (futures) is reachable one-way via a webhook bridge.

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

## FundedNext preset + MetaTrader 5 (the recommended path)

FundedNext is a **CFD/Forex** firm on **MT4 / MT5 / cTrader** — and MT5 has an
official Python package, so this is a **two-way** connection: the bot places
orders *and* reads real positions / balance / equity back (unlike the one-way
webhook). `FundedAccount.from_fundednext(model)` sets the rules:

| Model | Profit target | Daily loss | Overall DD (static) | Min days |
|---|---|---|---|---|
| `stellar_1step` | 10% | 3% | 6% | 2 |
| `stellar_2step_p1` (phase 1) | 8% | 5% | 10% | 5 |
| `stellar_2step_p2` (phase 2) | 5% | 5% | 10% | 5 |
| `express` | 25% | 5% | 10% | 10 |

Overall drawdown is **static** (from the initial balance); the daily loss resets
each day. No consistency rule by default. Confirm your model's exact target on the
FundedNext dashboard and override with `--profit-target` if needed.

### Connect via MetaTrader 5 (`MT5Broker`, `--broker mt5`)

1. `pip install MetaTrader5` (Windows; on Linux/Mac run MT5 under Wine).
2. Install the **MT5 terminal**, log into your FundedNext account, and enable
   **Algo Trading** (the terminal must stay open while the bot runs).
3. Trade FundedNext broker symbols (`EURUSD`, `XAUUSD`, `US30`, …).

**Automatic lot sizing.** The bot sizes each trade in **lots** so that hitting the
ATR stop loses the funded engine's target risk fraction of your **real equity**,
using the symbol's tick economics from MT5 (`MT5Broker.size_for_risk`):

```
loss for 1.0 lot = (stop_distance / trade_tick_size) × trade_tick_value
lots             = (equity × risk%) / loss_per_lot   (rounded to volume_step,
                                                       clamped to min/max lot)
```

So every trade risks the same fraction of equity whatever the instrument, and the
greedy-but-capped throttle scales the lots up/down with the cushion. Pass
`--fixed-qty 0.10` only if you want to **override** the sizer with a fixed lot.

```bash
python bot/intraday_bot.py --broker mt5 \
    --mt5-login 123456 --mt5-password "***" --mt5-server FundedNext-Server \
    --fundednext stellar_2step_p1 --auto-select \
    --symbols EURUSD,XAUUSD --journal logs/fn.csv
```

Because MT5 reports equity back, the funded engine tracks the **real** account:
the exam status line, greedy-but-capped sizing and the pass/fail auto-flatten all
work against your live FundedNext balance. `SL`/`TP` are attached to each order
and enforced by the broker.

**Data comes from MT5 too — no Yahoo.** When `--broker mt5`, the bot reads bars
straight from the MT5 terminal's feed (`MT5Broker.bars` → `copy_rates_from_pos`),
i.e. the **same FundedNext price data it trades on**. Yahoo is only a fallback for
brokers with no data feed (e.g. the webhook). So one MT5 connection gives the bot
everything: prices in, strategy applied, orders out — exactly as you described.

> TradeLocker / Match-Trader (FundedNext's other platforms) don't have a Python
> API; reach those through the `WebhookBroker` + a bridge (below).

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

## Connecting to a real Lucid account

**Important:** Lucid is a **futures** prop firm. It does **not** use Interactive
Brokers. It runs on futures data feeds (**Rithmic** / **CQG**) and platforms
(NinjaTrader, Tradovate, TradingView, Quantower, Sierra Chart). Connecting needs a
futures execution path and futures instruments.

Three ways to reach it, easiest first:

| Connector | Effort | Fills/equity back? | Status |
|---|---|---|---|
| **Webhook bridge** (TradersPost / CrossTrade → Tradovate) | lowest | no (one-way) | ✅ built: `WebhookBroker` |
| **Tradovate API** (REST + WebSocket) | medium | yes | next |
| **Rithmic API** (R\|Protocol, protobuf) | high | yes | later |

### The easy connector (built): `WebhookBroker`

`qflow/broker.py`'s `WebhookBroker` POSTs TradersPost-compatible JSON
(`buy`/`sell`/`exit` with `stopLoss`/`takeProfit`) to a bridge URL, which routes
the order into your Tradovate/Lucid account. It reuses the whole bot unchanged —
only `--broker webhook` and a URL differ.

> One-way limitation: a plain webhook can't report fills/positions/equity back, so
> `WebhookBroker` *shadow-tracks* what it sent to keep the bot/journal/funded
> engine running. Treat its equity as an estimate — the real truth is the Lucid
> dashboard until the Tradovate API adapter (two-way) lands.

Setup steps:

1. Make a Lucid **evaluation/demo** account and connect it to **Tradovate**.
2. Create a **TradersPost** (or CrossTrade) account, connect it to that Tradovate
   account, and create a strategy — it gives you a **webhook URL**.
3. Map the symbols you'll trade (e.g. `MES`, `MNQ`) in TradersPost.
4. Dry-run first (builds payloads, sends nothing), then go live:

```bash
# 1) inspect the payloads without sending
python bot/intraday_bot.py --broker webhook --webhook-url "https://webhooks.traderspost.io/..." \
    --lucid 50 --symbols MES,MNQ --fixed-qty 1 --journal logs/lucid.csv --webhook-dry-run

# 2) same command without --webhook-dry-run to actually route orders to Lucid
```

Use `--fixed-qty` to send a fixed number of **contracts** (futures are sized in
contracts, not shares, so the equity/ATR share-sizer is bypassed for now).

### Then: two-way Tradovate API (next)

Add a `TradovateBroker` next to `IBKRBroker` implementing the same interface
(`connect`, `account`, `positions`, `place_bracket`, `flatten`, `cancel_all`)
using Tradovate's REST + WebSocket. That gives real fills/positions/equity back so
the funded engine tracks the true account, and proper **contract sizing** by tick
value (`FundedAccount.risk_fraction()` × equity ÷ per-contract stop $) replacing
the equity `size()`.

### Validate first

Run `--broker webhook --lucid <size>` on the **evaluation/demo** account for 2–4
weeks, watch the journal's win-rate/expectancy and the exam status line, and only
then take a funded account at minimum size. Green paper ≠ profitable — it means
the machine works. See [`ROADMAP.md`](ROADMAP.md).
